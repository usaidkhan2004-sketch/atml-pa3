"""Task 2 training: the three-stage SigLIP-Qwen curriculum (manual Steps 2-4) with development evaluation.

  Stage 1  caption alignment     2,250 updates  connector            two seeded passes over the 18,000 captions
  Stage 2  visual instruction    1,500 updates  connector + LoRA     draws: presence .50, counting .25, caption .25
  Stage 3  natural VQA             750 updates  connector + LoRA     draws: VQAv2 .70, presence .10, counting .10, caption .10

Shared settings follow the manual's table: microbatch 8 x accumulation 2, AdamW (weight decay 0.01),
global-norm clipping 1.0, per-stage 5% warmup then cosine decay to 10% of the stage's peak LR, gradient
checkpointing, FP16 autocast with dynamic loss scaling, frozen FP16 Qwen, FP32 trainable parameters and
optimizer states. The loss is example-weighted: each example's mean token NLL, averaged over the 16
examples of an update. In Stages 2-3 each draw adds two examples (a question pair, or two replay
captions); a microbatch holds four draws. Open points: DECISIONS.md T2-9 to T2-11 and T2-24 onward.

Outputs (under --out-root)
  task2/run/config.json, train_log.jsonl, dev_eval.jsonl, dev_examples/update_NNNN.json, summary.json
  task2/run/weights/stage{1,2,3}/connector.safetensors (+ lora.safetensors from Stage 2)
  checkpoints/task2/update_NNNN/   resumable state every 250 updates
Usage
  python task2/train.py                      # main run (on Kaggle: features kernel + splits dataset mounted)
  python task2/train.py --resume-from DIR    # continue from a checkpoint directory
  python task2/train.py --smoke              # 6/4/4-update run, then a resume test; writes task2/smoke/
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task2" / "train.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]
sys.path.insert(0, str(SELF.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import vlm as V
from check_interface import load, training_examples  # the example list the scale was calibrated on

SEED = V.SEED
MICROBATCH, ACCUM = 8, 2
EFFECTIVE = MICROBATCH * ACCUM
BETAS, EPS, WEIGHT_DECAY, GRAD_CLIP = (0.9, 0.999), 1e-8, 0.01, 1.0
LR_FLOOR = 0.1  # cosine decays to 10% of the stage's peak LR
DEV_BATCH = 8

STAGES = [
    {"name": "caption_alignment", "updates": 2250, "warmup": 113, "lr": {"connector": 1e-3}, "mix": None},
    {"name": "visual_instruction", "updates": 1500, "warmup": 75, "lr": {"connector": 1e-4, "lora": 2e-5},
     "mix": {"presence": 0.50, "counting": 0.25, "caption": 0.25}},
    {"name": "natural_vqa", "updates": 750, "warmup": 38, "lr": {"connector": 5e-5, "lora": 1e-5},
     "mix": {"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}},
]
MAIN = {"stages": STAGES, "checkpoint_every": 250, "dev_every": 500, "dev_limit": None}
SMOKE = {"stages": [dict(s, updates=n, warmup=1) for s, n in zip(STAGES, (6, 4, 4))],
         "checkpoint_every": 2, "dev_every": 4, "dev_limit": 16}


def stage_ends(cfg):
    return [int(x) for x in np.cumsum([s["updates"] for s in cfg["stages"]])]


def dev_points(cfg):
    ends = stage_ends(cfg)
    return sorted({0, *range(cfg["dev_every"], ends[-1] + 1, cfg["dev_every"]), *ends})


def locate(cfg, update):
    """(stage index, position within the stage) of 0-based update `update`."""
    start = 0
    for k, s in enumerate(cfg["stages"]):
        if update < start + s["updates"]:
            return k, update - start
        start += s["updates"]
    raise IndexError(update)


def lr_factor(stage, u):
    """Linear from 0 over the warmup updates, then cosine from 1 to LR_FLOOR over the rest of the stage
    (the Task 1 convention: the first update of a stage runs at factor 0)."""
    if u < stage["warmup"]:
        return u / stage["warmup"]
    p = (u - stage["warmup"]) / (stage["updates"] - stage["warmup"])
    return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * p))


# --------------------------------------------------------------------------- data

class Curriculum:
    """Training examples and the seeded sampler. Stage 1 order is two permutations drawn first from
    default_rng(6304); Stages 2-3 draw from the same generator, whose state is checkpointed."""

    def __init__(self, fmt, cfg):
        self.cfg = cfg
        ex = training_examples(fmt)
        pools = load("grounding_train_pools")
        n_cap, n_vqa, n_pre = 18_000, 2 * len(load("vqa_train")), 2 * len(pools["presence"])
        pairs = lambda xs: [xs[i:i + 2] for i in range(0, len(xs), 2)]
        self.all_examples = ex
        self.captions = ex[:n_cap]
        self.pairs = {"vqa": pairs(ex[n_cap:n_cap + n_vqa]),
                      "presence": pairs(ex[n_cap + n_vqa:n_cap + n_vqa + n_pre]),
                      "counting": pairs(ex[n_cap + n_vqa + n_pre:])}
        assert all(e["kind"] == "caption" for e in self.captions)
        assert [len(self.pairs[k]) for k in ("vqa", "presence", "counting")] == \
               [len(load("vqa_train")), len(pools["presence"]), len(pools["counting"])]
        self.rng = np.random.default_rng(SEED)
        self.stage1_order = np.concatenate([self.rng.permutation(n_cap) for _ in range(2)])
        assert len(self.stage1_order) >= cfg["stages"][0]["updates"] * EFFECTIVE

    def update(self, update):
        """The 16 examples of 0-based update `update`, as two microbatches, with one kind per example."""
        k, u = locate(self.cfg, update)
        if k == 0:
            exs = [self.captions[i] for i in self.stage1_order[u * EFFECTIVE:(u + 1) * EFFECTIVE]]
            kinds = ["caption"] * EFFECTIVE
        else:
            mix = self.cfg["stages"][k]["mix"]
            names = list(mix)
            exs, kinds = [], []
            for _ in range(EFFECTIVE // 2):
                kind = names[self.rng.choice(len(names), p=[mix[n] for n in names])]
                if kind == "caption":  # two independent caption examples (T2-25)
                    exs += [self.captions[i] for i in self.rng.integers(len(self.captions), size=2)]
                else:
                    exs += self.pairs[kind][self.rng.integers(len(self.pairs[kind]))]
                kinds += [kind, kind]
        return [exs[:MICROBATCH], exs[MICROBATCH:]], kinds


def dev_sets(fmt, limit=None):
    """Caption dev (96 images, one caption) and both question suites. Each question carries its
    pair's other image as the complementary input."""
    out = {"captions": [fmt.caption_example(c["image_id"], c["captions"][0]) for c in load("captions_dev")],
           "vqa": [], "grounding": []}
    for suite, name, qtype, ans in (("vqa", "vqa_dev", "answer_type", "target"),
                                    ("grounding", "grounding_dev", "type", "answer")):
        for p in load(name):
            a, b = p["items"]
            for it, other in ((a, b), (b, a)):
                out[suite].append({"pair_id": p["pair_id"], "type": p[qtype], "image_id": it["image_id"],
                                   "partner_image_id": other["image_id"],
                                   "example": fmt.qa_example(it["image_id"], p["question"], it[ans])})
    if limit:
        out = {k: v[:limit] for k, v in out.items()}
    return out


# --------------------------------------------------------------------------- model and optimizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def gb(x):
    return round(x / 2**30, 4)


def load_qwen():
    model = AutoModelForCausalLM.from_pretrained(V.MODEL_ID, revision=V.MODEL_REVISION, dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    V.freeze_backbone(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def param_groups(named, tag):
    """Weight decay on >= 2-D weights, none on biases (T2-9). `tag` selects the stage's peak LR."""
    decay = [p for n, p in named if p.ndim >= 2 and not n.endswith(".bias")]
    no_decay = [p for n, p in named if not (p.ndim >= 2 and not n.endswith(".bias"))]
    return [g for g in ({"params": decay, "weight_decay": WEIGHT_DECAY, "tag": tag},
                        {"params": no_decay, "weight_decay": 0.0, "tag": tag}) if g["params"]]


def lora_named(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def enable_lora(model, opt):
    """Stage 2 start: add LoRA to q/k/v/o and to the same AdamW as a new group (connector moments carry
    over, T2-24)."""
    V.add_lora(model)
    for g in param_groups(lora_named(model), "lora"):
        opt.add_param_group({**g, "lr": 0.0, "betas": BETAS, "eps": EPS})


def set_lr(opt, stage, u):
    f = lr_factor(stage, u)
    for g in opt.param_groups:
        g["lr"] = stage["lr"][g["tag"]] * f
    return {t: stage["lr"][t] * f for t in stage["lr"]}


# --------------------------------------------------------------------------- update and dev evaluation

def train_update(model, connector, opt, scaler, store, microbatches, kinds, params):
    opt.zero_grad(set_to_none=True)
    losses = []
    for exs in microbatches:
        b = V.to_device(V.build_batch(exs, [True] * len(exs)), "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            per_example = V.example_nll(model, connector, b, store.get(b["image_ids"]))
        scaler.scale(per_example.sum() / EFFECTIVE).backward()
        losses.append(per_example.detach().float().cpu())
    scaler.unscale_(opt)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP))
    scale_before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    losses = torch.cat(losses)
    rec = {"loss": float(losses.mean()), "grad_norm": grad_norm, "loss_scale": scale_before,
           "skipped": scaler.get_scale() < scale_before,
           "loss_by_kind": {k: float(losses[[i for i, x in enumerate(kinds) if x == k]].mean())
                            for k in dict.fromkeys(kinds)},
           "examples_by_kind": {k: kinds.count(k) for k in dict.fromkeys(kinds)},
           "image_ids": [e["image_id"] for exs in microbatches for e in exs]}
    return rec


@torch.no_grad()
def dev_eval(model, connector, store, dev):
    """Caption reference loss (the training loss: caption tokens + <|im_end|>) and reference-answer NLL
    (termination excluded) under correct, complementary and no-image inputs; per-example means.
    delta_nll = NLL(partner image) - NLL(correct image)."""
    model.eval()

    def nll(exs, with_image, end):
        out = []
        for s in range(0, len(exs), DEV_BATCH):
            b = V.to_device(V.build_batch(exs[s:s + DEV_BATCH], with_image[s:s + DEV_BATCH], supervise_end=end), "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(V.example_nll(model, connector, b, store.get(b["image_ids"])).float().cpu())
        return torch.cat(out)

    rec = {"caption_loss": float(nll(dev["captions"], [True] * len(dev["captions"]), True).mean())}
    per_example = {}
    for suite in ("vqa", "grounding"):
        items = dev[suite]
        n = len(items)
        correct = [it["example"] for it in items]
        partner = [{**it["example"], "image_id": it["partner_image_id"]} for it in items]
        c, p, z = nll(correct, [True] * n, False), nll(partner, [True] * n, False), nll(correct, [False] * n, False)

        def summary(idx):
            idx = torch.tensor(idx)
            return {"n": len(idx), "nll_correct": float(c[idx].mean()), "nll_complementary": float(p[idx].mean()),
                    "nll_no_image": float(z[idx].mean()), "delta_nll": float((p[idx] - c[idx]).mean())}

        rec[suite] = summary(list(range(n)))
        rec[suite]["by_type"] = {t: summary([i for i, it in enumerate(items) if it["type"] == t])
                                 for t in dict.fromkeys(it["type"] for it in items)}
        per_example[suite] = [{"pair_id": it["pair_id"], "type": it["type"], "image_id": it["image_id"],
                               "nll_correct": float(c[i]), "nll_complementary": float(p[i]),
                               "nll_no_image": float(z[i])} for i, it in enumerate(items)]
    model.train()
    return rec, per_example


# --------------------------------------------------------------------------- checkpoints

def save_weights(path, model, connector, with_lora):
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in connector.state_dict().items()},
              str(path / "connector.safetensors"))
    if with_lora:
        save_file(V.lora_state(model), str(path / "lora.safetensors"))


def save_checkpoint(path, done, model, connector, opt, scaler, cur, state):
    save_weights(path, model, connector, with_lora=locate_done_stage(cur.cfg, done) > 0)
    torch.save({"update": done, "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "sampler": cur.rng.bit_generator.state, "rng": rng_state(), **copy.deepcopy(state)},
               path / "trainer_state.pt")


def locate_done_stage(cfg, done):
    """Stage whose parameters exist after `done` completed updates (LoRA appears with Stage 2's first update)."""
    return 0 if done <= cfg["stages"][0]["updates"] else locate(cfg, done - 1)[0]


def load_checkpoint(path, model, connector, opt, scaler, cur):
    st = torch.load(path / "trainer_state.pt", weights_only=False)
    connector.load_state_dict(load_file(str(path / "connector.safetensors")))
    if (path / "lora.safetensors").exists():
        lora = load_file(str(path / "lora.safetensors"))
        for n, m in model.named_modules():
            if isinstance(m, V.LoRALinear):
                m.lora_A.data.copy_(lora[f"{n}.lora_A"])
                m.lora_B.data.copy_(lora[f"{n}.lora_B"])
    opt.load_state_dict(st["optimizer"])
    scaler.load_state_dict(st["scaler"])
    cur.rng.bit_generator.state = st["sampler"]
    set_rng_state(st["rng"])
    return st


# --------------------------------------------------------------------------- main loop

def run(args, cfg, out_root):
    out = out_root / "task2" / "run"
    ckpt_dir = out_root / "checkpoints" / "task2"
    for d in (out, out / "dev_examples", ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(V.MODEL_ID, revision=V.MODEL_REVISION)
    fmt = V.Formatter(tok)
    cur = Curriculum(fmt, cfg)
    dev = dev_sets(fmt, cfg["dev_limit"])
    store = V.FeatureStore(V.find_feature_dir(), device="cuda")
    model = load_qwen()

    connector = V.new_connector().cuda()
    scale, calib = V.calibration_scale(model, connector, cur.all_examples, store.feats["train"])
    recorded = json.loads((REPO / "task2/results/interface_checks.json").read_text())["calibration"]["scale"]
    assert abs(scale - recorded) / recorded < 1e-5, (scale, recorded)
    connector.scale.fill_(scale)

    opt = torch.optim.AdamW(param_groups(list(connector.named_parameters()), "connector"), lr=0.0,
                            betas=BETAS, eps=EPS)
    scaler = torch.amp.GradScaler("cuda")
    ends = stage_ends(cfg)
    total = ends[-1]
    state = {"train_log": [], "dev_log": [], "stages": {}}
    start = 0
    if args.resume_from:
        path = Path(args.resume_from)
        peek = torch.load(path / "trainer_state.pt", weights_only=False)
        if locate_done_stage(cfg, peek["update"]) > 0:
            enable_lora(model, opt)
        st = load_checkpoint(path, model, connector, opt, scaler, cur)
        start = st["update"]
        state = {k: st[k] for k in ("train_log", "dev_log", "stages")}
        print(f"resumed at update {start}")
    model.train()

    config = {"model": V.MODEL_ID, "model_revision": V.MODEL_REVISION, "seed": SEED, "microbatch": MICROBATCH,
              "grad_accum": ACCUM, "effective_batch": EFFECTIVE, "betas": BETAS, "eps": EPS,
              "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "lr_floor": LR_FLOOR, "stages": cfg["stages"],
              "checkpoint_every": cfg["checkpoint_every"], "dev_points": dev_points(cfg), "dev_limit": cfg["dev_limit"],
              "lora": V.LORA, "caps": V.CAPS, "caption_text": V.CAPTION_TEXT, "answer_suffix": V.ANSWER_SUFFIX,
              "connector_scale": scale, "calibration": calib, "split_sha256": V.split_hashes(V.find_splits_dir(REPO)),
              "dev_sizes": {k: len(v) for k, v in dev.items()}, "gpu": torch.cuda.get_device_name(0),
              "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
              "resumed_from": args.resume_from, "setup_seconds": time.time() - t_start}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    def stage_rec(k):
        return state["stages"].setdefault(str(k + 1), {"name": cfg["stages"][k]["name"], "train_seconds": 0.0,
                                                       "dev_seconds": 0.0, "peak_mem_allocated_gb": 0.0,
                                                       "peak_mem_reserved_gb": 0.0, "draws_by_kind": {},
                                                       "examples_by_kind": {}, "skipped_updates": []})

    def note_memory(k):
        s = stage_rec(k)
        s["peak_mem_allocated_gb"] = max(s["peak_mem_allocated_gb"], gb(torch.cuda.max_memory_allocated()))
        s["peak_mem_reserved_gb"] = max(s["peak_mem_reserved_gb"], gb(torch.cuda.max_memory_reserved()))

    def do_dev(done, k):
        t = time.time()
        rec, per_example = dev_eval(model, connector, store, dev)
        rec.update(update=done, stage=k + 1, seconds=time.time() - t)
        stage_rec(k)["dev_seconds"] += rec["seconds"]
        state["dev_log"].append(rec)
        (out / "dev_examples" / f"update_{done:04d}.json").write_text(json.dumps(per_example))
        print(f"dev @ {done}: caption {rec['caption_loss']:.4f} | vqa NLL {rec['vqa']['nll_correct']:.3f} "
              f"dNLL {rec['vqa']['delta_nll']:+.3f} | grounding NLL {rec['grounding']['nll_correct']:.3f} "
              f"dNLL {rec['grounding']['delta_nll']:+.3f} | {rec['seconds']:.0f} s")

    def write_outputs(done):
        with open(out / "train_log.jsonl", "w") as f:
            for r in state["train_log"]:
                f.write(json.dumps(r) + "\n")
        with open(out / "dev_eval.jsonl", "w") as f:
            for r in state["dev_log"]:
                f.write(json.dumps(r) + "\n")
        summary = {"updates_completed": done, "updates_total": total,
                   "examples_presented": EFFECTIVE * done, "stages": state["stages"],
                   "connector_scale": scale, "gpu": torch.cuda.get_device_name(0),
                   "wall_seconds_this_session": time.time() - t_start,
                   "skipped_updates": [r["update"] for r in state["train_log"] if r["skipped"]]}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    session = {"seconds": 0.0, "updates": 0}
    torch.cuda.reset_peak_memory_stats()
    if start == 0:
        stage_rec(0)["trainable_params"] = sum(p.numel() for p in connector.parameters())
        do_dev(0, 0)
    for update in range(start, total):
        k, u = locate(cfg, update)
        stage = cfg["stages"][k]
        if u == 0 and k > 0:
            note_memory(k - 1)
            torch.cuda.reset_peak_memory_stats()
            if k == 1:
                enable_lora(model, opt)
            stage_rec(k)["trainable_params"] = sum(p.numel() for _, p in V.trainable(model, connector))
        params = [p for _, p in V.trainable(model, connector)]
        lrs = set_lr(opt, stage, u)
        t = time.time()
        microbatches, kinds = cur.update(update)
        rec = train_update(model, connector, opt, scaler, store, microbatches, kinds, params)
        torch.cuda.synchronize()
        done = update + 1
        rec.update(update=done, stage=k + 1, lr=lrs, seconds=time.time() - t)
        state["train_log"].append(rec)
        s = stage_rec(k)
        s["train_seconds"] += rec["seconds"]
        for kind, n in rec["examples_by_kind"].items():
            s["examples_by_kind"][kind] = s["examples_by_kind"].get(kind, 0) + n
            s["draws_by_kind"][kind] = s["draws_by_kind"].get(kind, 0) + (n if k == 0 else n // 2)
        if rec["skipped"]:
            s["skipped_updates"].append(done)
            print(f"update {done}: skipped (overflow), loss scale {rec['loss_scale']}")
        session["seconds"] += rec["seconds"]
        session["updates"] += 1
        if done % 10 == 0 or update == start:
            eta = (total - done) * session["seconds"] / session["updates"]
            print(f"update {done:4d}/{total} | stage {k + 1} | loss {rec['loss']:.4f} | "
                  f"lr {', '.join(f'{n} {v:.2e}' for n, v in lrs.items())} | grad_norm {rec['grad_norm']:.3f} | "
                  f"{rec['seconds']:.2f} s/update | ETA {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m")
        note_memory(k)
        if done in dev_points(cfg):
            do_dev(done, k)
        if done in ends:
            save_weights(out / "weights" / f"stage{k + 1}", model, connector, with_lora=k > 0)
        out_of_time = (time.time() - t_start) / 3600 > args.time_limit_hours
        if done % cfg["checkpoint_every"] == 0 or done == total or out_of_time:
            t = time.time()
            save_checkpoint(ckpt_dir / f"update_{done:04d}", done, model, connector, opt, scaler, cur, state)
            print(f"checkpoint {done} saved in {time.time() - t:.1f} s")
            write_outputs(done)
            if out_of_time and done < total:
                print(f"time limit reached at update {done}; resume from {ckpt_dir / f'update_{done:04d}'}")
                return out, ckpt_dir
    write_outputs(total)
    return out, ckpt_dir


def smoke(args):
    """Short run of every code path (6/4/4 updates, checkpoints every 2, dev every 4 on 16 items each),
    then a resume from the mid-Stage-2 checkpoint, compared with the uninterrupted run. Not a result."""
    root = args.out_root / "task2" / "smoke"
    shutil.rmtree(root, ignore_errors=True)
    out, ckpt = run(args, SMOKE, root / "full")
    gc.collect()
    torch.cuda.empty_cache()
    a = argparse.Namespace(**{**vars(args), "resume_from": str(ckpt / "update_0008")})
    out2, _ = run(a, SMOKE, root / "resumed")
    full = [json.loads(l) for l in open(out / "train_log.jsonl")]
    res = [json.loads(l) for l in open(out2 / "train_log.jsonl")]
    w1 = load_file(str(out / "weights/stage3/connector.safetensors"))
    w2 = load_file(str(out2 / "weights/stage3/connector.safetensors"))
    l1 = load_file(str(out / "weights/stage3/lora.safetensors"))
    l2 = load_file(str(out2 / "weights/stage3/lora.safetensors"))
    secs = {str(k + 1): float(np.mean([r["seconds"] for r in full if r["stage"] == k + 1][1:])) for k in range(3)}
    report = {
        "updates": len(full), "log_lines_resumed": len(res),
        "same_examples_every_update": [a["image_ids"] for a in full] == [b["image_ids"] for b in res],
        "first_resumed_update_loss": [full[8]["loss"], res[8]["loss"]],
        "loss_by_update_full_vs_resumed": [[a["update"], a["loss"], b["loss"]] for a, b in zip(full, res)],
        "loss_after_resume_max_abs_diff": max(abs(a["loss"] - b["loss"]) for a, b in zip(full[8:], res[8:])),
        "final_connector_max_abs_diff": max(float((w1[k] - w2[k]).abs().max()) for k in w1),
        "final_lora_max_abs_diff": max(float((l1[k] - l2[k]).abs().max()) for k in l1),
        "checkpoints": sorted(p.name for p in ckpt.iterdir()),
        "dev_points": [r["update"] for r in (json.loads(l) for l in open(out / "dev_eval.jsonl"))],
        "seconds_per_update_by_stage": secs,
        "estimated_main_run_train_hours": sum(secs[str(k + 1)] * s["updates"] for k, s in enumerate(STAGES)) / 3600,
        "summary": json.loads((out / "summary.json").read_text()),
        "last_records": full[-3:],
        "dev_log": [json.loads(l) for l in open(out / "dev_eval.jsonl")],
    }
    (args.out_root / "task2" / "results").mkdir(parents=True, exist_ok=True)
    (args.out_root / "task2" / "results" / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("summary", "last_records", "dev_log")}))
    shutil.rmtree(root)


KAGGLE_ARGS = []  # filled in by kaggle_push.py; Kaggle script kernels cannot take CLI arguments


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume-from")
    p.add_argument("--time-limit-hours", type=float, default=11.0)
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else REPO)
    args = p.parse_args()
    if args.smoke:
        smoke(args)
    else:
        run(args, MAIN, args.out_root)


if __name__ == "__main__":
    main()
