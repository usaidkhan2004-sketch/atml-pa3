"""Task 1 training: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16 on MetaMathQA.

Single file so it runs as a Kaggle script kernel. Reads the tokenized data produced by
prepare_data.py (Kaggle kernel output mounted under /kaggle/input).

Usage
  python train.py --method fullft|l2sp|lora4|lora16|oplora
  python train.py --benchmark            # short memory/speed test of every method
  python train.py --method X --resume-from <recovery dir>

Settings follow the manual's "Shared training settings" table; open points are in DECISIONS.md.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 per run
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation near the 14.6 GB cap

import argparse
import gc
import sys
import glob
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

sys.stdout.reconfigure(line_buffering=True)  # progress shows live in Kaggle's log stream

SEED = 6304
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
IGNORE_INDEX = -100
PAD_ID = 151643  # <|endoftext|>, Qwen3's pad token

SHARED = {
    "n_examples": 40_000,
    "updates": 1_250,
    "microbatch": 2,
    "grad_accum": 16,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
    "warmup_updates": 38,
    "grad_clip": 1.0,
    "max_len": 2_048,
    "checkpoint_updates": [312, 625, 937, 1250],
    "keep_model_updates": [625, 1250],
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
METHODS = {
    "fullft": {"kind": "full", "lr": 2e-5, "l2sp_lambda": 0.0},
    "l2sp": {"kind": "full", "lr": 2e-5, "l2sp_lambda": 1e-3},
    "lora4": {"kind": "lora", "lr": 2e-4, "r": 4, "alpha": 8, "dropout": 0.05, "protect_k": 0},
    "lora16": {"kind": "lora", "lr": 2e-4, "r": 16, "alpha": 32, "dropout": 0.05, "protect_k": 0},
    "oplora": {"kind": "lora", "lr": 2e-4, "r": 16, "alpha": 32, "dropout": 0.05, "protect_k": 128},
}
LOSS_CHUNK = 1_024  # supervised positions per lm_head chunk; changes memory only, not the loss


# --------------------------------------------------------------------------- adapters

class LoRALinear(nn.Module):
    """W0 x + (alpha/r) B A dropout(x); with protect_k > 0, OPLoRA's projected factors
    B~ = B - U_k(U_k^T B), A~ = A - (A V_k) V_k^T are used instead of A, B."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float, protect_k: int,
                 generator: torch.Generator):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        a = torch.empty(r, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        self.lora_A = nn.Parameter(a.to(device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=device))
        self.protect_k = protect_k
        if protect_k:
            U, _, Vh = torch.linalg.svd(base.weight.detach().float(), full_matrices=False)
            self.register_buffer("U_k", U[:, :protect_k].contiguous(), persistent=False)
            self.register_buffer("V_k", Vh[:protect_k].T.contiguous(), persistent=False)

    def effective_factors(self):
        A, B = self.lora_A, self.lora_B
        if self.protect_k:
            with torch.autocast("cuda", enabled=False):
                B = B - self.U_k @ (self.U_k.T @ B)
                A = A - (A @ self.V_k) @ self.V_k.T
        return A, B

    def forward(self, x):
        A, B = self.effective_factors()
        return self.base(x) + F.linear(F.linear(self.dropout(x), A), B) * self.scaling


def inject_adapters(model, cfg):
    """Wrap every target projection in every layer, in a fixed order, with A drawn from a
    dedicated CPU generator so LoRA-16 and OPLoRA start from identical factors."""
    gen = torch.Generator().manual_seed(SEED)
    for layer in model.model.layers:
        for name in LORA_TARGETS:
            parent = layer.self_attn if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else layer.mlp
            setattr(parent, name, LoRALinear(getattr(parent, name), cfg["r"], cfg["alpha"],
                                             cfg["dropout"], cfg["protect_k"], gen))


def adapter_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state(model, projected: bool):
    out = {}
    for n, m in adapter_modules(model).items():
        A, B = m.effective_factors() if projected else (m.lora_A, m.lora_B)
        out[f"{n}.lora_A"] = A.detach().float().cpu().contiguous()
        out[f"{n}.lora_B"] = B.detach().float().cpu().contiguous()
    return out


# --------------------------------------------------------------------------- data

def load_data(path):
    df = pd.read_parquet(path)
    return [(torch.tensor(np.asarray(i), dtype=torch.long), torch.tensor(np.asarray(l), dtype=torch.long))
            for i, l in zip(df["input_ids"], df["labels"])]


def collate(examples):
    T = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), T), PAD_ID, dtype=torch.long)
    labels = torch.full((len(examples), T), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(examples), T), dtype=torch.long)
    for k, (i, l) in enumerate(examples):
        ids[k, :len(i)], labels[k, :len(l)], mask[k, :len(i)] = i, l, 1
    return ids, labels, mask


def update_batches(data, update):
    """Microbatches for optimizer update `update` (0-based): examples [32u, 32u + 32) in order."""
    eff = SHARED["microbatch"] * SHARED["grad_accum"]
    chunk = data[update * eff:(update + 1) * eff]
    return [collate(chunk[j:j + SHARED["microbatch"]]) for j in range(0, eff, SHARED["microbatch"])]


# --------------------------------------------------------------------------- loss

def _chunk_ce(h, y, lm_head):
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum")


def summed_response_ce(model, ids, labels, mask):
    """Sum of next-token cross-entropy over supervised positions. lm_head runs only on those
    positions, in checkpointed chunks, so full-vocabulary logits are never materialized at once."""
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    h, y = hidden[:, :-1], labels[:, 1:]
    sel = y != IGNORE_INDEX
    h, y = h[sel], y[sel]
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, y.numel(), LOSS_CHUNK):
        total = total + checkpoint(_chunk_ce, h[s:s + LOSS_CHUNK], y[s:s + LOSS_CHUNK], model.lm_head,
                                   use_reentrant=False)
    return total, int(sel.sum())


# --------------------------------------------------------------------------- helpers

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


def param_groups(params):
    """Weight decay on >=2-D parameters except biases (manual: exclude biases and <2-D params)."""
    decay, no_decay = [], []
    for n, p in params:
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": SHARED["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0}]


def dir_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def gb(x):
    return round(x / 2**30, 4)


class L2SP:
    """Omega = 1/2 sum_j (theta_j - theta0_j)^2 over unique parameters (tied counted once).

    Its gradient lambda (theta - theta0) is added to the unscaled SFT gradient once per update,
    before clipping, which equals the gradient of L_SFT + lambda * Omega without passing the
    penalty through FP16 loss scaling."""

    def __init__(self, params, lam, device):
        self.params, self.lam, self.device = params, lam, device
        t = time.time()
        if device == "cuda":
            self.ref = [p.detach().clone() for p in params]
        else:
            self.ref = [p.detach().to("cpu").pin_memory() for p in params]
        torch.cuda.synchronize()
        self.setup_seconds = time.time() - t
        self.ref_bytes = sum(r.numel() * r.element_size() for r in self.ref)
        self.transfer_seconds = 0.0

    @torch.no_grad()
    def apply(self):
        omega = torch.zeros((), device="cuda", dtype=torch.float64)
        for p, r in zip(self.params, self.ref):
            if self.device != "cuda":
                t = time.time()
                r = r.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                self.transfer_seconds += time.time() - t
            d = p - r
            omega += 0.5 * d.double().pow(2).sum()
            if self.lam and p.grad is not None:
                p.grad.add_(d, alpha=self.lam)
        return float(omega)


@torch.no_grad()
def displacement(model, ref_cpu):
    """||theta - theta0||_2 over unique parameters."""
    sq = 0.0
    for p, r in zip(model.parameters(), ref_cpu):
        sq += float((p.detach().float() - r.to(p.device, non_blocking=True)).double().pow(2).sum())
    return math.sqrt(sq)


# --------------------------------------------------------------------------- model setup

def build(method, theta0_device, overrides=None):
    cfg = {**SHARED, **METHODS[method], "method": method, "seed": SEED, "model": MODEL_ID,
           "model_revision": MODEL_REVISION, "loss_chunk": LOSS_CHUNK, **(overrides or {})}
    set_seed(SEED)
    full = cfg["kind"] == "full"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32 if full else torch.float16,
        attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    overhead = {}
    if full:
        named = [(n, p) for n, p in model.named_parameters()]
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        t = time.time()
        inject_adapters(model, cfg)
        torch.cuda.synchronize()
        overhead["adapter_setup_seconds"] = time.time() - t
        if cfg["protect_k"]:
            mods = adapter_modules(model).values()
            overhead["oplora_basis_bytes"] = sum(m.U_k.numel() * 4 + m.V_k.numel() * 4 for m in mods)
            overhead["oplora_preprocessing_seconds"] = overhead.pop("adapter_setup_seconds")
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    cfg["trainable_params"] = sum(p.numel() for _, p in named)
    cfg["total_params"] = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(param_groups(named), lr=cfg["lr"], betas=cfg["betas"], eps=cfg["eps"])
    sched = get_cosine_schedule_with_warmup(opt, cfg["warmup_updates"], cfg["updates"])
    scaler = torch.amp.GradScaler("cuda")

    l2sp = None
    if full and cfg["l2sp_lambda"]:
        l2sp = L2SP([p for _, p in named], cfg["l2sp_lambda"], theta0_device)
        cfg["theta0_device"] = theta0_device
        overhead["l2sp_reference_bytes"] = l2sp.ref_bytes
        overhead["l2sp_reference_setup_seconds"] = l2sp.setup_seconds
    return cfg, model, opt, sched, scaler, l2sp, named, overhead


def train_update(model, opt, sched, scaler, l2sp, named, cfg, batches):
    """One optimizer update over 16 microbatches. Returns the log record."""
    n_valid = sum(int((b[1][:, 1:] != IGNORE_INDEX).sum()) for b in batches)
    opt.zero_grad(set_to_none=True)
    sft = 0.0
    for ids, labels, mask in batches:
        ids, labels, mask = ids.cuda(non_blocking=True), labels.cuda(non_blocking=True), mask.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss_sum, _ = summed_response_ce(model, ids, labels, mask)
        loss = loss_sum / n_valid
        scaler.scale(loss).backward()
        sft += float(loss.detach())
    scaler.unscale_(opt)
    rec = {"lr": opt.param_groups[0]["lr"], "sft_loss": sft, "n_valid_tokens": n_valid}
    if l2sp is not None:
        omega = l2sp.apply()
        rec.update({"omega": omega, "lambda_omega": cfg["l2sp_lambda"] * omega,
                    "total_loss": sft + cfg["l2sp_lambda"] * omega})
    rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], cfg["grad_clip"]))
    scale_before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    sched.step()
    rec["loss_scale"] = scale_before
    rec["skipped"] = scaler.get_scale() < scale_before
    return rec


# --------------------------------------------------------------------------- checkpoints

def save_model_state(model, cfg, path):
    path.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if cfg["kind"] == "full":
        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        state.pop("lm_head.weight", None)  # tied to model.embed_tokens.weight
        save_file(state, str(path / "model.safetensors"))
    else:
        save_file(adapter_state(model, projected=False), str(path / "adapter.safetensors"))
        if cfg["protect_k"]:
            save_file(adapter_state(model, projected=True), str(path / "adapter_projected.safetensors"))
    return time.time() - t


def save_recovery(path, update, model, opt, sched, scaler, cfg, log):
    path.mkdir(parents=True, exist_ok=True)
    save_model_state(model, cfg, path)
    torch.save({"update": update, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(), "data_position": update * 32,
                "log": log}, path / "trainer_state.pt")


def load_recovery(path, model, opt, sched, scaler, cfg):
    if cfg["kind"] == "full":
        state = load_file(str(path / "model.safetensors"))
        model.load_state_dict(state, strict=False)
        model.tie_weights()
    else:
        state = load_file(str(path / "adapter.safetensors"))
        for n, m in adapter_modules(model).items():
            m.lora_A.data.copy_(state[f"{n}.lora_A"])
            m.lora_B.data.copy_(state[f"{n}.lora_B"])
    st = torch.load(path / "trainer_state.pt", weights_only=False)
    opt.load_state_dict(st["optimizer"])
    sched.load_state_dict(st["scheduler"])
    scaler.load_state_dict(st["scaler"])
    set_rng_state(st["rng"])
    return st["update"], st["log"]


# --------------------------------------------------------------------------- main loops

def find_data(arg):
    if arg:
        return arg
    hits = glob.glob("/kaggle/input/**/train_tokenized.parquet", recursive=True)
    if not hits:
        raise FileNotFoundError("train_tokenized.parquet not found; pass --data")
    return hits[0]


def run(args):
    out = args.out_root / "task1" / "runs" / args.method
    ckpt_dir = args.out_root / "checkpoints" / "task1" / args.method
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(find_data(args.data))
    assert len(data) == SHARED["n_examples"]

    torch.cuda.reset_peak_memory_stats()
    cfg, model, opt, sched, scaler, l2sp, named, overhead = build(args.method, args.theta0_device,
                                                                  getattr(args, "overrides", None))
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    theta0_cpu = [p.detach().float().cpu() for p in model.parameters()] if cfg["kind"] == "full" else None

    start, log = 0, []
    if args.resume_from:
        start, log = load_recovery(Path(args.resume_from), model, opt, sched, scaler, cfg)
        print(f"resumed at update {start}")

    t_run = time.time()
    train_seconds = 0.0
    summary = {"method": args.method, "displacement": {}, "checkpoints": {}}
    prev_recovery = None

    def write_outputs():
        """Called at every checkpoint and at the end, so a later failure cannot lose these."""
        summary.update({
            "updates_completed": len(log),
            "skipped_updates": [r["update"] for r in log if r["skipped"]],
            "train_seconds": sum(r["seconds"] for r in log),
            "wall_seconds_this_session": time.time() - t_run,
            "peak_mem_allocated_gb": gb(torch.cuda.max_memory_allocated()),
            "peak_mem_reserved_gb": gb(torch.cuda.max_memory_reserved()),
            "trainable_params": cfg["trainable_params"],
            "gpu": torch.cuda.get_device_name(0),
            "overhead": overhead,
        })
        with open(out / "train_log.jsonl", "w") as f:
            for r in log:
                f.write(json.dumps(r) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for u in range(start, cfg["updates"]):
        t = time.time()
        rec = train_update(model, opt, sched, scaler, l2sp, named, cfg, update_batches(data, u))
        torch.cuda.synchronize()
        rec["update"] = u + 1
        rec["seconds"] = time.time() - t
        train_seconds += rec["seconds"]
        log.append(rec)
        if rec["skipped"]:
            print(f"update {u + 1}: skipped (overflow), loss scale {rec['loss_scale']}")
        if (u + 1) % 10 == 0 or u == start:
            eta = (cfg["updates"] - u - 1) * train_seconds / (u + 1 - start)
            extra = f" | omega {rec['omega']:.4g}" if "omega" in rec else ""
            print(f"update {u + 1:4d}/{cfg['updates']} | sft_loss {rec['sft_loss']:.4f}{extra} | "
                  f"lr {rec['lr']:.2e} | grad_norm {rec['grad_norm']:.3f} | {rec['seconds']:.1f} s/update | "
                  f"ETA {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m")

        done = u + 1
        out_of_time = (time.time() - t_run) / 3600 > args.time_limit_hours
        if done in cfg["checkpoint_updates"] or out_of_time:
            rpath = ckpt_dir / f"recovery_{done:04d}"
            t = time.time()
            save_recovery(rpath, done, model, opt, sched, scaler, cfg, log)
            summary["checkpoints"][done] = {"recovery_bytes": dir_size(rpath), "save_seconds": time.time() - t}
            if prev_recovery is not None:
                shutil.rmtree(prev_recovery, ignore_errors=True)  # rolling recovery (DECISIONS.md #3)
            prev_recovery = rpath
            if done in cfg["keep_model_updates"] and (cfg["method"] == "fullft" or done == cfg["updates"]):
                mpath = out.parent.parent / "models" / args.method / f"update_{done:04d}"
                mpath.mkdir(parents=True, exist_ok=True)
                for f in rpath.glob("*.safetensors"):
                    os.link(f, mpath / f.name)  # same bytes as the recovery weights; survives its deletion
                summary["checkpoints"][done]["model_bytes"] = dir_size(mpath)
            if theta0_cpu is not None and done in cfg["keep_model_updates"]:
                summary["displacement"][done] = displacement(model, theta0_cpu)
            if l2sp is not None:
                overhead["l2sp_transfer_seconds"] = l2sp.transfer_seconds
            write_outputs()
            if out_of_time and done < cfg["updates"]:
                print(f"time limit reached at update {done}; resume from {rpath}")
                break

    if cfg.get("protect_k"):
        t = time.time()
        adapter_state(model, projected=True)
        overhead["oplora_export_seconds"] = time.time() - t
    write_outputs()
    print(json.dumps(summary, indent=2, default=str))


def benchmark(args):
    """Memory/speed test: per method, one update on the 32 longest examples (worst-case memory),
    then `--bench-updates` ordinary updates from the start of the data (throughput)."""
    data = load_data(find_data(args.data))
    longest = sorted(range(len(data)), key=lambda k: -len(data[k][0]))[:32]
    stress = [collate([data[longest[j]], data[longest[j + 1]]]) for j in range(0, 32, 2)]
    results = []
    variants = [("fullft", "none"), ("l2sp", "cuda"), ("l2sp", "cpu"),
                ("lora4", "none"), ("lora16", "none"), ("oplora", "none")]
    for method, dev in variants:
        res = {"method": method, "theta0_device": dev}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cfg, model, opt, sched, scaler, l2sp, named, overhead = build(method, dev)
            res["trainable_params"] = cfg["trainable_params"]
            res["overhead"] = overhead
            t = time.time()
            rec = train_update(model, opt, sched, scaler, l2sp, named, cfg, stress)
            torch.cuda.synchronize()
            res["stress_update_seconds"] = time.time() - t
            res["stress_tokens"] = sum(int(b[2].sum()) for b in stress)
            secs, toks = [], 0
            for u in range(args.bench_updates):
                batches = update_batches(data, u)
                t = time.time()
                rec = train_update(model, opt, sched, scaler, l2sp, named, cfg, batches)
                torch.cuda.synchronize()
                secs.append(time.time() - t)
                toks += sum(int(b[2].sum()) for b in batches)
            res["seconds_per_update"] = float(np.mean(secs[1:] if len(secs) > 1 else secs))
            res["tokens_per_second"] = toks / sum(secs)
            res["est_hours_1250_updates"] = res["seconds_per_update"] * SHARED["updates"] / 3600
            res["last_record"] = rec
            if l2sp is not None:
                res["overhead"]["l2sp_transfer_seconds"] = l2sp.transfer_seconds
            res["peak_mem_allocated_gb"] = gb(torch.cuda.max_memory_allocated())
            res["peak_mem_reserved_gb"] = gb(torch.cuda.max_memory_reserved())
            res["status"] = "ok"
        except torch.cuda.OutOfMemoryError as e:
            res["status"] = f"OOM: {str(e)[:300]}"
            res["peak_mem_allocated_gb"] = gb(torch.cuda.max_memory_allocated())
        finally:
            model = opt = sched = scaler = l2sp = named = None
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps(res, default=str))
        results.append(res)
    out = args.out_root / "task1" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_t4.json").write_text(json.dumps(
        {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": results},
        indent=2, default=str))


KAGGLE_ARGS = []  # filled in by kaggle_push.py; Kaggle script kernels cannot take CLI arguments


def smoke(args):
    """End-to-end check of the full code path (training, checkpoints, rolling deletion, kept models,
    displacement, summary, resume) on 3 updates per method. Not a result; written under smoke/."""
    root = args.out_root / "smoke"
    overrides = {"updates": 3, "checkpoint_updates": [1, 2, 3], "keep_model_updates": [2, 3]}
    report = {}
    for method in args.smoke_methods or list(METHODS):
        shutil.rmtree(root, ignore_errors=True)  # one method's files at a time (Kaggle's 20 GB cap)
        a = argparse.Namespace(**{**vars(args), "method": method, "out_root": root,
                                  "resume_from": None, "overrides": overrides})
        run(a)
        ckpts = root / "checkpoints" / "task1" / method
        a.resume_from = str(ckpts / "recovery_0003")
        a.out_root = root / "resumed"
        run(a)  # resume at the last update: loads every state dict, then runs the end-of-run code
        summ = json.loads((root / "task1" / "runs" / method / "summary.json").read_text())
        report[method] = {
            "recovery_dirs_left": sorted(p.name for p in ckpts.iterdir()),
            "kept_models": sorted(p.name for p in (root / "task1" / "models" / method).iterdir())
            if (root / "task1" / "models" / method).exists() else [],
            "log_lines": sum(1 for _ in open(root / "task1" / "runs" / method / "train_log.jsonl")),
            "displacement": summ["displacement"],
            "peak_mem_allocated_gb": summ["peak_mem_allocated_gb"],
            "overhead": summ["overhead"],
            "resumed_summary_exists": (root / "resumed" / "task1" / "runs" / method / "summary.json").exists(),
        }
        print(json.dumps({method: report[method]}, default=str))
        gc.collect()
        torch.cuda.empty_cache()
    (args.out_root / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    shutil.rmtree(root)


def main():
    import sys
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-methods", nargs="*", choices=list(METHODS))
    p.add_argument("--bench-updates", type=int, default=4)
    p.add_argument("--theta0-device", choices=["cuda", "cpu"], default="cpu")  # cuda OOMs on a T4 (benchmark_t4.json)
    p.add_argument("--resume-from")
    p.add_argument("--data")
    p.add_argument("--time-limit-hours", type=float, default=11.0)
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("."))
    args = p.parse_args()
    if args.smoke:
        smoke(args)
    elif args.benchmark:
        benchmark(args)
    elif args.method:
        run(args)
    else:
        p.error("--method or --benchmark is required")


if __name__ == "__main__":
    main()
