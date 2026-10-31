"""Task 2 Step 1 checks and connector-scale calibration.

Manual: "Check feature shapes, frozen parameter sets, connector gradients, and split isolation.
Calibrate connector output magnitude against text embeddings using training data only, and
record the rule and resulting scale."

Checks
  splits      every training image is train2014, every evaluation image is val2014; evaluation sets
              are image-disjoint from each other and from training; each val2014 image is used once
  features    [N, 196, 768] FP16, finite, one row per image in images.json, same order
  template    the no-image token sequence decodes to Qwen's own chat template for that user turn and
              answer; prompt|target split tokenization equals joint tokenization
  batch       labels only on target + <|im_end|>; 196 visual positions per image example; the
              no-image row is the image row with those positions removed; positions run 0..L-1
  equivalence no-image forward through inputs_embeds equals the input_ids forward; left- and
              right-padded batches give the same per-example NLL as one example at a time (exact
              test in FP32; the FP16 differences are recorded)
  frozen      Stage 1: only the connector trains; Stages 2-3: connector + LoRA A/B on q/k/v/o;
              every original Qwen parameter frozen, no trainable biases
  gradients   one FP16-autocast forward/backward with gradient checkpointing (loss unscaled):
              connector (and LoRA B) gradients finite and non-zero, no gradient on any frozen parameter
  calibration DECISIONS.md T2-8 rule on all training examples and all training images

Usage
  python task2/check_interface.py --local        # CPU: splits, template and batch checks only
  on Kaggle (GPU, features kernel mounted):       all checks + calibration
Output: task2/results/interface_checks.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task2" / "check_interface.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]
sys.path.insert(0, str(SELF.parent))
sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import vlm as V

SPLITS = V.find_splits_dir(REPO)


def load(name):
    return json.loads((SPLITS / f"{name}.json").read_text())


def training_examples(fmt):
    """Every training example once: 18,000 captions, 3,072 VQAv2 questions, both images of every
    presence and counting pool pair."""
    ex = [fmt.caption_example(c["image_id"], cap) for c in load("captions_train") for cap in c["captions"]]
    ex += [fmt.qa_example(it["image_id"], p["question"], it["target"]) for p in load("vqa_train") for it in p["items"]]
    pools = load("grounding_train_pools")
    ex += [fmt.qa_example(i, p["question"], a) for kind in ("presence", "counting") for p in pools[kind]
           for i, a in zip(p["images"], p["answers"])]
    return ex


def eval_examples(fmt, split):
    """(example, pair_id) for the development or final-test question suites."""
    out = []
    for p in load(f"vqa_{split}"):
        out += [(fmt.qa_example(it["image_id"], p["question"], it["target"]), p["pair_id"]) for it in p["items"]]
    for p in load(f"grounding_{split}"):
        out += [(fmt.qa_example(it["image_id"], p["question"], it["answer"]), p["pair_id"]) for it in p["items"]]
    return out


# --------------------------------------------------------------------------- CPU checks

def check_splits():
    images = load("images")
    train_ids = {x["image_id"] for x in images["train"]}
    val_ids = {x["image_id"] for x in images["val"]}
    assert all(x["file"].startswith("COCO_train2014_") for x in images["train"])
    assert all(x["file"].startswith("COCO_val2014_") for x in images["val"])
    pools = load("grounding_train_pools")
    used_train = ({c["image_id"] for c in load("captions_train")} |
                  {it["image_id"] for p in load("vqa_train") for it in p["items"]} |
                  {i for k in ("presence", "counting") for p in pools[k] for i in p["images"]})
    eval_sets = {"captions_dev": [c["image_id"] for c in load("captions_dev")]}
    for s in ("dev", "test"):
        eval_sets[f"vqa_{s}"] = [it["image_id"] for p in load(f"vqa_{s}") for it in p["items"]]
        eval_sets[f"grounding_{s}"] = [it["image_id"] for p in load(f"grounding_{s}") for it in p["items"]]
    all_eval = [i for v in eval_sets.values() for i in v]
    rec = {"train_images_used": len(used_train), "train_images_cached": len(train_ids),
           "eval_images": len(all_eval), "eval_images_cached": len(val_ids),
           "eval_set_sizes": {k: len(v) for k, v in eval_sets.items()}}
    assert used_train <= train_ids, "training example uses an image without train2014 features"
    assert set(all_eval) <= val_ids, "evaluation example uses an image outside val2014"
    assert not (train_ids & val_ids)
    assert len(all_eval) == len(set(all_eval)), "an evaluation image is used twice (T2-18)"
    assert rec["eval_set_sizes"] == {"captions_dev": 96, "vqa_dev": 64, "grounding_dev": 48,
                                     "vqa_test": 192, "grounding_test": 128}
    return rec


def check_template(tok, fmt, train):
    """No-image token ids decode to Qwen's template for (user = "\\n" + text, assistant = target)."""
    samples = [train[0], train[18_000], train[-1]] + [e for e, _ in eval_examples(fmt, "dev")[:2]]
    for ex in samples:
        ids = ex["pre"] + ex["post"] + ex["target"] + [V.IM_END_ID]
        text = tok.decode(ex["post"])
        user = text[:text.index("<|im_end|>")]
        ref = tok.apply_chat_template([{"role": "user", "content": user},
                                       {"role": "assistant", "content": tok.decode(ex["target"])}],
                                      tokenize=False, enable_thinking=False)
        assert tok.decode(ids) + "\n" == ref, (tok.decode(ids), ref)
    # prompt|target boundary: split tokenization equals joint tokenization of the same text
    mismatches = sum(tok(tok.decode(ex["post"] + ex["target"]), add_special_tokens=False)["input_ids"]
                     != ex["post"] + ex["target"] for ex in train[::50])
    assert mismatches == 0, mismatches
    return {"samples_checked": len(samples), "boundary_checked": len(train[::50]), "boundary_mismatches": mismatches,
            "pre_text": tok.decode(fmt.pre),
            "caption_post_text": tok.decode(fmt.post(V.CAPTION_TEXT)),
            "example_question_post_text": tok.decode(train[18_000]["post"])}


def cap_report(tok, fmt):
    """Content lengths against the manual's caps (question 80, caption target 64, answer 16)."""
    def lens(texts):
        n = [len(fmt.ids(t)) for t in texts]
        return n

    caps = {"caption": [c for x in load("captions_train") + load("captions_dev") for c in x.get("captions", [x.get("caption")])],
            "question": [p["question"] for s in ("train", "dev", "test") for p in load(f"vqa_{s}")] +
                        [p["question"] for s in ("dev", "test") for p in load(f"grounding_{s}")] +
                        sorted({p["question"] for k in ("presence", "counting") for p in load("grounding_train_pools")[k]}),
            "answer": [it["target"] for s in ("train", "dev", "test") for p in load(f"vqa_{s}") for it in p["items"]]}
    out = {}
    for kind, texts in caps.items():
        n = lens(texts)
        out[kind] = {"cap": V.CAPS[kind], "texts": len(n), "max_tokens": max(n), "over_cap": sum(x > V.CAPS[kind] for x in n)}
    return out


def check_batch(fmt, train):
    exs = [train[0], train[18_000], train[18_001]]
    for pad in ("right", "left"):
        b_img = V.build_batch(exs, [True, True, True], pad_side=pad)
        b_no = V.build_batch(exs, [False, False, False], pad_side=pad)
        for k, ex in enumerate(exs):
            m = b_img["attention_mask"][k].bool()
            ids, vis, lab, pos = (b_img[x][k][m] for x in ("input_ids", "vis_mask", "labels", "position_ids"))
            assert int(vis.sum()) == V.N_VIS and vis.nonzero()[0].item() == len(ex["pre"])
            assert (lab[lab != V.IGNORE_INDEX].tolist() == ex["target"] + [V.IM_END_ID])
            assert (lab != V.IGNORE_INDEX).nonzero()[0].item() == len(ex["pre"]) + V.N_VIS + len(ex["post"])
            assert pos.tolist() == list(range(int(m.sum())))
            assert (b_img["labels"][k][~m] == V.IGNORE_INDEX).all() and not b_img["vis_mask"][k][~m].any()
            m2 = b_no["attention_mask"][k].bool()
            assert b_no["input_ids"][k][m2].tolist() == ids[~vis].tolist()
            assert b_no["labels"][k][m2].tolist() == lab[~vis].tolist()
            assert b_no["position_ids"][k][m2].tolist() == list(range(int(m2.sum())))
        assert b_img["image_ids"] == [e["image_id"] for e in exs] and b_no["image_ids"] == []
    nll = V.build_batch(exs, [True] * 3, supervise_end=False)
    assert all(int((nll["labels"][k] != V.IGNORE_INDEX).sum()) == len(e["target"]) for k, e in enumerate(exs))
    return {"ok": True, "image_example_length": int(V.build_batch(exs[:1], [True])["attention_mask"].sum()),
            "no_image_example_length": int(V.build_batch(exs[:1], [False])["attention_mask"].sum())}


# --------------------------------------------------------------------------- GPU checks

def load_qwen(device):
    model = AutoModelForCausalLM.from_pretrained(V.MODEL_ID, revision=V.MODEL_REVISION, dtype=torch.float16,
                                                 attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    V.freeze_backbone(model)
    return model


@torch.no_grad()
def check_equivalence(model, connector, fmt, store, device):
    """Run in FP32 (exact structure test: padding and positions must not change any example's
    NLL) and in FP16 (the precision used later, for information)."""
    model.eval()
    exs = [e for e, _ in eval_examples(fmt, "dev")[:8]]
    flags = [True, False, True, True, False, True, True, True]  # mixed image / no-image rows
    rec = {}
    for dtype, tol in ((torch.float32, 1e-4), (torch.float16, None)):
        model.to(dtype)
        tag = str(dtype).split(".")[-1]
        b = V.to_device(V.build_batch(exs[:4], [False] * 4), device)
        a = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"]).logits
        e = model(inputs_embeds=V.embed(model, connector, b, None), attention_mask=b["attention_mask"],
                  position_ids=b["position_ids"]).logits
        m = b["attention_mask"].bool()
        rec[f"{tag}_no_image_embeds_vs_ids_max_abs_logit_diff"] = float((a - e)[m].abs().max())
        single = torch.cat([V.example_nll(model, connector, V.to_device(V.build_batch([x], [f]), device),
                                          store.get([x["image_id"]] if f else []))
                            for x, f in zip(exs, flags)])
        rec[f"{tag}_single_example_nll"] = [round(float(x), 4) for x in single]
        for pad in ("right", "left"):
            bb = V.to_device(V.build_batch(exs, flags, pad_side=pad), device)
            batched = V.example_nll(model, connector, bb, store.get(bb["image_ids"]))
            rec[f"{tag}_{pad}_pad_vs_single_max_abs_nll_diff"] = float((batched - single).abs().max())
        if tol is not None:
            assert rec[f"{tag}_no_image_embeds_vs_ids_max_abs_logit_diff"] < tol, rec
            assert rec[f"{tag}_right_pad_vs_single_max_abs_nll_diff"] < tol, rec
            assert rec[f"{tag}_left_pad_vs_single_max_abs_nll_diff"] < tol, rec
    assert next(model.parameters()).dtype == torch.float16
    return rec


def check_frozen_and_grads(model, connector, fmt, store, train, device, stage):
    names = [n for n, _ in V.trainable(model, connector)]
    base = [n for n, p in model.named_parameters() if not n.endswith(("lora_A", "lora_B"))]
    rec = {"stage": stage, "trainable_tensors": len(names),
           "trainable_params": sum(p.numel() for _, p in V.trainable(model, connector)),
           "connector_params": sum(p.numel() for p in connector.parameters()),
           "frozen_qwen_params": sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)}
    assert all(not p.requires_grad for n, p in model.named_parameters() if n in base)
    assert not any(n.endswith(".bias") for n in names if not n.startswith("connector."))
    if stage == 1:
        assert all(n.startswith("connector.") for n in names) and len(names) == 4
    else:
        lora = [n for n in names if not n.startswith("connector.")]
        assert len(lora) == 2 * len(V.LORA["targets"]) * len(model.model.layers), len(lora)
        assert all(n.split(".")[-2] in V.LORA["targets"] for n in lora)
        rec["lora_params"] = rec["trainable_params"] - rec["connector_params"]
    assert all(p.dtype == torch.float32 for _, p in V.trainable(model, connector))

    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    exs = train[:4] + train[18_000:18_004]
    b = V.to_device(V.build_batch(exs, [True] * 8), device)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = V.example_nll(model, connector, b, store.get(b["image_ids"])).mean()
    loss.backward()  # unscaled, so an overflow here is a real problem rather than GradScaler's first-step skip
    grads = {n: p.grad for n, p in V.trainable(model, connector)}
    rec["grad_none"] = [n for n, g in grads.items() if g is None]
    rec["grad_nonfinite"] = [n for n, g in grads.items() if g is not None and not torch.isfinite(g).all()]
    print("gradient status", {k: rec[k] for k in ("grad_none", "grad_nonfinite")})
    assert not rec["grad_none"] and not rec["grad_nonfinite"], rec
    rec["loss"] = float(loss)
    rec["connector_grad_norm"] = float(torch.stack([g.norm() for n, g in grads.items() if n.startswith("connector.")]).norm())
    assert rec["connector_grad_norm"] > 0
    if stage > 1:
        rec["lora_B_grad_norm"] = float(torch.stack([g.norm() for n, g in grads.items() if n.endswith("lora_B")]).norm())
        rec["lora_A_grad_norm"] = float(torch.stack([g.norm() for n, g in grads.items() if n.endswith("lora_A")]).norm())
        assert rec["lora_B_grad_norm"] > 0  # A's gradient is zero at init because B = 0
    assert all(p.grad is None for n, p in model.named_parameters() if n in base)
    for _, p in V.trainable(model, connector):
        p.grad = None
    model.gradient_checkpointing_disable()
    model.eval()
    return rec


def check_features(store):
    images = load("images")
    rec = {}
    for split in ("train", "val"):
        f = store.feats[split]
        ids = [i for i, (s, _) in store.index.items() if s == split]
        assert ids == [x["image_id"] for x in images[split]]
        assert tuple(f.shape) == (len(ids), V.N_VIS, V.D_VIS) and f.dtype == torch.float16
        finite = all(bool(torch.isfinite(f[s:s + 512]).all()) for s in range(0, len(f), 512))
        assert finite
        rec[split] = {"shape": list(f.shape), "dtype": str(f.dtype)}
    return rec


KAGGLE_ARGS = []


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true", help="CPU-only checks (no features, no model)")
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else REPO)
    args = p.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(V.MODEL_ID, revision=V.MODEL_REVISION)
    fmt = V.Formatter(tok)
    train = training_examples(fmt)
    rec = {"model": V.MODEL_ID, "model_revision": V.MODEL_REVISION, "caps": V.CAPS,
           "caption_text": V.CAPTION_TEXT, "answer_suffix": V.ANSWER_SUFFIX, "training_examples": len(train),
           "splits_dir": str(SPLITS), "split_sha256": V.split_hashes(SPLITS)}
    rec["splits"] = check_splits()
    print("splits ok", rec["splits"])
    rec["content_lengths"] = cap_report(tok, fmt)
    print("content lengths", rec["content_lengths"])
    rec["template"] = check_template(tok, fmt, train)
    print("template ok")
    rec["batch"] = check_batch(fmt, train)
    print("batch ok", rec["batch"])
    if not args.local:
        import transformers
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()
        store = V.FeatureStore(V.find_feature_dir(), device=device)
        rec["features"] = check_features(store)
        print("features ok")
        model = load_qwen(device)
        connector = V.new_connector().to(device)
        t = time.time()
        scale, rec["calibration"] = V.calibration_scale(model, connector, train, store.feats["train"])
        rec["calibration"]["seconds"] = time.time() - t
        connector.scale.fill_(scale)
        print("calibration", rec["calibration"])
        rec["equivalence"] = check_equivalence(model, connector, fmt, store, device)
        print("equivalence ok", rec["equivalence"])
        rec["stage1"] = check_frozen_and_grads(model, connector, fmt, store, train, device, stage=1)
        print("stage 1 ok", rec["stage1"])
        V.add_lora(model)
        rec["stage2_3"] = check_frozen_and_grads(model, connector, fmt, store, train, device, stage=2)
        print("stages 2-3 ok", rec["stage2_3"])
        rec["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 4)
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    rec["seconds"] = time.time() - t0
    rec["status"] = "local checks passed" if args.local else "all checks passed"
    out = args.out_root / "task2/results" / ("interface_checks_local.json" if args.local else "interface_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(rec["status"], "->", out)


if __name__ == "__main__":
    main()
