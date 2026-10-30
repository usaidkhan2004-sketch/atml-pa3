"""Task 1 evaluation of the seven conditions with lm-evaluation-harness (pinned version).

GSM8K: `gsm8k_cot`, zero-shot, non-thinking chat template, greedy, 1,024 new tokens, stock
stopping rules and answer filters. HellaSwag, BoolQ, ARC-Easy: stock completion prompts, no
chat template. Every model runs in FP16 in eval mode (dropout off). Adapters are merged into a
fresh backbone copy (OPLoRA: the exported projected factors), after a merged-vs-unmerged logits
check on a small batch.

Each condition has a generation part (GSM8K) and a likelihood part (the other three). Each part
runs in its own worker process; with two GPUs the parts run in parallel, one per GPU.

Usage
  python evaluate.py --conditions baseline fullft_0625 fullft_1250
  python evaluate.py --smoke         # few documents per task, synthetic adapters; not a result
On Kaggle, push with
  --with task1/train.py --with task1/splits/probe_ids.json --with task1/results/data_prep_summary.json
so the training module, split IDs, and this file exist on disk for the worker processes.
"""

import os

PHYSICAL_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES")  # before train.py pins the default to GPU 0

import argparse
import glob
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task1" / "evaluate.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]
sys.path.insert(0, str(SELF.parent))

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import train as T

LM_EVAL_VERSION = "0.4.13"
GEN_TASK = "gsm8k_cot"
LL_TASKS = ["hellaswag", "boolq", "arc_easy"]
MAX_GEN_TOKS = 1_024
GEN_BATCH = 32  # same for every condition
LL_BATCH = 8
PRIMARY = {"gsm8k_cot": "exact_match,flexible-extract", "hellaswag": "acc_norm,none",
           "boolq": "acc,none", "arc_easy": "acc_norm,none"}
CONDITIONS = {  # name: (training method, optimizer update); baseline is the unchanged model
    "baseline": (None, 0),
    "fullft_0625": ("fullft", 625),
    "fullft_1250": ("fullft", 1250),
    "l2sp_1250": ("l2sp", 1250),
    "lora4_1250": ("lora4", 1250),
    "lora16_1250": ("lora16", 1250),
    "oplora_1250": ("oplora", 1250),
}
CHAT_PROMPT_END = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MERGE_CHECK_PER_SOURCE = 4  # sequences from each probe source in the merged-vs-unmerged batch


# --------------------------------------------------------------------------- helpers

def gb(x):
    return round(x / 2**30, 4)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def gpu_ids():
    if PHYSICAL_GPUS is not None:
        return [g for g in PHYSICAL_GPUS.split(",") if g.strip()]
    try:
        lines = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    except FileNotFoundError:
        return []
    return [str(i) for i, l in enumerate(l for l in lines if l.startswith("GPU"))]


def ensure_lm_eval():
    try:
        import lm_eval
        if lm_eval.__version__ == LM_EVAL_VERSION:
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"lm_eval[hf]=={LM_EVAL_VERSION}"], check=True)


def versions():
    import importlib.metadata as md
    return {k: md.version(k) for k in ["lm_eval", "torch", "transformers", "datasets", "accelerate"]}


def dataset_revision(path):
    """Commit of the dataset repo at evaluation time (lm-eval loads its default branch)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(path).sha
    except Exception as e:  # recorded, not fatal
        return f"unavailable: {e!r}"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(T.MODEL_ID, revision=T.MODEL_REVISION)


def load_base(dtype):
    return AutoModelForCausalLM.from_pretrained(T.MODEL_ID, revision=T.MODEL_REVISION, dtype=dtype,
                                                attn_implementation="sdpa").cuda()


def model_dir(roots, method, update):
    for root in roots:
        hits = sorted(glob.glob(f"{root}/**/task1/models/{method}/update_{update:04d}", recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"no saved model for {method} update {update} under {roots}")


# --------------------------------------------------------------------------- models

def merge_adapters(model, state, cfg):
    """W = W0 + (alpha/r) B A in FP32 for every target projection; every tensor must be used."""
    scaling = cfg["alpha"] / cfg["r"]
    for i, layer in enumerate(model.model.layers):
        for name in T.LORA_TARGETS:
            attn = name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
            key = f"model.layers.{i}.{'self_attn' if attn else 'mlp'}.{name}"
            A, B = state.pop(f"{key}.lora_A").cuda(), state.pop(f"{key}.lora_B").cuda()
            getattr(layer.self_attn if attn else layer.mlp, name).weight.data += scaling * (B @ A)
    assert not state, f"unused adapter tensors: {sorted(state)[:4]}"


def build_model(cond, roots):
    """FP16 eval-mode model for a condition, plus a record of where its weights came from."""
    method, update = CONDITIONS[cond]
    info = {"condition": cond, "method": method, "update": update,
            "base": T.MODEL_ID, "base_revision": T.MODEL_REVISION}
    if method is None:
        model = load_base(torch.float16)
    else:
        cfg = T.METHODS[method]
        d = model_dir(roots, method, update)
        if cfg["kind"] == "full":
            f = d / "model.safetensors"
            model = load_base(torch.float16)
            missing, unexpected = model.load_state_dict(load_file(str(f)), strict=False)
            assert not unexpected and missing == ["lm_head.weight"], (missing, unexpected)
            model.tie_weights()
        else:
            f = d / ("adapter_projected.safetensors" if cfg["protect_k"] else "adapter.safetensors")
            model = load_base(torch.float32)  # merge in FP32, then cast the merged weights once
            merge_adapters(model, load_file(str(f)), cfg)
            model.half()
        info.update(weights=str(f), weights_sha256=sha256(f))
    model.eval()
    model.config.use_cache = True
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters()), "non-finite FP16 weights"
    info["dtype"] = str(model.dtype)
    return model, info


def probe_batch(tok):
    """First MERGE_CHECK_PER_SOURCE probe questions from each source: GSM8K in the evaluation's
    chat format, HellaSwag as plain completion text. Right-padded."""
    from datasets import load_dataset
    probe = json.loads((REPO / "task1/splits/probe_ids.json").read_text())
    revs = json.loads((REPO / "task1/results/data_prep_summary.json").read_text())["revisions"]
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=revs["openai/gsm8k"])
    hs = load_dataset("Rowan/hellaswag", split="validation", revision=revs["Rowan/hellaswag"])
    texts = [tok.apply_chat_template([{"role": "user", "content": f"Q: {gsm[i]['question']}\nA:"}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for i in probe["gsm8k_test"][:MERGE_CHECK_PER_SOURCE]]
    texts += [f"{hs[i]['activity_label']}: {hs[i]['ctx']}" for i in probe["hellaswag_validation"][:MERGE_CHECK_PER_SOURCE]]
    side, tok.padding_side = tok.padding_side, "right"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    tok.padding_side = side
    return enc["input_ids"].cuda(), enc["attention_mask"].cuda()


@torch.no_grad()
def merge_check(cond, roots, merged, tok):
    """Logits of the merged FP16 model vs the unmerged one (FP16 backbone + FP32 adapters under
    FP16 autocast, as in training). For OPLoRA the unmerged model projects the raw factors with
    bases recomputed from the backbone, which also checks the exported projected factors."""
    method, update = CONDITIONS[cond]
    cfg = T.METHODS[method]
    d = model_dir(roots, method, update)
    ids, mask = probe_batch(tok)
    out = {"n_sequences": int(ids.shape[0]), "n_positions": int(mask.sum())}

    unmerged = load_base(torch.float16)
    for p in unmerged.parameters():
        p.requires_grad_(False)
    T.inject_adapters(unmerged, cfg)
    raw = load_file(str(d / "adapter.safetensors"))
    mods = T.adapter_modules(unmerged)
    assert set(raw) == {f"{n}.lora_{x}" for n in mods for x in "AB"}
    for n, m in mods.items():
        m.lora_A.data.copy_(raw[f"{n}.lora_A"])
        m.lora_B.data.copy_(raw[f"{n}.lora_B"])
    unmerged.eval()
    if cfg["protect_k"]:
        proj = load_file(str(d / "adapter_projected.safetensors"))
        export_diff, ub, av = 0.0, 0.0, 0.0
        for n, m in mods.items():
            A, B = m.effective_factors()
            export_diff = max(export_diff, float((A - proj[f"{n}.lora_A"].cuda()).abs().max()),
                              float((B - proj[f"{n}.lora_B"].cuda()).abs().max()))
            ub = max(ub, float((m.U_k.T @ B).abs().max()))
            av = max(av, float((A @ m.V_k).abs().max()))
        out.update(projected_export_max_abs_diff=export_diff, max_abs_UkT_Btilde=ub, max_abs_Atilde_Vk=av)
    with torch.autocast("cuda", dtype=torch.float16):
        lu = unmerged(input_ids=ids, attention_mask=mask).logits.float()
    del unmerged, mods, raw
    torch.cuda.empty_cache()
    lm = merged(input_ids=ids, attention_mask=mask).logits.float()

    sel = mask.bool()
    diff = (lu - lm)[sel].abs()
    last = mask.sum(1) - 1
    rows = torch.arange(ids.shape[0], device=ids.device)
    out.update(
        max_abs_logit_diff=float(diff.max()), mean_abs_logit_diff=float(diff.mean()),
        max_abs_logit=float(lm[sel].abs().max()),
        top1_agreement=float((lu.argmax(-1) == lm.argmax(-1))[sel].float().mean()),
        last_token_top1_agreement=float((lu[rows, last].argmax(-1) == lm[rows, last].argmax(-1)).float().mean()),
    )
    return out


# --------------------------------------------------------------------------- lm-eval

def tracking_hflm():
    from lm_eval.models.huggingface import HFLM

    class TrackingHFLM(HFLM):
        """HFLM that records, per prompt, why generation stopped (stop sequence, end-of-sequence
        token, or the new-token cap), so truncation can be told apart from wrong answers."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.stop_records = {}
            eos = self.model.generation_config.eos_token_id
            self.stop_ids = set(eos if isinstance(eos, list) else [eos]) | {
                self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}

        def _model_generate(self, context, max_length, stop, **kw):
            out = super()._model_generate(context, max_length, stop, **kw)
            n_ctx = context.shape[1]
            for row, m, o in zip(context, kw["attention_mask"].bool(), out):
                prompt = self.tokenizer.decode(row[m].tolist(), skip_special_tokens=False)
                new = o[n_ctx:].tolist()
                k = next((j for j, t in enumerate(new) if t in self.stop_ids), None)
                text = self.tokenizer.decode(new[:k], skip_special_tokens=False)
                hits = [text.find(s) for s in stop if s and s in text]
                if hits:
                    reason = "stop_sequence"
                    n = len(self.tokenizer(text[:min(hits)], add_special_tokens=False)["input_ids"])
                elif k is not None:
                    reason, n = "eos", k
                else:
                    reason, n = "length", len(new)
                self.stop_records[prompt] = {"stop_reason": reason, "n_new_tokens": n,
                                             "new_token_cap": max_length - n_ctx}
            return out

    return TrackingHFLM


def evaluate_task(lm, task, limit, **kw):
    from lm_eval import simple_evaluate
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    res = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, log_samples=True, limit=limit, **kw)
    seconds = time.time() - t
    cfg = res["configs"][task]
    record = {
        "metrics": res["results"][task], "n_samples": res["n-samples"][task],
        "task_version": res["versions"][task], "task_config": cfg,
        "dataset": {"path": cfg.get("dataset_path"), "name": cfg.get("dataset_name"),
                    "revision_at_eval": dataset_revision(cfg.get("dataset_path"))},
        "seconds": seconds,
        "peak_mem_allocated_gb": gb(torch.cuda.max_memory_allocated()),
        "peak_mem_reserved_gb": gb(torch.cuda.max_memory_reserved()),
    }
    return record, res["samples"][task]


def gsm8k_rows(samples, stop_records):
    """One row per question: response, both filters' extracted answers and scores, stop reason."""
    rows = {}
    for s in samples:
        r = rows.setdefault(s["doc_id"], {
            "doc_id": s["doc_id"], "question": s["doc"]["question"], "target": s["target"],
            "prompt": s["arguments"][0][0], "response": s["resps"][0][0]})
        r[f"extracted_{s['filter']}"] = s["filtered_resps"][0]
        r[f"exact_match_{s['filter']}"] = s["exact_match"]
    out = []
    for r in sorted(rows.values(), key=lambda r: r["doc_id"]):
        assert r["prompt"].endswith(CHAT_PROMPT_END), "GSM8K prompt is not the non-thinking chat format"
        rec = stop_records[r["prompt"]]  # KeyError: a prompt the tracker did not see
        r.update(rec, truncated=rec["stop_reason"] == "length",
                 no_extraction_flexible=r["extracted_flexible-extract"] == "[invalid]",
                 no_extraction_strict=r["extracted_strict-match"] == "[invalid]")
        out.append(r)
    return out


def ll_rows(samples):
    return [{"doc_id": s["doc_id"], "target": s["target"],
             "loglikelihoods": [float(x[0]) if isinstance(x, (list, tuple)) else float(x)
                                for x in s["filtered_resps"]],
             **{m: s[m] for m in ("acc", "acc_norm") if m in s},
             "doc_hash": s["doc_hash"], "prompt_hash": s["prompt_hash"]}
            for s in sorted(samples, key=lambda s: s["doc_id"])]


def worker(args):
    """One part (gen or ll) of one condition, on the single GPU this process sees."""
    out = eval_dir(args) / args.condition
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{args.condition}/{args.worker}] start on {torch.cuda.get_device_name(0)}")
    tok = load_tokenizer()
    t = time.time()
    model, info = build_model(args.condition, args.models_root)
    part = {"model": info, "model_load_seconds": time.time() - t, "versions": versions(),
            "gpu": torch.cuda.get_device_name(0), "tasks": {}}
    method = CONDITIONS[args.condition][0]

    if args.worker == "gen":
        if method and T.METHODS[method]["kind"] == "lora":
            part["merge_check"] = merge_check(args.condition, args.models_root, model, tok)
            print(f"[{args.condition}] merge check: {json.dumps(part['merge_check'])}")
        lm = tracking_hflm()(pretrained=model, tokenizer=tok, batch_size=GEN_BATCH, enable_thinking=False)
        part["settings"] = {"batch_size": GEN_BATCH, "max_gen_toks": MAX_GEN_TOKS, "num_fewshot": 0,
                            "apply_chat_template": True, "enable_thinking": False, "dtype": "float16"}
        rec, samples = evaluate_task(lm, GEN_TASK, args.limit, apply_chat_template=True,
                                     gen_kwargs={"max_gen_toks": MAX_GEN_TOKS})
        rows = gsm8k_rows(samples, lm.stop_records)
        rec.update(n_truncated=sum(r["truncated"] for r in rows),
                   n_no_extraction_flexible=sum(r["no_extraction_flexible"] for r in rows),
                   n_no_extraction_strict=sum(r["no_extraction_strict"] for r in rows),
                   mean_new_tokens=sum(r["n_new_tokens"] for r in rows) / len(rows),
                   stop_reasons={k: sum(r["stop_reason"] == k for r in rows)
                                 for k in ("stop_sequence", "eos", "length")})
        part["tasks"][GEN_TASK] = rec
        write_jsonl(out / f"samples_{GEN_TASK}.jsonl", rows)
    else:
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=LL_BATCH)
        part["settings"] = {"batch_size": LL_BATCH, "num_fewshot": 0, "apply_chat_template": False,
                            "dtype": "float16"}
        for task in LL_TASKS:
            rec, samples = evaluate_task(lm, task, args.limit, apply_chat_template=False)
            part["tasks"][task] = rec
            write_jsonl(out / f"samples_{task}.jsonl", ll_rows(samples))
            (out / f"part_{args.worker}.json").write_text(json.dumps(part, indent=2, default=str))
    (out / f"part_{args.worker}.json").write_text(json.dumps(part, indent=2, default=str))
    print(f"[{args.condition}/{args.worker}] done")


# --------------------------------------------------------------------------- orchestration

def eval_dir(args):
    return args.out_root / "task1" / ("eval_smoke" if args.smoke else "eval")


def run_condition(args, cond):
    out = eval_dir(args) / cond
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("part_*.json"):
        f.unlink()
    gpus = gpu_ids() or ["0"]
    base = [sys.executable, str(SELF), "--condition", cond, "--out-root", str(args.out_root),
            "--models-root", *args.models_root] + (["--limit", str(args.limit)] if args.limit else []) \
        + (["--smoke"] if args.smoke else [])
    t = time.time()
    if len(gpus) >= 2:
        procs = {part: subprocess.Popen(base + ["--worker", part], env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
                 for part, g in zip(["gen", "ll"], gpus)}
        codes = {part: p.wait() for part, p in procs.items()}
    else:
        codes = {part: subprocess.run(base + ["--worker", part],
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[0]}).returncode
                 for part in ["gen", "ll"]}
    parts = {p: json.loads((out / f"part_{p}.json").read_text())
             for p in ["gen", "ll"] if (out / f"part_{p}.json").exists()}
    tasks = {k: v for p in parts.values() for k, v in p["tasks"].items()}
    summary = {
        "condition": cond,
        "status": "ok" if all(c == 0 for c in codes.values()) and set(tasks) == {GEN_TASK, *LL_TASKS} else "failed",
        "return_codes": codes,
        "model": parts.get("gen", parts.get("ll", {})).get("model"),
        "merge_check": parts.get("gen", {}).get("merge_check"),
        "primary": {t: {"metric": PRIMARY[t], "value": tasks[t]["metrics"].get(PRIMARY[t])} for t in tasks},
        "scores": {t: {k: v for k, v in r["metrics"].items() if k != "alias"} for t, r in tasks.items()},
        "n_samples": {t: r["n_samples"] for t, r in tasks.items()},
        "gsm8k_generation": {k: tasks[GEN_TASK][k] for k in ("n_truncated", "n_no_extraction_flexible",
                             "n_no_extraction_strict", "mean_new_tokens", "stop_reasons")} if GEN_TASK in tasks else None,
        "resources": {
            "wall_seconds": time.time() - t, "gpus": len(gpus), "parallel_parts": len(gpus) >= 2,
            "model_load_seconds": {p: v["model_load_seconds"] for p, v in parts.items()},
            "tasks": {t: {k: r[k] for k in ("seconds", "peak_mem_allocated_gb", "peak_mem_reserved_gb")}
                      for t, r in tasks.items()},
        },
        "dataset_revisions": {t: r["dataset"] for t, r in tasks.items()},
        "settings": {p: v.get("settings") for p, v in parts.items()},
        "lm_eval_version": LM_EVAL_VERSION,
        "versions": parts.get("gen", parts.get("ll", {})).get("versions"),
        "gpu": parts.get("gen", parts.get("ll", {})).get("gpu"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    line = " | ".join(f"{t} {v['value']:.4f}" for t, v in summary["primary"].items() if v["value"] is not None)
    print(f"== {cond}: {summary['status']} | {line} | {summary['resources']['wall_seconds'] / 60:.1f} min")
    return summary


def make_synthetic_adapters(root):
    """Smoke only: LoRA-16 and OPLoRA adapters with small random B, saved exactly as training
    saves them, so the merge path and the merged-vs-unmerged check run before real adapters exist."""
    for method in ["lora16", "oplora"]:
        cfg = {**T.SHARED, **T.METHODS[method]}
        model = load_base(torch.float16)
        for p in model.parameters():
            p.requires_grad_(False)
        T.inject_adapters(model, cfg)
        gen = torch.Generator().manual_seed(T.SEED)
        for m in T.adapter_modules(model).values():
            m.lora_B.data.copy_(0.02 * torch.randn(m.lora_B.shape, generator=gen))
        T.save_model_state(model, cfg, root / "task1" / "models" / method / "update_1250")
        del model
        torch.cuda.empty_cache()


KAGGLE_ARGS = []


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, help="documents per task (smoke tests only)")
    p.add_argument("--models-root", nargs="+", default=["/kaggle/input"])
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("."))
    p.add_argument("--worker", choices=["gen", "ll"], help=argparse.SUPPRESS)
    p.add_argument("--condition", choices=list(CONDITIONS), help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        worker(args)
        return
    ensure_lm_eval()
    if args.smoke:
        root = args.out_root / "smoke_models"
        make_synthetic_adapters(root)
        args.models_root = [*args.models_root, str(root)]
        args.limit = args.limit or 8
        args.conditions = args.conditions or ["baseline", "fullft_1250", "lora16_1250", "oplora_1250"]
    if not args.conditions:
        p.error("--conditions or --smoke is required")
    for cond in args.conditions:
        run_condition(args, cond)
    if args.smoke:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
