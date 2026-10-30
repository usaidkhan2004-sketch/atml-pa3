"""Task 1 weight-space and activation diagnostics (manual, step 6).

Checkpoints: the five variants at update 1,250 and Full FT at update 625. Matrices: down_proj in
layers 0, 13 and 27 (zero-based). With W0 the starting weight, dW the local update and U_k the
k = 128 leading left singular vectors of W0 (computed as OPLoRA computes its bases):

  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2      eta = ||dW||_F / ||W0||_F
  R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)

z is the starting model's down_proj input at the last prompt token, cached once with forward
hooks for the probe set: 32 GSM8K test questions in the evaluation's chat format (math) and 32
HellaSwag validation contexts as stock completion prompts (non-math). The same cached z is
reused for every checkpoint. Everything runs in FP32.
dW: Full FT variants W_trained - W0; adapters (alpha/r) B A, using OPLoRA's exported projected
factors.

Usage
  python diagnostics.py --prepare-prompts   # once, locally: probe prompts from the baseline eval samples
  python diagnostics.py                     # on Kaggle, with the five training kernels mounted
On Kaggle, push with
  --with task1/train.py --with task1/evaluate.py --with task1/splits/probe_ids.json
  --with task1/splits/probe_prompts.json --with task1/results/data_prep_summary.json
"""

import os
import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task1" / "diagnostics.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]
sys.path.insert(0, str(SELF.parent))

LAYERS = [0, 13, 27]
PROTECT_K = 128
EPS = 1e-8
CHECKPOINTS = ["fullft_0625", "fullft_1250", "l2sp_1250", "lora4_1250", "lora16_1250", "oplora_1250"]
PROMPTS_FILE = REPO / "task1/splits/probe_prompts.json"


def down_proj_key(i):
    return f"model.layers.{i}.mlp.down_proj"


def prepare_prompts():
    """Probe prompts in the benchmark formats, taken from the baseline evaluation itself: the exact
    GSM8K prompt string lm-eval generated, and lm-eval's prompt hash for each HellaSwag context
    (the context text is rebuilt on Kaggle with lm-eval's own process_docs and checked against it)."""
    probe = json.loads((REPO / "task1/splits/probe_ids.json").read_text())
    ev = REPO / "task1/results/eval/baseline"
    gsm = {r["doc_id"]: r for r in map(json.loads, open(ev / "samples_gsm8k_cot.jsonl"))}
    hs = {r["doc_id"]: r for r in map(json.loads, open(ev / "samples_hellaswag.jsonl"))}
    out = {
        "source": "task1/results/eval/baseline/samples_{gsm8k_cot,hellaswag}.jsonl",
        "gsm8k_test": [{"doc_id": i, "prompt": gsm[i]["prompt"]} for i in probe["gsm8k_test"]],
        "hellaswag_validation": [{"doc_id": i, "prompt_hash": hs[i]["prompt_hash"]}
                                 for i in probe["hellaswag_validation"]],
    }
    PROMPTS_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {PROMPTS_FILE}: {len(out['gsm8k_test'])} GSM8K, {len(out['hellaswag_validation'])} HellaSwag")


def probe_prompts():
    """[(group, doc_id, text)] for the 64 probe prompts, in probe-file order."""
    import lm_eval
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location(
        "hellaswag_utils", Path(lm_eval.__file__).parent / "tasks/hellaswag/utils.py")
    hs_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hs_utils)

    saved = json.loads(PROMPTS_FILE.read_text())
    revs = json.loads((REPO / "task1/results/data_prep_summary.json").read_text())["revisions"]
    hs = hs_utils.process_docs(load_dataset("Rowan/hellaswag", split="validation",
                                            revision=revs["Rowan/hellaswag"]))
    prompts = [("math", r["doc_id"], r["prompt"]) for r in saved["gsm8k_test"]]
    for r in saved["hellaswag_validation"]:
        text = hs[r["doc_id"]]["query"]  # stock context: doc_to_text "{{query}}", zero-shot
        assert hashlib.sha256(text.encode()).hexdigest() == r["prompt_hash"], \
            f"HellaSwag {r['doc_id']}: context differs from the one evaluated"
        prompts.append(("nonmath", r["doc_id"], text))
    return prompts


def describe(xs):
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
            "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs)}


def run(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    import evaluate as E
    import train as T

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out_root / "task1" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Starting weights. W0 is the checkpoint loaded in FP32, which is where Full FT starts.
    # Adapter runs start from the FP16 backbone; U_k is computed exactly as OPLoRA computes it,
    # from that FP16 weight cast to FP32, on the same device type.
    tok = E.load_tokenizer()
    base = E.load_base(torch.float32).to(dev).eval() if dev == "cuda" else \
        E.AutoModelForCausalLM.from_pretrained(T.MODEL_ID, revision=T.MODEL_REVISION, dtype=torch.float32).eval()
    W0, Uk, start = {}, {}, {}
    for i in LAYERS:
        w = base.model.layers[i].mlp.down_proj.weight.detach().float()
        W0[i] = w.clone()
        U, _, _ = torch.linalg.svd(w.half().float(), full_matrices=False)
        Uk[i] = U[:, :PROTECT_K].contiguous()
        start[i] = {"shape": list(w.shape), "W0_fro": float(w.norm()),
                    "fp16_roundtrip_max_abs_diff": float((w - w.half().float()).abs().max())}

    # Cache z = down_proj input at the last prompt token (batch of one: no padding).
    prompts = probe_prompts()
    cache, hooks = {}, []
    for i in LAYERS:
        def hook(mod, inp, outp, i=i):
            cache[i] = (inp[0][0, -1].detach().float().clone(), outp[0, -1].detach().float().clone())
        hooks.append(base.model.layers[i].mlp.down_proj.register_forward_hook(hook))
    Z = {i: [] for i in LAYERS}
    hook_check, n_tokens = 0.0, []
    with torch.no_grad():
        for group, doc_id, text in prompts:
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
            n_tokens.append(int(ids.shape[1]))
            base(input_ids=ids)
            for i in LAYERS:
                z, y = cache[i]
                Z[i].append(z)
                hook_check = max(hook_check, float((W0[i] @ z - y).abs().max()))
    for h in hooks:
        h.remove()
    Z = {i: torch.stack(v) for i, v in Z.items()}  # [64, in_features]
    save_file({f"layer{i}": Z[i].cpu().contiguous() for i in LAYERS}, str(out / "probe_activations.safetensors"))
    del base
    W0z = {i: (Z[i] @ W0[i].T).norm(dim=1) for i in LAYERS}  # ||W0 z||, shared by every checkpoint

    weight_rows, act_rows, summary_rows = [], [], []
    sources = {}
    for cond in CHECKPOINTS:
        method, update = E.CONDITIONS[cond]
        cfg = T.METHODS[method]
        d = E.model_dir(args.models_root, method, update)
        if cfg["kind"] == "full":
            f = d / "model.safetensors"
            with safe_open(str(f), "pt") as st:
                dW = {i: st.get_tensor(f"{down_proj_key(i)}.weight").float().to(dev) - W0[i] for i in LAYERS}
        else:
            f = d / ("adapter_projected.safetensors" if cfg["protect_k"] else "adapter.safetensors")
            with safe_open(str(f), "pt") as st:
                dW = {i: (cfg["alpha"] / cfg["r"]) * (st.get_tensor(f"{down_proj_key(i)}.lora_B").float().to(dev)
                                                     @ st.get_tensor(f"{down_proj_key(i)}.lora_A").float().to(dev))
                      for i in LAYERS}
        sources[cond] = {"file": str(f), "sha256": E.sha256(f)}
        for i in LAYERS:
            d_fro = dW[i].norm()
            weight_rows.append({
                "condition": cond, "layer": i,
                "rho_k": float((Uk[i] @ (Uk[i].T @ dW[i])).norm() ** 2 / d_fro ** 2),
                "eta": float(d_fro / W0[i].norm()),
                "dW_fro": float(d_fro), "W0_fro": float(W0[i].norm()),
            })
            dWz = (Z[i] @ dW[i].T).norm(dim=1)
            R = dWz / (W0z[i] + EPS)
            for (group, doc_id, _), r, a, b in zip(prompts, R.tolist(), dWz.tolist(), W0z[i].tolist()):
                act_rows.append({"condition": cond, "layer": i, "group": group, "doc_id": doc_id,
                                 "R": r, "dW_z_norm": a, "W0_z_norm": b})
        print(f"[{cond}] " + " | ".join(f"L{r['layer']} rho {r['rho_k']:.4f} eta {r['eta']:.5f}"
                                        for r in weight_rows if r["condition"] == cond))

    for cond in CHECKPOINTS:
        for layer in LAYERS + ["all"]:
            for group in ["math", "nonmath"]:
                rs = [r for r in act_rows if r["condition"] == cond and r["group"] == group
                      and (layer == "all" or r["layer"] == layer)]
                summary_rows.append({"condition": cond, "layer": layer, "group": group,
                                     **{f"R_{k}": v for k, v in describe([r["R"] for r in rs]).items()},
                                     "dW_z_norm_mean": statistics.fmean(r["dW_z_norm"] for r in rs),
                                     "W0_z_norm_mean": statistics.fmean(r["W0_z_norm"] for r in rs)})

    with open(out / "activation_ratios.jsonl", "w") as fh:
        for r in act_rows:
            fh.write(json.dumps(r) + "\n")
    for name, rows in [("weight_diagnostics.csv", weight_rows), ("activation_summary.csv", summary_rows)]:
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "diagnostics.json").write_text(json.dumps({
        "checkpoints": CHECKPOINTS, "layers": LAYERS, "matrix": "mlp.down_proj", "protect_k": PROTECT_K,
        "eps": EPS, "dtype": "float32", "device": dev,
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
        "bases": "U_k = first k left singular vectors of torch.linalg.svd(W0.half().float()), as in OPLoRA",
        "starting_weights": start, "weight_sources": sources,
        "probe": {"n_math": sum(g == "math" for g, _, _ in prompts),
                  "n_nonmath": sum(g == "nonmath" for g, _, _ in prompts),
                  "prompt_tokens": {"math": [n for n, (g, _, _) in zip(n_tokens, prompts) if g == "math"],
                                    "nonmath": [n for n, (g, _, _) in zip(n_tokens, prompts) if g == "nonmath"]},
                  "hook_vs_W0z_max_abs_diff": hook_check},
        "weight": weight_rows, "activation_summary": summary_rows,
        "seconds": time.time() - t0, "versions": E.versions() if "lm_eval" in sys.modules else None,
    }, indent=2) + "\n")
    print(f"done in {time.time() - t0:.0f} s -> {out}")


KAGGLE_ARGS = []


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--prepare-prompts", action="store_true")
    p.add_argument("--models-root", nargs="+", default=["/kaggle/input"])
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("."))
    args = p.parse_args()
    if args.prepare_prompts:
        prepare_prompts()
        return
    import evaluate as E
    E.ensure_lm_eval()  # its HellaSwag process_docs builds the non-math contexts
    run(args)


if __name__ == "__main__":
    main()
