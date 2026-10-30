"""Task 1 qualitative generations (manual, step 7).

All 24 prompts in task1/configs/first_plot_questions.yaml (eight questions in plain, JSON and
template formats), one greedy response each with thinking disabled and a 512-new-token cap, for
the seven evaluation conditions: 168 responses. Each condition is built exactly as for the
benchmarks (evaluate.build_model: FP16, eval mode, adapters merged, OPLoRA's projected factors).
Prompts use the Qwen3 chat template; the JSON-format entries carry the file's system message.
Each prompt is generated alone (batch of one, no padding).

Outputs (task1/qualitative/):
  responses.json            grouped by question, then format, then condition
  responses.jsonl           one row per response, with prompt, stop reason and truncation flag
  annotation_subset.md      the first and last questions (all formats, all conditions) = 42 responses

Usage
  python generate_qualitative.py
On Kaggle, push with
  --with task1/train.py --with task1/evaluate.py --with task1/configs/first_plot_questions.yaml
"""

import os
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task1" / "generate_qualitative.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]
sys.path.insert(0, str(SELF.parent))

import torch
import yaml
from transformers import GenerationConfig

import evaluate as E

QUESTIONS_FILE = REPO / "task1/configs/first_plot_questions.yaml"
MAX_NEW_TOKENS = 512
FORMATS = ["plain", "json", "template"]
CONDITIONS = list(E.CONDITIONS)  # baseline, Full FT midpoint, five final models
WORKER_SPLIT = [  # two GPUs; the three full-weight loads are slow, so they are spread out
    ["baseline", "fullft_0625", "l2sp_1250", "lora16_1250"],
    ["fullft_1250", "lora4_1250", "oplora_1250"],
]


def load_prompts():
    """[{question, format, id, system, user}] in file order (plain questions first)."""
    out = []
    for e in yaml.safe_load(QUESTIONS_FILE.read_text()):
        fmt = next((f for f in ("json", "template") if e["id"].endswith(f"_{f}")), "plain")
        question = e["id"][: -len(fmt) - 1] if fmt != "plain" else e["id"]
        assert len(e["paraphrases"]) == 1, e["id"]
        out.append({"question": question, "format": fmt, "id": e["id"],
                    "system": e.get("system"), "user": e["paraphrases"][0]})
    assert len(out) == 24 and {p["format"] for p in out} == set(FORMATS)
    return out


def chat_prompt(tok, p):
    msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + \
           [{"role": "user", "content": p["user"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def worker(args):
    out_dir = args.out_root / "task1" / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = E.load_tokenizer()
    prompts = load_prompts()
    for cond in args.condition:
        print(f"[{cond}] start on {torch.cuda.get_device_name(0)}")
        t = time.time()
        model, info = E.build_model(cond, args.models_root)
        load_s = time.time() - t
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        gcfg = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=sorted(eos),
                                pad_token_id=tok.pad_token_id)
        rows = []
        t = time.time()
        for p in prompts:
            text = chat_prompt(tok, p)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            new = model.generate(**enc, generation_config=gcfg)[0, enc["input_ids"].shape[1]:].tolist()
            k = next((j for j, x in enumerate(new) if x in eos), None)
            rows.append({**p, "condition": cond, "prompt": text,
                         "response": tok.decode(new[:k], skip_special_tokens=True),
                         "stop_reason": "eos" if k is not None else "length",
                         "n_new_tokens": k if k is not None else len(new),
                         "truncated": k is None})
        gen_s = time.time() - t
        (out_dir / f"part_{cond}.json").write_text(json.dumps(
            {"model": info, "load_seconds": load_s, "generation_seconds": gen_s, "rows": rows}, indent=2))
        print(f"[{cond}] done: {sum(r['truncated'] for r in rows)} truncated, {gen_s:.0f} s")
        del model
        torch.cuda.empty_cache()


def collect(out_dir):
    parts = {c: json.loads((out_dir / f"part_{c}.json").read_text()) for c in CONDITIONS}
    rows = [r for c in CONDITIONS for r in parts[c]["rows"]]
    assert len(rows) == 168
    questions = list(dict.fromkeys(r["question"] for r in rows if r["format"] == "plain"))
    grouped = {q: {f: {r["condition"]: {k: r[k] for k in ("id", "system", "user", "response", "stop_reason",
                                                              "n_new_tokens", "truncated")}
                       for r in rows if r["question"] == q and r["format"] == f}
                   for f in FORMATS} for q in questions}
    meta = {"source": str(QUESTIONS_FILE.relative_to(REPO)), "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy", "enable_thinking": False, "dtype": "float16", "batch_size": 1,
            "conditions": CONDITIONS, "question_order": questions,
            "models": {c: parts[c]["model"] for c in CONDITIONS},
            "seconds": {c: {"load": parts[c]["load_seconds"], "generation": parts[c]["generation_seconds"]}
                        for c in CONDITIONS},
            "n_truncated": {c: sum(r["truncated"] for r in parts[c]["rows"]) for c in CONDITIONS}}
    (out_dir / "responses.json").write_text(json.dumps({"meta": meta, "responses": grouped}, indent=2) + "\n")
    with open(out_dir / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # The manual's annotation subset: first and last underlying questions, all formats and conditions.
    lines = ["# Annotation subset (42 responses)", "",
             f"Questions: `{questions[0]}` and `{questions[-1]}` (first and last in the file's plain order), "
             "3 formats × 7 conditions. Greedy, thinking disabled, 512-new-token cap.", ""]
    for q in (questions[0], questions[-1]):
        for f in FORMATS:
            first = grouped[q][f][CONDITIONS[0]]
            lines += [f"## {q} — {f}", ""]
            if first["system"]:
                lines += ["System:", "", "```", first["system"], "```", ""]
            lines += ["User:", "", "```", first["user"], "```", ""]
            for c in CONDITIONS:
                r = grouped[q][f][c]
                flag = " — TRUNCATED at 512 tokens" if r["truncated"] else ""
                lines += [f"### {c} ({r['n_new_tokens']} tokens{flag})", "", "```", r["response"], "```", ""]
    (out_dir / "annotation_subset.md").write_text("\n".join(lines))
    for c in CONDITIONS:
        (out_dir / f"part_{c}.json").unlink()
    print(f"wrote {out_dir}: 168 responses; truncated per condition {meta['n_truncated']}")


KAGGLE_ARGS = []


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--models-root", nargs="+", default=["/kaggle/input"])
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("."))
    p.add_argument("--condition", nargs="+", choices=CONDITIONS, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.condition:
        worker(args)
        return

    gpus = E.gpu_ids() or ["0"]
    splits = WORKER_SPLIT if len(gpus) >= 2 else [CONDITIONS]
    procs = [subprocess.Popen([sys.executable, str(SELF), "--out-root", str(args.out_root),
                               "--models-root", *args.models_root, "--condition", *conds],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
             for conds, g in zip(splits, gpus)]
    codes = [p.wait() for p in procs]
    assert all(c == 0 for c in codes), f"worker exit codes {codes}"
    collect(args.out_root / "task1" / "qualitative")


if __name__ == "__main__":
    main()
