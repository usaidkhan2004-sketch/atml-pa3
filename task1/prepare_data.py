"""Task 1 data preparation.

Selects the first 40,000 eligible MetaMathQA examples in source order, tokenizes them with
Qwen3's chat template (thinking disabled), builds response-only loss masks, and samples the
activation-probe set. Runs on CPU; no model weights are needed. Run on Kaggle with
`--out-root /kaggle/working`; the repo layout below is recreated under that root.

Eligibility (manual, Task 1 "Training data"):
  - non-empty query and response
  - formatted length <= 2,048 tokens
  - neither `query` nor `original_question` matches a GSM8K test question after normalization

Outputs
  task1/splits/metamath_selected_indices.json   source row indices of the 40,000 examples, in order
  task1/splits/metamath_exclusions.json         excluded source rows (before the cutoff) and reasons
  task1/splits/probe_ids.json                   32 GSM8K test + 32 HellaSwag validation indices
  task1/results/data_prep_summary.json          counts, revisions, length statistics
  data/task1/train_tokenized.parquet            input_ids / labels for training (not committed)
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

SEED = 6304
MODEL_ID = "Qwen/Qwen3-0.6B"
METAMATH_ID = "meta-math/MetaMathQA"
GSM8K_ID, GSM8K_CONFIG = "openai/gsm8k", "main"
HELLASWAG_ID = "Rowan/hellaswag"
N_TRAIN = 40_000
MAX_LEN = 2_048
N_PROBE = 32
IGNORE_INDEX = -100



def normalize(text: str) -> str:
    """NFKC, lowercase, collapse whitespace, strip (DECISIONS.md, Task 1 #4)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def revision(repo_id: str, repo_type: str) -> str:
    api = HfApi()
    info = api.dataset_info(repo_id) if repo_type == "dataset" else api.model_info(repo_id)
    return info.sha


def build_example(tok, query: str, response: str):
    """Return (input_ids, labels) for one user/assistant pair.

    Prompt = chat template up to and including the empty think block that Qwen3 inserts when
    thinking is disabled; it is masked. Supervised = response text + <|im_end|>. The template's
    trailing newline after <|im_end|> is not part of the training sequence.
    """
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full_text = tok.apply_chat_template(
        [{"role": "user", "content": query}, {"role": "assistant", "content": response}],
        tokenize=False, enable_thinking=False,
    )
    target_text = full_text[len(prompt_text):]
    if not full_text.startswith(prompt_text) or not target_text.endswith("<|im_end|>\n"):
        raise ValueError("chat template layout differs from the expected prompt + response + <|im_end|>\\n")
    target_text = target_text[: -len("\n")]

    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    return input_ids, labels, prompt_text + target_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-boundary-every", type=int, default=100,
                        help="compare split vs. joint tokenization on every k-th kept example")
    on_kaggle = Path("/kaggle/working").is_dir()
    parser.add_argument("--out-root", type=Path,
                        default=Path("/kaggle/working") if on_kaggle else Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    SPLITS = args.out_root / "task1" / "splits"
    RESULTS = args.out_root / "task1" / "results"
    DATA = args.out_root / "data" / "task1"

    for d in (SPLITS, RESULTS, DATA):
        d.mkdir(parents=True, exist_ok=True)

    revisions = {
        MODEL_ID: revision(MODEL_ID, "model"),
        METAMATH_ID: revision(METAMATH_ID, "dataset"),
        GSM8K_ID: revision(GSM8K_ID, "dataset"),
        HELLASWAG_ID: revision(HELLASWAG_ID, "dataset"),
    }
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=revisions[MODEL_ID])
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")

    gsm8k_test = load_dataset(GSM8K_ID, GSM8K_CONFIG, split="test", revision=revisions[GSM8K_ID])
    hellaswag_val = load_dataset(HELLASWAG_ID, split="validation", revision=revisions[HELLASWAG_ID])
    gsm8k_norm = {normalize(q) for q in gsm8k_test["question"]}

    metamath = load_dataset(METAMATH_ID, split="train", revision=revisions[METAMATH_ID])

    kept_idx, rows, exclusions = [], [], []
    boundary_checks = boundary_mismatches = 0
    for i, ex in enumerate(metamath):
        if len(kept_idx) == N_TRAIN:
            break
        query, response = ex["query"] or "", ex["response"] or ""
        if not query.strip() or not response.strip():
            exclusions.append({"index": i, "reason": "empty"})
            continue
        if normalize(query) in gsm8k_norm or normalize(ex["original_question"] or "") in gsm8k_norm:
            exclusions.append({"index": i, "reason": "gsm8k_test_overlap"})
            continue
        input_ids, labels, text = build_example(tok, query, response)
        if len(input_ids) > MAX_LEN:
            exclusions.append({"index": i, "reason": "too_long", "length": len(input_ids)})
            continue
        if len(kept_idx) % args.check_boundary_every == 0:
            boundary_checks += 1
            boundary_mismatches += tok(text, add_special_tokens=False)["input_ids"] != input_ids
        assert input_ids[-1] == eos_id and labels[-1] == eos_id
        kept_idx.append(i)
        rows.append({"source_index": i, "input_ids": input_ids, "labels": labels})

    if len(kept_idx) != N_TRAIN:
        raise RuntimeError(f"only {len(kept_idx)} eligible examples found")

    lengths = np.array([len(r["input_ids"]) for r in rows])
    n_supervised = np.array([sum(l != IGNORE_INDEX for l in r["labels"]) for r in rows])
    pd.DataFrame(rows).to_parquet(DATA / "train_tokenized.parquet", index=False)

    rng = np.random.default_rng(SEED)
    probe = {
        "seed": SEED,
        "method": "numpy.random.default_rng(6304); choice(len(gsm8k_test), 32, replace=False), "
                  "then choice(len(hellaswag_validation), 32, replace=False)",
        "gsm8k_test": rng.choice(len(gsm8k_test), N_PROBE, replace=False).tolist(),
        "hellaswag_validation": rng.choice(len(hellaswag_val), N_PROBE, replace=False).tolist(),
    }

    reasons = pd.Series([e["reason"] for e in exclusions]).value_counts().to_dict()
    summary = {
        "revisions": revisions,
        "n_selected": len(kept_idx),
        "source_rows_scanned": kept_idx[-1] + 1,
        "excluded_before_cutoff": len(exclusions),
        "excluded_by_reason": reasons,
        "gsm8k_test_size": len(gsm8k_test),
        "hellaswag_validation_size": len(hellaswag_val),
        "length_tokens": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                          "max": int(lengths.max()), "total": int(lengths.sum())},
        "supervised_tokens": {"mean": float(n_supervised.mean()), "total": int(n_supervised.sum())},
        "boundary_checks": boundary_checks,
        "boundary_mismatches": int(boundary_mismatches),
    }

    (SPLITS / "metamath_selected_indices.json").write_text(json.dumps(kept_idx))
    (SPLITS / "metamath_exclusions.json").write_text(json.dumps(exclusions, indent=1))
    (SPLITS / "probe_ids.json").write_text(json.dumps(probe, indent=2))
    (RESULTS / "data_prep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
