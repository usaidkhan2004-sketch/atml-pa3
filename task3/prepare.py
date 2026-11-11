"""Task 3 Step 1 preparation: splits, rollout orders, joined tables, answer groups and correctness.

Manual: "Load and join the supplied tables, verify token-list alignment, and prepare answer groups
and correctness labels." Splits: "sort the 1,000 question IDs lexicographically and permute them with
NumPy's default_rng(6304). Use the first 200 questions for development and the remaining 800 for
held-out evaluation [...] Use the first 50 MATH evaluation questions in this permutation for the live
generation subtask." Orders: "ten seeded random permutations of the 32 rollout IDs per question".

Per rollout (task3/cache/<dataset>_rollouts.parquet):
  n_tokens, hit_cap, extracted, key (group key), group (index within the question), correct,
  L = mean logp, K = mean kl_u_p, and for MathVista V = mean v_t with
  v_t = img / (img + prompt_text + generated_text).
Token arrays for VisConf (MathVista): task3/cache/mathvista_tokens.npz (kl, v, offsets by row).

Answer groups are formed without gold answers. MATH: unique extracted strings in rollout-ID order;
each joins the first earlier group whose representative math-verify accepts, else starts a group.
MathVista: equal normalized keys (DECISIONS.md T3-3 to T3-5). NO_ANSWER is one group, graded wrong.

Outputs: task3/splits/<dataset>.json, task3/cache/*, task3/results/prepare_summary.json
Usage: python task3/prepare.py      (local CPU; reads data/task3/)
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

import answers as A

SEED = 6304
N_DEV, N_LIVE, N_ROLLOUTS, N_ORDERS = 200, 50, 32, 10
DATA = REPO / "data/task3"
DATASETS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
MV_META = DATA / "mathvista_meta/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
KEYS = ["dataset", "sample_id", "rollout_id"]


def splits_and_orders(ids):
    """Lexicographic sort, one default_rng(6304) permutation; then, from a second default_rng(6304)
    stream, ten permutations of rollout IDs 0..31 per question in sorted-ID order (T3-9)."""
    ids = sorted(ids)
    perm = [ids[k] for k in np.random.default_rng(SEED).permutation(len(ids))]
    rng = np.random.default_rng(SEED)
    orders = {q: [rng.permutation(N_ROLLOUTS).tolist() for _ in range(N_ORDERS)] for q in ids}
    return {"seed": SEED, "permuted_ids": perm, "dev": perm[:N_DEV], "eval": perm[N_DEV:]}, orders


def load_joined(name):
    folder = DATA / DATASETS[name]
    g = pq.read_table(folder / "generations.parquet").to_pandas()
    p = pq.read_table(folder / "probability_metrics.parquet").to_pandas()
    df = g.merge(p, on=KEYS, how="inner", validate="one_to_one")
    assert len(df) == len(g) == len(p) == 32_000
    if name == "mathvista":
        a = pq.read_table(folder / "attention_metrics.parquet").to_pandas()
        a = a.rename(columns={"n_generated_text_tokens": "attn_n_prev_tokens"})
        df = df.merge(a, on=KEYS, how="inner", validate="one_to_one")
        assert len(df) == 32_000
    return df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)


def check_alignment(df, name):
    """Every per-token list has n_generated_text_tokens entries; the attention file's preceding-token
    counts run 0..N-1 (README)."""
    n = df["n_generated_text_tokens"].to_numpy()
    cols = ["generated_token_ids", "logp", "kl_u_p"]
    if name == "mathvista":
        cols += ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total", "attn_n_prev_tokens"]
    bad = {c: int(sum(len(x) != k for x, k in zip(df[c], n))) for c in cols}
    out = {"rows": len(df), "list_length_mismatches": bad, "min_tokens": int(n.min())}
    if name == "mathvista":
        out["prev_token_count_mismatches"] = int(sum(not np.array_equal(np.asarray(x), np.arange(len(x)))
                                                     for x in df["attn_n_prev_tokens"]))
        out["first_token_generated_attention_nonzero"] = int(sum(float(x[0]) != 0 for x in df["generated_text_attn_total"]))
    assert all(v == 0 for v in bad.values()), bad
    return out


def rollout_signals(df, name):
    df["L"] = [float(np.mean(x)) for x in df["logp"]]
    df["K"] = [float(np.mean(x)) for x in df["kl_u_p"]]
    rec = {}
    if name == "mathvista":
        v_all, kl_all, offsets, zero_den = [], [], [0], 0
        for img, pr, ge, kl in zip(df["img_attn_total"], df["prompt_text_attn_total"], df["generated_text_attn_total"],
                                   df["kl_u_p"]):
            img, den = np.asarray(img, np.float64), np.asarray(img, np.float64) + np.asarray(pr) + np.asarray(ge)
            zero_den += int((den <= 0).sum())
            v = np.divide(img, den, out=np.zeros_like(img), where=den > 0)
            v_all.append(v.astype(np.float32))
            kl_all.append(np.asarray(kl, np.float32))
            offsets.append(offsets[-1] + len(v))
        df["V"] = [float(v.mean()) for v in v_all]
        rec["zero_attention_denominators"] = zero_den
        tokens = {"kl": np.concatenate(kl_all), "v": np.concatenate(v_all), "offsets": np.asarray(offsets, np.int64),
                  "sample_id": df["sample_id"].to_numpy().astype(str), "rollout_id": df["rollout_id"].to_numpy()}
        return rec, tokens
    return rec, None


def math_groups(df):
    """Answer groups and correctness for MATH (T3-6). Also counts where the greedy grouping would
    depend on order: a string equivalent to several groups' representatives."""
    from math_verify import parse  # noqa: F401  (imported to fail early if missing)
    keys, groups, correct = [None] * len(df), [None] * len(df), [None] * len(df)
    multi, gold_unparsed, asym = 0, 0, 0
    for q, sub in df.groupby("sample_id", sort=False):
        gold = A.math_parse(sub["gold_answer"].iloc[0])
        gold_unparsed += len(gold) == 0
        reps, parsed, rep_of, verdict = [], {}, {}, {}
        for idx, ans in zip(sub.index, sub["extracted_answer"]):
            if ans is None or pd.isna(ans):
                keys[idx], groups[idx], correct[idx] = A.NO_ANSWER, A.NO_ANSWER, False
                continue
            if ans not in rep_of:
                parsed[ans] = A.math_parse(ans)
                hits = [r for r in reps if A.math_equal(parsed[r], parsed[ans])]
                multi += len(hits) > 1
                asym += any(not A.math_equal(parsed[ans], parsed[r]) for r in hits)
                if hits:
                    rep_of[ans] = hits[0]
                else:
                    reps.append(ans)
                    rep_of[ans] = ans
                    verdict[ans] = A.math_equal(gold, parsed[ans])
            keys[idx], groups[idx], correct[idx] = ans, rep_of[ans], verdict[rep_of[ans]]
    df["key"], df["group"], df["correct"] = keys, groups, correct
    return {"gold_unparsed": gold_unparsed, "strings_matching_several_groups": multi,
            "asymmetric_verdicts_with_representative": asym}


def mathvista_groups(df):
    meta = pq.read_table(MV_META, columns=["pid", "choices", "precision", "answer", "question_type", "answer_type"]
                         ).to_pandas().set_index("pid")
    rows = [meta.loc[q] for q in df["sample_id"]]
    df["question_type"] = [m["question_type"] for m in rows]
    df["answer_type"] = [m["answer_type"] for m in rows]
    df["key"] = [A.mathvista_key(None if (e is None or pd.isna(e)) else e, m) for e, m in zip(df["extracted_answer"], rows)]
    gold = [A.mathvista_gold_key(m) for m in rows]
    df["group"] = df["key"]
    df["correct"] = [k != A.NO_ANSWER and A.safe_equal(k, g) for k, g in zip(df["key"], gold)]
    supplied_gold_mismatch = int(sum(a != b for a, b in zip(df["gold_answer"], gold)))
    mc = df["question_type"] == "multi_choice"
    ext = df["extracted_answer"].notna()
    letters = [chr(ord("A") + i) for i in range(26)]
    return {"supplied_gold_differs_from_metadata": supplied_gold_mismatch,
            "mc_rollouts": int(mc.sum()), "mc_null": int((mc & ~ext).sum()),
            "mc_nearest_choice_used": int(sum(1 for e, m, r in zip(df["extracted_answer"], mc, rows)
                                              if m and isinstance(e, str) and
                                              e.strip() not in letters[:len(r["choices"])] and
                                              not re.findall(r'\(([a-zA-Z])\)', e))),
            "free_form_unparsed_to_no_answer": int((~mc & ext & (df["key"] == A.NO_ANSWER)).sum())}



def summarize(df):
    by_q = df.groupby("sample_id")
    return {"rollouts": len(df), "questions": int(by_q.ngroups),
            "no_answer_rollouts": int((df["key"] == A.NO_ANSWER).sum()),
            "capped_rollouts": int(df["hit_max_new_tokens"].sum()),
            "rollout_accuracy": float(df["correct"].mean()),
            "oracle_coverage_32": float(by_q["correct"].any().mean()),
            "mean_groups_32_incl_no_answer": float(by_q["group"].nunique().mean()),
            "retained_tokens": int(df["n_generated_text_tokens"].sum())}


def main():
    out_cache, out_splits, out_res = REPO / "task3/cache", REPO / "task3/splits", REPO / "task3/results"
    for d in (out_cache, out_splits, out_res):
        d.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED}
    for name in DATASETS:
        t0 = time.time()
        df = load_joined(name)
        rec = {"alignment": check_alignment(df, name)}
        split, orders = splits_and_orders(df["sample_id"].unique().tolist())
        if name == "math":
            split["live"] = split["eval"][:N_LIVE]
        (out_splits / f"{name}.json").write_text(json.dumps({**split, "orders": orders}))
        sig, tokens = rollout_signals(df, name)
        rec["signals"] = sig
        rec["grouping"] = math_groups(df) if name == "math" else mathvista_groups(df)
        rec["all_questions"] = summarize(df)
        for part in ("dev", "eval"):
            rec[f"{part}_questions"] = summarize(df[df["sample_id"].isin(set(split[part]))])
        if tokens is not None:
            np.savez(out_cache / f"{name}_tokens.npz", **tokens)
        keep = ["sample_id", "rollout_id", "n_generated_text_tokens", "hit_max_new_tokens", "extracted_answer",
                "gold_answer", "key", "group", "correct", "L", "K"] + (["V", "question_type", "answer_type"]
                                                                        if name == "mathvista" else [])
        df[keep].to_parquet(out_cache / f"{name}_rollouts.parquet", index=False)
        rec["seconds"] = time.time() - t0
        summary[name] = rec
        print(name, json.dumps(rec, default=str))
        del df, tokens
    import importlib.metadata as md
    summary["versions"] = {p: md.version(p) for p in ("math-verify", "latex2sympy2_extended", "pyarrow", "numpy",
                                                       "pandas", "Levenshtein")}
    (out_res / "prepare_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
