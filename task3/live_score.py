"""Task 3 Step 4 scoring of the live calls (task3/live/calls.jsonl from task3/live.py).

Per strategy on the 50 questions: accuracy, actual input and output tokens, runtime, truncation rate
(responses that hit their cap) and extraction rate (responses with a boxed answer). Refinement is
charged the shared initial response (parallel response 1) plus the critique-and-revision call with its
longer prompt. Parallel sampling returns the self-consistency answer of its three responses (groups and
ties as offline, DECISIONS.md T3-10 and T3-12). Revision transitions are relative to the shared initial
answer. Paired question bootstrap (1,000 resamples, seed 6304) for accuracy differences between
strategies (T3-8). Same scorer as the offline experiments (math-verify, T3-6).

Usage: python task3/live_score.py [--calls task3/live/calls.jsonl]
Output: task3/results/live/summary.json, per_question.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import answers as A

SEED = 6304
N_BOOT = 1000


def self_consistency(extracted, gold):
    """Group the answers in response order (first matching representative), one vote each; ties to
    the group whose earliest member comes first. Returns (winning answer or None, correct)."""
    reps, votes, first, parsed = [], {}, {}, {}
    for pos, a in enumerate(extracted):
        if a is None:
            key = A.NO_ANSWER
        else:
            parsed.setdefault(a, A.math_parse(a))
            key = next((r for r in reps if A.math_equal(parsed[r], parsed[a])), None)
            if key is None:
                reps.append(a)
                key = a
        votes[key] = votes.get(key, 0) + 1
        first.setdefault(key, pos)
    best = max(votes.values())
    win = min((k for k, v in votes.items() if v == best), key=lambda k: first[k])
    return (None if win == A.NO_ANSWER else win), (win != A.NO_ANSWER and A.math_equal(gold, parsed[win]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calls", type=Path, default=REPO / "task3/live/calls.jsonl")
    a = p.parse_args()
    calls = [json.loads(l) for l in a.calls.open()]
    gold = {q["sample_id"]: q["gold_answer"] for q in json.loads((REPO / "task3/splits/math_live.json").read_text())}
    by_q = {}
    for c in calls:
        by_q.setdefault(c["sample_id"], {})[c["slot"]] = c
    assert len(by_q) == 50 and all(sorted(v) == [0, 1, 2, 3, 4] for v in by_q.values())

    def ok(q, ans):
        return ans is not None and A.math_equal(A.math_parse(gold[q]), A.math_parse(ans))

    rows = []
    for q, c in by_q.items():
        par_ans, par_ok = self_consistency([c[s]["extracted"] for s in (1, 2, 3)], A.math_parse(gold[q]))
        init_ok, rev_ok = ok(q, c[1]["extracted"]), ok(q, c[4]["extracted"])
        strategies = {"single": [c[0]], "parallel": [c[1], c[2], c[3]], "refinement": [c[1], c[4]]}
        correct = {"single": ok(q, c[0]["extracted"]), "parallel": par_ok, "refinement": rev_ok}
        row = {"sample_id": q, "initial_correct": init_ok, "parallel_answer": par_ans}
        for s, cs in strategies.items():
            row.update({f"{s}_correct": correct[s], f"{s}_input_tokens": sum(x["input_tokens"] for x in cs),
                        f"{s}_output_tokens": sum(x["output_tokens"] for x in cs),
                        f"{s}_retained_tokens": sum(x["retained_tokens"] for x in cs),
                        f"{s}_seconds": sum(x["seconds"] for x in cs),
                        f"{s}_responses_capped": sum(x["hit_cap"] for x in cs),
                        f"{s}_responses_extracted": sum(x["extracted"] is not None for x in cs),
                        f"{s}_responses": len(cs)})
        # the answer each strategy returns must itself be extracted (refinement: the revised one)
        row["single_final_extracted"] = c[0]["extracted"] is not None
        row["parallel_final_extracted"] = par_ans is not None
        row["refinement_final_extracted"] = c[4]["extracted"] is not None
        row["initial_hit_cap"] = bool(c[1]["hit_cap"])
        row["revision_changed_answer"] = not (c[1]["extracted"] is not None and c[4]["extracted"] is not None and
                                              A.math_equal(A.math_parse(c[1]["extracted"]), A.math_parse(c[4]["extracted"])))
        row["revision_input_tokens"] = c[4]["input_tokens"]
        row["revision_output_tokens"] = c[4]["output_tokens"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("sample_id")
    out = REPO / "task3/results/live"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_question.csv", index=False)

    summary = {"n_questions": len(df), "strategies": {}}
    for s in ("single", "parallel", "refinement"):
        n_resp = df[f"{s}_responses"].sum()
        summary["strategies"][s] = {
            "accuracy": float(df[f"{s}_correct"].mean()),
            "mean_input_tokens": float(df[f"{s}_input_tokens"].mean()),
            "mean_output_tokens": float(df[f"{s}_output_tokens"].mean()),
            "total_output_tokens": int(df[f"{s}_output_tokens"].sum()),
            "mean_seconds": float(df[f"{s}_seconds"].mean()), "total_seconds": float(df[f"{s}_seconds"].sum()),
            "truncation_rate_responses": float(df[f"{s}_responses_capped"].sum() / n_resp),
            "extraction_rate_responses": float(df[f"{s}_responses_extracted"].sum() / n_resp),
            "final_answer_extraction_rate": float(df[f"{s}_final_extracted"].mean())}
    summary["refinement_extra"] = {"mean_revision_input_tokens": float(df["revision_input_tokens"].mean()),
                                   "mean_revision_output_tokens": float(df["revision_output_tokens"].mean()),
                                   "initial_accuracy": float(df["initial_correct"].mean())}
    i, r = df["initial_correct"], df["refinement_correct"]
    summary["revision_transitions"] = {"wrong_to_correct": int((~i & r).sum()), "correct_to_wrong": int((i & ~r).sum()),
                                       "correct_to_correct": int((i & r).sum()), "wrong_to_wrong": int((~i & ~r).sum())}
    # transitions split by whether the initial response hit its cap and whether the revision changed the
    # answer (an initial response without an extracted answer counts as changed when the revision has one)
    summary["revision_by_initial_state"] = [
        {"initial_hit_cap": bool(k[0]), "revision_changed_answer": bool(k[1]), "initial_correct": bool(k[2]),
         "revised_correct": bool(k[3]), "questions": int(n)}
        for k, n in df.groupby(["initial_hit_cap", "revision_changed_answer", "initial_correct",
                                "refinement_correct"]).size().items()]
    draws = np.random.default_rng(SEED).integers(0, len(df), size=(N_BOOT, len(df)))
    summary["bootstrap_differences"] = {}
    for x, y in (("parallel", "single"), ("refinement", "single"), ("refinement", "parallel")):
        d = df[f"{x}_correct"].to_numpy(float) - df[f"{y}_correct"].to_numpy(float)
        b = d[draws].mean(1)
        summary["bootstrap_differences"][f"{x}_minus_{y}"] = {
            "diff": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    info = a.calls.parent / "run_info.json"
    if info.exists():
        summary["run_info"] = json.loads(info.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
