"""Task 3 Step 3: what the quality signals capture, at N = 32 on the 800 held-out questions.

Manual: "examine within-question associations between rollout length, scores, and correctness. Compare
correctness in each score's top and bottom quartiles within a question, then average the differences
across questions. Include shortest-rollout selection as a length-only control. For MathVista, compute
confidence-vision correlation within each rollout and summarize these correlations across questions.
Compare VisConf with tokenwise fusion [...] Use a fixed set of illustrative trajectories".

With all 32 rollouts revealed, VisConf calibration uses the question's whole pool (the N = 32 candidate
set). Scores: L, K, and for MathVista V, Z (VisConf), Zt (tokenwise fusion); "short" = -length.
  associations  within-question Spearman of length with each score, and of each score with
                correctness (questions with mixed correctness only); averaged across questions
  quartiles     top 8 vs bottom 8 rollouts by score (ties by rollout id), correctness difference,
                averaged over all questions and over mixed questions
  conf-vision   Pearson correlation of c_t and v_t within each rollout (rollouts with >= 3 tokens and
                non-constant series), averaged per question, then summarized across questions
  fusion        checks Zt - Z = Cov_t(c, v) and summarizes the covariance
  trajectories  fixed selection rule (see illustrative()); token series saved for figures
Outputs: task3/results/signals/<dataset>.json, task3/results/signals/mathvista_trajectories.json
Usage: python task3/signals.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import offline as O


def ranks(x):
    return pd.Series(x).rank(method="average").to_numpy()


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def scores_at_32(q, name):
    s = {"L": q.L, "K": q.K, "short": -q.n.astype(float)}
    if name == "mathvista":
        Z, Zt = q.visconf(np.arange(32))
        s.update({"V": q.V, "Z": Z, "Zt": Zt})
    return s


def quartile_diff(score, correct):
    order = np.lexsort((np.arange(32), -score))  # descending score, ties by rollout id
    return float(correct[order[:8]].mean() - correct[order[-8:]].mean())


def illustrative(qs, split, all_scores):
    """Fixed rule: walk the held-out questions in their permuted order and take the first question in
    each category (SC = self-consistency over 32, top-Z = highest VisConf rollout):
      both_right, both_wrong, visconf_right_sc_wrong, sc_right_visconf_wrong.
    For each, keep the highest- and lowest-VisConf rollouts with their c_t and v_t series."""
    picks = {}
    for qid in split["eval"]:
        q = qs[qid]
        sc = q.correct[O.vote(q.g.tolist(), [1] * 32)[0]]
        top = q.correct[int(np.argmax(all_scores[qid]["Z"]))]
        cat = {(True, True): "both_right", (False, False): "both_wrong",
               (False, True): "visconf_right_sc_wrong", (True, False): "sc_right_visconf_wrong"}[(bool(sc), bool(top))]
        if cat not in picks:
            picks[cat] = qid
        if len(picks) == 4:
            break
    out = {}
    for cat, qid in picks.items():
        q = qs[qid]
        pool = np.concatenate(q.tokens[0])
        q05, q95 = np.percentile(pool, [5, 95])
        Z = all_scores[qid]["Z"]
        rec = {"sample_id": qid, "rollouts": []}
        for label, i in (("highest_Z", int(np.argmax(Z))), ("lowest_Z", int(np.argmin(Z)))):
            c = np.clip((q.tokens[0][i] - q05) / max(q95 - q05, 1e-8), 0, 1)
            rec["rollouts"].append({"label": label, "rollout_id": i, "correct": bool(q.correct[i]),
                                    "group": q.group_names[q.g[i]], "n_tokens": int(q.n[i]), "Z": float(Z[i]),
                                    "c": c.round(4).tolist(), "v": q.tokens[1][i].round(4).tolist()})
        out[cat] = rec
    return out


def main():
    out = REPO / "task3/results/signals"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["math", "mathvista"]:
        qs, split = O.load(name)
        ev = split["eval"]
        all_scores = {qid: scores_at_32(qs[qid], name) for qid in ev}
        keys = list(all_scores[ev[0]])
        assoc = {k: {"spearman_with_length": [], "spearman_with_correct": []} for k in keys}
        quart = {k: {"all": [], "mixed": []} for k in keys}
        mixed = 0
        for qid in ev:
            q, s = qs[qid], all_scores[qid]
            is_mixed = 0 < q.correct.sum() < 32
            mixed += is_mixed
            for k in keys:
                assoc[k]["spearman_with_length"].append(spearman(q.n, s[k]))
                if is_mixed:
                    assoc[k]["spearman_with_correct"].append(spearman(s[k], q.correct.astype(float)))
                d = quartile_diff(s[k], q.correct.astype(float))
                quart[k]["all"].append(d)
                if is_mixed:
                    quart[k]["mixed"].append(d)
        rec = {"n_questions": len(ev), "mixed_questions": int(mixed),
               "length_correct_spearman_mixed": float(np.nanmean([spearman(qs[x].n, qs[x].correct.astype(float))
                                                                   for x in ev if 0 < qs[x].correct.sum() < 32])),
               "associations": {k: {m: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                                        "n_defined": int(np.isfinite(v).sum())} for m, v in d.items()}
                                for k, d in assoc.items()},
               "quartile_correctness_difference": {k: {g: float(np.mean(v)) for g, v in d.items()}
                                                   for k, d in quart.items()}}
        if name == "mathvista":
            per_q, n_def, n_undef, cov_err, covs = [], 0, 0, 0.0, []
            for qid in ev:
                q = qs[qid]
                pool = np.concatenate(q.tokens[0])
                q05, q95 = np.percentile(pool, [5, 95])
                rs = []
                for i in range(32):
                    c = np.clip((q.tokens[0][i] - q05) / max(q95 - q05, 1e-8), 0, 1)
                    v = q.tokens[1][i]
                    cov = float(((c - c.mean()) * (v - v.mean())).mean())
                    covs.append(cov)
                    cov_err = max(cov_err, abs(all_scores[qid]["Zt"][i] - all_scores[qid]["Z"][i] - cov))
                    if len(c) >= 3 and c.std() > 0 and v.std() > 0:
                        rs.append(float(np.corrcoef(c, v)[0, 1]))
                        n_def += 1
                    else:
                        n_undef += 1
                if rs:
                    per_q.append(float(np.mean(rs)))
            per_q = np.array(per_q)
            covs = np.array(covs)
            rec["confidence_vision_correlation"] = {
                "rollouts_defined": n_def, "rollouts_undefined": n_undef, "questions": int(len(per_q)),
                "mean_of_question_means": float(per_q.mean()), "median_of_question_means": float(np.median(per_q)),
                "share_questions_positive": float((per_q > 0).mean()),
                "question_mean_quantiles_5_25_75_95": [float(x) for x in np.percentile(per_q, [5, 25, 75, 95])]}
            rec["tokenwise_fusion"] = {"max_abs_error_Zt_minus_Z_vs_cov": cov_err, "cov_mean": float(covs.mean()),
                                       "cov_quantiles_5_50_95": [float(x) for x in np.percentile(covs, [5, 50, 95])],
                                       "share_rollouts_negative_cov": float((covs < 0).mean())}
            (out / "mathvista_trajectories.json").write_text(json.dumps(illustrative(qs, split, all_scores)))
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(name, json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
