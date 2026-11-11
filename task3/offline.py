"""Task 3 offline selection on the supplied pools (manual Steps 1, 2 and 5).

Candidate budgets N in {1, 2, 4, 8, 16, 32}; ten saved rollout orders per question, nested prefixes.
Outcomes are averaged over orders within a question, then across questions. Ties in a score or an
answer vote go to the earliest supporting rollout in the current order; RWC uses average ranks for
tied scores. At N = 1 every rule returns the only candidate. Cost = retained tokens of every candidate
the method consumes (FirstAns: its first candidate only).

Methods
  first                      FirstAns
  sc                         self-consistency, one vote per rollout (NO_ANSWER votes like any group)
  high_L, high_K             highest mean log-probability / highest self-certainty
  rwc_K                      self-certainty + rank-weighted consensus
  shortest                   shortest rollout (length-only control, Step 3)
  MathVista only: high_V, rwc_V (vision only), high_Z, rwc_Z (VisConf), high_Zt, rwc_Zt (tokenwise
  fusion ablation, Step 3)
VisConf: token KL pooled over the current candidate set, calibrated with its 5th / 95th percentiles
(linear interpolation), recomputed for every prefix; Z = V * mean(c), Zt = mean(c * v).

Adaptive stopping (Step 5): checks after n in {4, 8, 16} revealed candidates, else 32; both
controllers return the current self-consistency answer. Settings are chosen on the development
questions and frozen before the held-out evaluation.

Outputs (task3/results/offline/<dataset>/): per_question.csv.gz (order-averaged), selections.csv.gz
(every question x order x N x method: selected rollout, group, correct, cost), adaptive.json,
summary.json (eval-split curves, method table, failure decomposition, bootstrap differences).
Usage: python task3/offline.py   (local CPU; needs task3/prepare.py outputs)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(line_buffering=True)

SEED = 6304
BUDGETS = [1, 2, 4, 8, 16, 32]
CHECKS = [4, 8, 16]
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
N_BOOT = 1000
NO_ANSWER = "NO_ANSWER"
QUALITY = {"math": "K", "mathvista": "Z"}  # quality-aware stopping: self-certainty / VisConf


# --------------------------------------------------------------------------- data

class Question:
    """One question's 32 rollouts indexed by rollout id."""

    def __init__(self, sub, tokens=None):
        sub = sub.sort_values("rollout_id")
        assert sub["rollout_id"].tolist() == list(range(32))
        self.n = sub["n_generated_text_tokens"].to_numpy()
        groups = sub["group"].tolist()
        names = list(dict.fromkeys(groups))
        self.group_names = names
        self.g = np.array([names.index(x) for x in groups])
        self.no_answer = names.index(NO_ANSWER) if NO_ANSWER in names else -1
        self.correct = sub["correct"].to_numpy().astype(bool)
        self.L, self.K = sub["L"].to_numpy(), sub["K"].to_numpy()
        self.V = sub["V"].to_numpy() if "V" in sub else None
        self.tokens = tokens  # (kl list, v list) per rollout id, MathVista only

    def visconf(self, cand):
        """Z = V * mean(c) and Zt = mean(c * v) for the candidates `cand`, with c calibrated on the
        pooled token KL of exactly these candidates."""
        kl = [self.tokens[0][i] for i in cand]
        pool = np.concatenate(kl)
        q05, q95 = np.percentile(pool, [5, 95])
        den = max(q95 - q05, 1e-8)
        Z, Zt = np.empty(len(cand)), np.empty(len(cand))
        for j, i in enumerate(cand):
            c = np.clip((self.tokens[0][i] - q05) / den, 0.0, 1.0)
            Z[j] = self.V[i] * c.mean()
            Zt[j] = (c * self.tokens[1][i]).mean()
        return Z, Zt


def load(name):
    df = pd.read_parquet(REPO / f"task3/cache/{name}_rollouts.parquet")
    split = json.loads((REPO / f"task3/splits/{name}.json").read_text())
    tok = None
    if name == "mathvista":
        z = np.load(REPO / "task3/cache/mathvista_tokens.npz")
        off = z["offsets"]
        tok = {}
        for r, (q, rid) in enumerate(zip(z["sample_id"], z["rollout_id"])):
            tok.setdefault(q, {})[int(rid)] = (z["kl"][off[r]:off[r + 1]].astype(np.float64),
                                               z["v"][off[r]:off[r + 1]].astype(np.float64))
    qs = {}
    for q, sub in df.groupby("sample_id"):
        t = None if tok is None else ([tok[q][i][0] for i in range(32)], [tok[q][i][1] for i in range(32)])
        qs[q] = Question(sub, t)
    return qs, split


# --------------------------------------------------------------------------- rules

def highest(scores):
    """Position of the highest score; np.argmax returns the first maximum, i.e. the earliest."""
    return int(np.argmax(scores))


def vote(groups, weights):
    """Winner position for weighted votes: the group with the largest total, ties to the group whose
    earliest member comes first; returns (position of that earliest member, totals by group)."""
    totals, first = {}, {}
    for pos, (g, w) in enumerate(zip(groups, weights)):
        totals[g] = totals.get(g, 0) + w
        first.setdefault(g, pos)
    best = max(totals.values())
    winner = min((first[g] for g, t in totals.items() if t == best))
    return winner, totals


def descending_average_ranks(scores):
    """Rank 1 = highest score; tied scores share the mean of the ranks they span."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def rwc_weights2(scores):
    """2 (N - r_i) with r_i the descending average rank (rank 1 best): integers, so vote totals compare
    exactly; proportional to w_i = (N - r_i) / (N - 1)."""
    N = len(scores)
    r = descending_average_ranks(scores)
    return np.rint(2 * (N - r)).astype(np.int64)


def rwc(scores, groups):
    if len(scores) == 1:
        return 0, {groups[0]: 1}
    return vote(groups, rwc_weights2(scores))


def run_prefix(q, cand, name):
    """Every method on the candidate prefix `cand` (rollout ids in revealed order)."""
    cand = np.asarray(cand)
    groups = q.g[cand].tolist()
    sel = {"first": 0, "sc": vote(groups, [1] * len(cand))[0],
           "high_L": highest(q.L[cand]), "high_K": highest(q.K[cand]),
           "rwc_K": rwc(q.K[cand], groups)[0], "shortest": int(np.argmin(q.n[cand]))}
    if name == "mathvista":
        Z, Zt = q.visconf(cand)
        sel.update({"high_V": highest(q.V[cand]), "rwc_V": rwc(q.V[cand], groups)[0],
                    "high_Z": highest(Z), "rwc_Z": rwc(Z, groups)[0],
                    "high_Zt": highest(Zt), "rwc_Zt": rwc(Zt, groups)[0]})
    return {m: int(cand[p]) for m, p in sel.items()}


# --------------------------------------------------------------------------- fixed budgets

def fixed_budget(qs, split, name):
    rows = []
    for qid, q in qs.items():
        for o, order in enumerate(split["orders"][qid]):
            for N in BUDGETS:
                cand = order[:N]
                cost_all = int(q.n[cand].sum())
                oracle = bool(q.correct[cand].any())
                distinct = len(set(q.g[cand].tolist()))
                for m, rid in run_prefix(q, cand, name).items():
                    rows.append((qid, o, N, m, rid, q.group_names[q.g[rid]], bool(q.correct[rid]),
                                 int(q.n[cand[0]]) if m == "first" else cost_all, oracle, distinct))
    return pd.DataFrame(rows, columns=["sample_id", "order", "N", "method", "selected_rollout", "selected_group",
                                       "correct", "cost", "oracle", "distinct_groups"])


def bootstrap_diff(a, b, rng_draws):
    """Paired question bootstrap of mean(a - b), a and b order-averaged per question."""
    d = np.asarray(a) - np.asarray(b)
    boots = d[rng_draws].mean(1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def summarize_fixed(sel, split):
    per_q = sel.groupby(["sample_id", "N", "method"], as_index=False)[["correct", "cost", "oracle", "distinct_groups"]].mean()
    ev = per_q[per_q["sample_id"].isin(set(split["eval"]))]
    table = ev.groupby(["N", "method"])[["correct", "cost", "oracle", "distinct_groups"]].mean()
    out = {"n_questions": int(ev["sample_id"].nunique()), "by_budget": {}}
    # failure decomposition: no correct candidate vs correct candidate missed (order-averaged per question)
    ev = ev.assign(no_correct=1 - ev["oracle"], missed=ev["oracle"] - ev["correct"])
    dec = ev.groupby(["N", "method"])[["no_correct", "missed"]].mean()
    qids = sorted(ev["sample_id"].unique())
    draws = np.random.default_rng(SEED).integers(0, len(qids), size=(N_BOOT, len(qids)))
    for N in BUDGETS:
        rec = {}
        sc = ev[(ev.N == N) & (ev.method == "sc")].set_index("sample_id").loc[qids, "correct"]
        for m in table.loc[N].index:
            t, d = table.loc[(N, m)], dec.loc[(N, m)]
            acc = ev[(ev.N == N) & (ev.method == m)].set_index("sample_id").loc[qids, "correct"]
            rec[m] = {"accuracy": float(t["correct"]), "oracle_coverage": float(t["oracle"]),
                      "mean_cost_tokens": float(t["cost"]), "distinct_groups": float(t["distinct_groups"]),
                      "fail_no_correct_candidate": float(d["no_correct"]), "fail_correct_missed": float(d["missed"]),
                      "minus_sc": None if m == "sc" else bootstrap_diff(acc, sc, draws)}
        out["by_budget"][str(N)] = rec
    return per_q, out


# --------------------------------------------------------------------------- adaptive stopping

def adaptive_states(qs, split, name):
    """For every question, order and check point n: the SC answer, whether its leader is unique and
    not NO_ANSWER, its vote share, whether SC and RWC agree, and the RWC margin (quality score)."""
    score = QUALITY[name]
    rows = []
    for qid, q in qs.items():
        for o, order in enumerate(split["orders"][qid]):
            for n in CHECKS + [32]:
                cand = np.asarray(order[:n])
                groups = q.g[cand].tolist()
                sc_pos, counts = vote(groups, [1] * n)
                top = sorted(counts.values(), reverse=True)
                leader = q.g[cand[sc_pos]]
                unique = len(top) == 1 or top[0] > top[1]
                s = q.visconf(cand)[0] if score == "Z" else q.K[cand]
                rwc_pos, W = rwc(s, groups)
                wt = sorted(W.values(), reverse=True)
                margin = (wt[0] - (wt[1] if len(wt) > 1 else 0)) / sum(W.values())
                rows.append((qid, o, n, int(cand[sc_pos]), bool(q.correct[cand[sc_pos]]), int(q.n[cand].sum()),
                             unique and leader != q.no_answer, top[0] / n, q.g[cand[rwc_pos]] == leader, margin))
    return pd.DataFrame(rows, columns=["sample_id", "order", "n", "sc_rollout", "sc_correct", "cost", "unique_leader",
                                       "leader_share", "sc_equals_rwc", "rwc_margin"])


def simulate(states, tau, gamma=None):
    """Stop at the first check n in (4, 8, 16) whose condition holds, else at 32. gamma None =
    agreement-based; otherwise quality-aware. Returns one row per (question, order)."""
    s = states
    ok = s["unique_leader"] & (s["leader_share"] >= tau - 1e-12)
    if gamma is not None:
        ok &= s["sc_equals_rwc"] & (s["rwc_margin"] >= gamma - 1e-12)
    ok |= s["n"] == 32
    stop = s[ok].sort_values("n").groupby(["sample_id", "order"], as_index=False).first()
    return stop


def adaptive(states, split, sc32_dev):
    out = {"settings": {}, "selected": {}, "eval": {}}
    dev, ev = set(split["dev"]), set(split["eval"])

    def score(stop, ids):
        part = stop[stop["sample_id"].isin(ids)]
        per_q = part.groupby("sample_id")[["sc_correct", "cost", "n"]].mean()
        return {"accuracy": float(per_q["sc_correct"].mean()), "mean_cost_tokens": float(per_q["cost"].mean()),
                "mean_rollouts": float(per_q["n"].mean()),
                "stop_distribution": {str(k): float(v) for k, v in part["n"].value_counts(normalize=True).sort_index().items()}}

    families = {"agreement": [(t, None) for t in TAUS], "quality": [(t, g) for t in TAUS for g in GAMMAS]}
    for fam, grid in families.items():
        cands = []
        for tau, gamma in grid:
            r = score(simulate(states, tau, gamma), dev)
            out["settings"][f"{fam}|tau={tau}|gamma={gamma}"] = r
            cands.append((tau, gamma, r))
        fixed32 = score(states[states["n"] == 32], dev)
        cands.append((None, None, fixed32))  # the always-32 fallback
        ok = [c for c in cands if c[2]["accuracy"] >= sc32_dev - 0.01 - 1e-12]
        # lowest mean cost; ties: higher accuracy, then smaller tau and gamma (fallback last)
        best = min(ok, key=lambda c: (c[2]["mean_cost_tokens"], -c[2]["accuracy"],
                                      c[0] if c[0] is not None else 9, c[1] if c[1] is not None else 9))
        out["selected"][fam] = {"tau": best[0], "gamma": best[1], "dev": best[2]}
    return out, score


def main():
    root = REPO / "task3/results/offline"
    for name in ["math", "mathvista"]:
        t0 = time.time()
        qs, split = load(name)
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        sel = fixed_budget(qs, split, name)
        sel.to_csv(out / "selections.csv.gz", index=False)
        per_q, summary = summarize_fixed(sel, split)
        per_q.to_csv(out / "per_question.csv.gz", index=False)
        print(name, "fixed budgets", round(time.time() - t0), "s")

        states = adaptive_states(qs, split, name)
        states.to_csv(out / "adaptive_states.csv.gz", index=False)
        s32 = states[states["n"] == 32]
        sc32_dev = float(s32[s32["sample_id"].isin(set(split["dev"]))].groupby("sample_id")["sc_correct"].mean().mean())
        ad, score = adaptive(states, split, sc32_dev)
        ad["fixed32_sc_dev_accuracy"] = sc32_dev
        qids = sorted(split["eval"])
        draws = np.random.default_rng(SEED).integers(0, len(qids), size=(N_BOOT, len(qids)))
        base = s32.groupby("sample_id")["sc_correct"].mean().loc[qids]
        for fam, chosen in ad["selected"].items():
            stop = (simulate(states, chosen["tau"], chosen["gamma"] if fam == "quality" else None)
                    if chosen["tau"] is not None else s32)
            res = score(stop, set(qids))
            acc = stop[stop["sample_id"].isin(set(qids))].groupby("sample_id")["sc_correct"].mean().loc[qids]
            res["minus_fixed32_sc"] = bootstrap_diff(acc, base, draws)
            ad["eval"][fam] = res
            stop.to_csv(out / f"adaptive_{fam}_stops.csv.gz", index=False)
        ad["eval"]["fixed32_sc"] = score(s32, set(qids))
        (out / "adaptive.json").write_text(json.dumps(ad, indent=1))
        summary["seconds"] = time.time() - t0
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(name, "done", round(time.time() - t0), "s;", json.dumps(ad["selected"]))


if __name__ == "__main__":
    main()
