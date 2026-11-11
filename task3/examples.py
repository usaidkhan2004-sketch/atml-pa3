"""Task 3 Steps 5-6: early-stop analysis and selected examples, from the offline outputs.

Manual, Step 5: "Analyze incorrect early stops, including cases where fixed-32 selection would have
recovered the correct answer." Step 6: "Relate the stopping behavior to answer agreement and to the
fraction of correct candidates in the full pool, using the latter only for retrospective analysis.
Present representative cases of selection disagreement, revision, and premature stopping using a
stated example-selection rule."

On the 800 held-out questions with the frozen adaptive settings (task3/results/offline/<d>/adaptive.json).
  early_stops        per policy: share of (question, order) runs stopped before 32; accuracy of early
                     stops; wrong early stops for which fixed-32 SC would have been correct
  by_pool_fraction   per policy: mean rollouts, early-stop share and accuracy by the full pool's
                     fraction of correct rollouts (retrospective only)
  examples (rule)    walk held-out questions in their permuted order, order 0 only, and take the first
                     two questions of each kind:
                       selection_disagreement  N = 32, SC and the dataset's RWC selector (MATH: RWC
                                               with self-certainty; MathVista: RWC with VisConf) pick
                                               different groups and exactly one is correct
                                               (one example per direction, then a second each)
                       premature_stop          the quality-aware policy stops before 32 with a wrong
                                               answer while fixed-32 SC is correct
Revision examples come from the live run (task3/live_score.py).
Output: task3/results/examples/<dataset>.json
Usage: python task3/examples.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import offline as O

RWC = {"math": "rwc_K", "mathvista": "rwc_Z"}
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
BINS = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.0)]
BIN_LABELS = ["0", "(0, .25]", "(.25, .5]", "(.5, .75]", "(.75, 1)", "1"]


def counts_by_group(q, cand):
    out = {}
    for i in cand:
        name = q.group_names[q.g[i]]
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    out_dir = REPO / "task3/results/examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["math", "mathvista"]:
        qs, split = O.load(name)
        res = REPO / f"task3/results/offline/{name}"
        ev = split["eval"]
        gen = pd.read_parquet(REPO / f"task3/cache/{name}_rollouts.parquet", columns=["sample_id", "rollout_id",
                                                                                      "gold_answer"])
        gold = gen[gen.rollout_id == 0].set_index("sample_id")["gold_answer"].to_dict()
        g = pq.read_table(REPO / f"data/task3/{FOLDERS[name]}/generations.parquet",
                          columns=["sample_id", "rollout_id", "question_text"]).to_pandas()
        texts = g[g.rollout_id == 0].set_index("sample_id")["question_text"].to_dict()
        states = pd.read_csv(res / "adaptive_states.csv.gz", dtype={"sample_id": str})
        s32 = states[states.n == 32].set_index(["sample_id", "order"])["sc_correct"]
        pool_frac = {qid: float(qs[qid].correct.mean()) for qid in ev}
        rec = {"early_stops": {}, "by_pool_fraction": {}, "examples": {}}
        for fam in ("agreement", "quality"):
            stop = pd.read_csv(res / f"adaptive_{fam}_stops.csv.gz", dtype={"sample_id": str})
            stop = stop[stop.sample_id.isin(set(ev))].copy()
            stop["fixed32_correct"] = [bool(s32.loc[(q, o)]) for q, o in zip(stop.sample_id, stop.order)]
            early = stop[stop.n < 32]
            wrong = early[~early.sc_correct.astype(bool)]
            rec["early_stops"][fam] = {
                "runs": len(stop), "early_share": float(len(early) / len(stop)),
                "early_accuracy": float(early.sc_correct.mean()), "late_accuracy": float(stop[stop.n == 32].sc_correct.mean()),
                "wrong_early_stops": int(len(wrong)), "wrong_early_share_of_runs": float(len(wrong) / len(stop)),
                "wrong_early_recovered_by_fixed32": int(wrong.fixed32_correct.sum()),
                "correct_early_lost_by_fixed32": int((early.sc_correct.astype(bool) & ~early.fixed32_correct).sum())}
            stop["pool_fraction"] = stop.sample_id.map(pool_frac)
            rows = {}
            for (lo, hi), lab in zip(BINS, BIN_LABELS):
                part = stop[(stop.pool_fraction > lo) & (stop.pool_fraction <= hi)]
                if len(part):
                    rows[lab] = {"questions": int(part.sample_id.nunique()), "mean_rollouts": float(part.n.mean()),
                                 "early_share": float((part.n < 32).mean()), "accuracy": float(part.sc_correct.mean())}
            rec["by_pool_fraction"][fam] = rows
            if fam == "quality":
                prem = []
                for qid in ev:
                    r = stop[(stop.sample_id == qid) & (stop.order == 0)]
                    if len(r) and r.n.iloc[0] < 32 and not r.sc_correct.iloc[0] and r.fixed32_correct.iloc[0]:
                        q = qs[qid]
                        order = split["orders"][qid][0]
                        n = int(r.n.iloc[0])
                        prem.append({"sample_id": qid, "question": texts[qid][:600], "gold": gold[qid], "stopped_at": n,
                                     "revealed_votes": counts_by_group(q, order[:n]),
                                     "full_pool_votes": counts_by_group(q, order),
                                     "full_pool_fraction_correct": pool_frac[qid]})
                    if len(prem) == 2:
                        break
                rec["examples"]["premature_stop"] = prem
        sel = pd.read_csv(res / "selections.csv.gz", dtype={"sample_id": str})
        sel = sel[(sel.N == 32) & (sel.order == 0)].set_index(["sample_id", "method"])
        dis = {"sc_right": [], "rwc_right": []}
        for qid in ev:
            a, b = sel.loc[(qid, "sc")], sel.loc[(qid, RWC[name])]
            if a.selected_group != b.selected_group and bool(a.correct) != bool(b.correct):
                key = "sc_right" if a.correct else "rwc_right"
                if len(dis[key]) < 2:
                    q = qs[qid]
                    order = split["orders"][qid][0]
                    dis[key].append({"sample_id": qid, "question": texts[qid][:600], "gold": gold[qid],
                                     "sc_answer": a.selected_group, "rwc_answer": b.selected_group,
                                     "votes": counts_by_group(q, order), "rwc_score": RWC[name][4:]})
            if all(len(v) == 2 for v in dis.values()):
                break
        rec["examples"]["selection_disagreement"] = dis
        rec["selection_rule"] = __doc__.split("examples (rule)")[1].split("Revision examples")[0].strip()
        (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print(name, json.dumps({k: rec[k] for k in ("early_stops", "by_pool_fraction")}, indent=1))


if __name__ == "__main__":
    main()
