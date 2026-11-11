"""Task 3 Step 6, revision examples from the live run (task3/live/calls.jsonl, scored by live_score.py).

Manual, Step 6: "Present representative cases of selection disagreement, revision, and premature
stopping using a stated example-selection rule."

Rule: walk the 50 live questions in live order (task3/splits/math_live.json) and take the first two
questions of each revision transition, relative to the shared initial answer (parallel response 1):
wrong_to_correct, correct_to_wrong, wrong_to_wrong, correct_to_correct. A kind with no such question
is reported empty. Each example keeps the question, the gold answer, both extracted answers, whether
the revision changed the answer (math-verify), and the text of both responses (initial: last 800
characters; revision: first 1,500 characters). The full responses are in calls.jsonl (slots 1 and 4).
Output: task3/results/examples/math_live_revision.json
Usage: python task3/live_examples.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import answers as A

KINDS = ["wrong_to_correct", "correct_to_wrong", "wrong_to_wrong", "correct_to_correct"]


def main():
    live = json.loads((REPO / "task3/splits/math_live.json").read_text())
    calls = {}
    for line in (REPO / "task3/live/calls.jsonl").open():
        c = json.loads(line)
        calls[(c["sample_id"], c["slot"])] = c

    def ok(gold, ans):
        return ans is not None and A.math_equal(A.math_parse(gold), A.math_parse(ans))

    picks = {k: [] for k in KINDS}
    for q in live:
        init, rev = calls[(q["sample_id"], 1)], calls[(q["sample_id"], 4)]
        a, b = ok(q["gold_answer"], init["extracted"]), ok(q["gold_answer"], rev["extracted"])
        kind = f"{'correct' if a else 'wrong'}_to_{'correct' if b else 'wrong'}"
        if len(picks[kind]) == 2:
            continue
        same = (init["extracted"] is not None and rev["extracted"] is not None
                and A.math_equal(A.math_parse(init["extracted"]), A.math_parse(rev["extracted"])))
        picks[kind].append({
            "sample_id": q["sample_id"], "live_index": init["index"], "question": q["question_text"],
            "gold": q["gold_answer"], "initial_answer": init["extracted"], "revised_answer": rev["extracted"],
            "answer_changed": not same and not (init["extracted"] is None and rev["extracted"] is None),
            "initial_hit_cap": init["hit_cap"], "revision_hit_cap": rev["hit_cap"],
            "initial_output_tokens": init["output_tokens"], "revision_output_tokens": rev["output_tokens"],
            "initial_response_tail": init["response"][-800:], "revision_response_head": rev["response"][:1500]})
    rec = {"selection_rule": __doc__.split("Rule: ")[1].split("Output:")[0].strip(), "examples": picks}
    out = REPO / "task3/results/examples/math_live_revision.json"
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    for k, v in picks.items():
        print(k, [(e["sample_id"], e["initial_answer"], e["revised_answer"], e["gold"], e["answer_changed"]) for e in v])


if __name__ == "__main__":
    main()
