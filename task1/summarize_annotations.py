"""Task 1 annotation summary: per-condition counts from task1/results/qualitative/annotations.csv.

The 42 annotated responses are the first and last questions, 3 formats x 7 conditions (DECISIONS.md #27
gives the scale). Format compliance is counted over the JSON and template formats only (plain asks for
no format). Output: task1/results/qualitative/annotation_summary.csv
Usage: python task1/summarize_annotations.py
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
QUAL = REPO / "task1/results/qualitative"
ORDER = ["baseline", "fullft_0625", "fullft_1250", "l2sp_1250", "lora4_1250", "lora16_1250", "oplora_1250"]


def main():
    d = pd.read_csv(QUAL / "annotations.csv")
    assert len(d) == 42
    rows = []
    for c in ORDER:
        x = d[d.condition == c]
        f = x[x.format != "plain"]
        row = {"condition": c, "responses": len(x), "truncated": int(x.truncated.sum())}
        for col, frame in (("format_compliance", f), ("coherent_relevant", x), ("repetition", x),
                           ("unnecessary_math", x), ("inappropriate", x)):
            for v in ("yes", "partly", "no"):
                row[f"{col}_{v}"] = int((frame[col] == v).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(QUAL / "annotation_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
