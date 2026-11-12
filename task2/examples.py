"""Task 2 illustrative complementary-image examples from the final test (manual, Step 6: "Present complementary
images with the shared question, references, and predictions, using a stated selection rule for illustrative
examples").

Selection rule, fixed before looking at the examples: for each suite, walk the test pairs in split-file order
and take the first pair of each kind, judged on the correct-image predictions with the pair-accuracy criterion
(normalized prediction equals the image's canonical answer):
  both_right     both predictions equal their own canonical answer
  same_answer    the two predictions are identical, so the answer did not change with the image
  changed_wrong  the predictions differ but at least one is wrong
Grounding pairs flagged as likely mislabelled (DECISIONS.md T2-32) are skipped.
Each example shows both images, the shared question, each image's canonical answer (VQAv2: also the ten
annotator answers), the prediction with that image, and the no-image prediction for the question. Under the
complementary swap a question receives its partner's image, and the input is then token-identical to the
partner's correct-image input, so its prediction is the partner's correct-image prediction.

Images are fetched once from images.cocodataset.org into data/task2/images/ (gitignored).
Outputs: task2/results/examples/{examples.json, complementary_<suite>.png}
Usage: python task2/examples.py
"""

import csv
import json
import textwrap
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "task2/results"
IMG_DIR = REPO / "data/task2/images"
KINDS = ["both_right", "same_answer", "changed_wrong"]
TITLES = {"both_right": "Both answers right", "same_answer": "Same answer for both images",
          "changed_wrong": "Answer changes, but not correctly"}
SUITES = [("vqa", "VQAv2"), ("grounding", "Grounding")]
INK, MUTED = "#2b2b29", "#6b6b66"


def image(file):
    p = IMG_DIR / file
    if not p.exists():
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"http://images.cocodataset.org/val2014/{file}", timeout=60) as r:
            p.write_bytes(r.read())
    return Image.open(p).convert("RGB")


def main():
    preds = [json.loads(l) for l in open(RES / "eval_test/predictions.jsonl")]
    by = {(r["suite"], r["condition"], r["pair_id"], r["image_id"]): r for r in preds}
    with open(RES / "label_inspection/inspection.csv") as f:
        flagged = {r["pair_id"] for r in csv.DictReader(f) if r["split"] == "test" and r["verdict"] == "likely_wrong"}
    out = {"selection_rule": __doc__.split("Selection rule, fixed before looking at the examples: ")[1]
           .split("Images are fetched")[0].strip(), "suites": {}}
    for suite, name in SUITES:
        pairs = json.load(open(REPO / f"task2/splits/{suite}_test.json"))
        picked = {}
        for p in pairs:
            if p["pair_id"] in flagged:
                continue
            a, b = p["items"]
            ra, rb = (by[(suite, "correct", p["pair_id"], it["image_id"])] for it in (a, b))
            ok = [ra["norm"] == normalize_canon(ra), rb["norm"] == normalize_canon(rb)]
            kind = ("both_right" if all(ok) else "same_answer" if ra["norm"] == rb["norm"] else "changed_wrong")
            if kind in picked:
                continue
            q = p.get("question") or a.get("question")
            no_img = by[(suite, "no_image", p["pair_id"], a["image_id"])]["generated"]
            picked[kind] = {"pair_id": p["pair_id"], "type": p.get("answer_type") or p.get("type"), "question": q,
                            "no_image_prediction": no_img,
                            "images": [{"image_id": it["image_id"], "file": it["file"],
                                        "canonical": r["canonical"], "answers": r.get("answers"),
                                        "prediction": r["generated"], "right": bool(k)}
                                       for it, r, k in zip((a, b), (ra, rb), ok)]}
            if len(picked) == len(KINDS):
                break
        out["suites"][suite] = {k: picked.get(k) for k in KINDS}
        figure(suite, name, out["suites"][suite])
    (RES / "examples").mkdir(parents=True, exist_ok=True)
    (RES / "examples/examples.json").write_text(json.dumps(out, indent=1))
    for s, d in out["suites"].items():
        for k, e in d.items():
            print(s, k, e and (e["pair_id"], e["question"], [(i["canonical"], i["prediction"]) for i in e["images"]],
                               "no image:", e["no_image_prediction"]))


def normalize_canon(r):
    from vqa_normalize import normalize
    return normalize(r["canonical"])


def figure(suite, name, picked):
    rows = [k for k in KINDS if picked.get(k)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 3.9 * len(rows)))
    for row, kind in zip(axes, rows):
        e = picked[kind]
        for ax, im in zip(row, e["images"]):
            ax.imshow(image(im["file"]))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            mark = "right" if im["right"] else "wrong"
            ax.set_xlabel(f"Reference: {im['canonical']}\nPrediction: {im['prediction']} ({mark})",
                          fontsize=9, color=INK, loc="left")
        head = textwrap.fill(f"{TITLES[kind]}.  Q: {e['question']}", 95)
        row[0].set_title(f"{head}\nNo-image prediction: {e['no_image_prediction']}   ({e['pair_id']})",
                         fontsize=9, color=INK, loc="left", pad=6)
    fig.suptitle(f"{name} test: complementary pairs, first pair of each kind in split order (rule in task2/examples.py)",
                 x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=4)
    (RES / "examples").mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / f"examples/complementary_{suite}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
