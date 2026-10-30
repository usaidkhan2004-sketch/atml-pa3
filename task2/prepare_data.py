"""Task 2 data preparation (manual: Task 2 "Data" and "Splits and sizes"). CPU only; no images are read.

Downloads the official annotation files to data/task2/raw/ (not committed) and selects, with one
seed-6304 numpy Generator used in the fixed order below:
  1. natural VQAv2 training pairs: 640 / 256 / 640 (yes/no, number, other) from train2014
  2. caption training images: 6,000 train2014 images x 3 of their captions
  3. evaluation sets from val2014, every image used at most once across all of them (T2-18):
     VQAv2 test 36/24/36 and dev 12/8/12; grounding test 32 presence + 32 counting and
     dev 12 + 12 (T2-5); caption dev 96 images x 1 caption
  4. grounding training pools (presence, counting) from the cached training images (T2-3, T2-16)
Answers are normalized with the official VQA rules (vqa_normalize.py) everywhere. Split isolation
is checked and recorded.

Outputs
  task2/splits/captions_{train,dev}.json, vqa_{train,dev,test}.json, grounding_{dev,test}.json,
  task2/splits/grounding_train_pools.json, task2/splits/images.json (images to cache, per split)
  task2/results/data_prep_summary.json
Usage: python task2/prepare_data.py
"""

import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "task2"))
import vqa_normalize as V

SEED = 6304
RAW = REPO / "data/task2/raw"
SPLITS = REPO / "task2/splits"
RESULTS = REPO / "task2/results"
SOURCES = {
    "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "v2_Questions_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "v2_Questions_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_Annotations_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "v2_Complementary_Pairs_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "v2_Complementary_Pairs_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}
N_CAPTION_TRAIN, CAPTIONS_PER_TRAIN_IMAGE, N_CAPTION_DEV = 6_000, 3, 96
VQA_COUNTS = {"train": {"yes/no": 640, "number": 256, "other": 640},
              "dev": {"yes/no": 12, "number": 8, "other": 12},
              "test": {"yes/no": 36, "number": 24, "other": 36}}
GROUNDING_COUNTS = {"dev": {"presence": 12, "counting": 12}, "test": {"presence": 32, "counting": 32}}
MIN_AGREEMENT = 8  # of 10 annotators, after normalization
COUNT_RANGE = range(1, 5)
OBJECTS = json.loads((REPO / "task2/configs/coco_objects.json").read_text())


def coco_file(split, image_id):
    return f"COCO_{split}2014_{image_id:012d}.jpg"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in SOURCES.items():
        path = RAW / name
        if not path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        out[name] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
    return out


def read_json(zip_name, member_suffix):
    with zipfile.ZipFile(RAW / zip_name) as z:
        member = next(m for m in z.namelist() if m.endswith(member_suffix))
        with z.open(member) as f:
            return json.load(f)


# --------------------------------------------------------------------------- VQAv2

def consensus(answers):
    """(normalized consensus, training target) or None. Consensus: a normalized answer given by
    >= 8 of 10 annotators. Target: its most frequent original string, ties by first occurrence."""
    norm = [V.normalize(a["answer"]) for a in answers]
    top, n = Counter(norm).most_common(1)[0]
    if n < MIN_AGREEMENT:
        return None
    originals = [a["answer"] for a, m in zip(answers, norm) if m == top]
    return top, Counter(originals).most_common(1)[0][0]


def eligible_vqa_pairs(split, stats):
    qs = {q["question_id"]: q for q in read_json(f"v2_Questions_{split.title()}_mscoco.zip",
                                                 f"{split}2014_questions.json")["questions"]}
    anns = {a["question_id"]: a for a in read_json(f"v2_Annotations_{split.title()}_mscoco.zip",
                                                   f"{split}2014_annotations.json")["annotations"]}
    pairs = read_json(f"v2_Complementary_Pairs_{split.title()}_mscoco.zip", "complementary_pairs.json")
    c = Counter()
    c["complementary_pairs"] = len(pairs)
    out = defaultdict(list)
    for q1, q2 in sorted(tuple(sorted(p)) for p in pairs):
        a, b = qs[q1], qs[q2]
        if a["question"] != b["question"] or a["image_id"] == b["image_id"]:
            c["not_same_question_different_images"] += 1
            continue
        ca, cb = consensus(anns[q1]["answers"]), consensus(anns[q2]["answers"])
        if ca is None or cb is None:
            c["below_agreement"] += 1
            continue
        if ca[0] == cb[0]:
            c["same_consensus"] += 1
            continue
        t = anns[q1]["answer_type"]
        if anns[q2]["answer_type"] != t:
            c["answer_type_differs"] += 1
            continue
        c[f"eligible_{t}"] += 1
        out[t].append({
            "pair_id": f"{q1}_{q2}", "answer_type": t, "question": a["question"],
            "items": [{"question_id": q, "image_id": qs[q]["image_id"], "file": coco_file(split, qs[q]["image_id"]),
                       "canonical": cn[0], "target": cn[1], "answers": [x["answer"] for x in anns[q]["answers"]]}
                      for q, cn in ((q1, ca), (q2, cb))],
        })
    stats[f"vqa_{split}"] = dict(c)
    return out


def take_unique(rng, candidates, n, used, images_of):
    """First n candidates in a seeded random order whose images are all unused; marks them used."""
    taken = []
    for i in rng.permutation(len(candidates)):
        imgs = images_of(candidates[i])
        if len(set(imgs)) == len(imgs) and not used.intersection(imgs):
            taken.append(candidates[i])
            used.update(imgs)
            if len(taken) == n:
                return taken
    raise RuntimeError(f"only {len(taken)} of {n} candidates available")


# --------------------------------------------------------------------------- grounding

def instance_index(split):
    """{image_id: {category name: (non-crowd count, has crowd)}}, all images of the split, and
    the category id -> name map."""
    inst = read_json("annotations_trainval2014.zip", f"instances_{split}2014.json")
    names = {c["id"]: c["name"] for c in inst["categories"]}
    assert set(names.values()) == set(k for k in OBJECTS if not k.startswith("_"))
    idx = {im["id"]: {} for im in inst["images"]}
    for a in inst["annotations"]:
        n, crowd = idx[a["image_id"]].get(names[a["category_id"]], (0, False))
        idx[a["image_id"]][names[a["category_id"]]] = (n + (not a["iscrowd"]), crowd or bool(a["iscrowd"]))
    return idx, sorted(names.values())


def presence_question(cat):
    return f"Does this image contain {OBJECTS[cat]['presence']}?"


def counting_question(cat):
    return f"How many {OBJECTS[cat]['plural']} can you see?"


def is_positive(idx, img, cat):  # >= 1 non-crowd instance and no crowd region (T2-4)
    n, crowd = idx[img].get(cat, (0, False))
    return n >= 1 and not crowd


def is_negative(idx, img, cat):  # no annotation of the category at all
    return cat not in idx[img]


def count_of(idx, img, cat):  # count in 1..4 with no crowd region, else None
    n, crowd = idx[img].get(cat, (0, False))
    return n if n in COUNT_RANGE and not crowd else None


def grounding_pair(kind, cat, a, b, idx):
    if kind == "presence":
        return {"type": kind, "category": cat, "question": presence_question(cat),
                "images": [a, b], "answers": ["yes", "no"]}
    return {"type": kind, "category": cat, "question": counting_question(cat),
            "images": [a, b], "answers": [str(count_of(idx, a, cat)), str(count_of(idx, b, cat))]}


def training_pools(rng, idx, cats, images):
    """T2-16: one presence pair per (image, category) occurrence, partner = a random image of the
    set without that category; one counting pair per (image, category) with count 1-4, partner =
    a random image of the set with a different count of 1-4."""
    images = sorted(images)
    pools = {"presence": [], "counting": []}
    for cat in cats:
        neg = [i for i in images if is_negative(idx, i, cat)]
        by_count = defaultdict(list)
        for i in images:
            if count_of(idx, i, cat):
                by_count[count_of(idx, i, cat)].append(i)
        others_by_k = {k: [i for c, lst in sorted(by_count.items()) if c != k for i in lst] for k in COUNT_RANGE}
        for img in images:
            if is_positive(idx, img, cat) and neg:
                pools["presence"].append(grounding_pair("presence", cat, img, neg[rng.integers(len(neg))], idx))
            k = count_of(idx, img, cat)
            if k:
                others = others_by_k[k]
                if others:
                    pools["counting"].append(grounding_pair("counting", cat, img, others[rng.integers(len(others))], idx))
    return pools


def eval_grounding(rng, idx, cats, n, kind, used, split_name):
    """n pairs whose two images are unused val2014 images, marking both as used."""
    occ = [(img, cat) for img in sorted(idx) for cat in cats
           if (is_positive(idx, img, cat) if kind == "presence" else count_of(idx, img, cat))]
    pool = sorted(idx)
    out = []
    for j in rng.permutation(len(occ)):
        img, cat = occ[j]
        if img in used:
            continue
        if kind == "presence":
            ok = lambda i: is_negative(idx, i, cat)
        else:
            ok = lambda i: count_of(idx, i, cat) and count_of(idx, i, cat) != count_of(idx, img, cat)
        partner = next((pool[i] for i in rng.permutation(len(pool))
                        if pool[i] not in used and pool[i] != img and ok(pool[i])), None)
        if partner is None:
            continue
        p = grounding_pair(kind, cat, img, partner, idx)
        used.update(p["images"])
        out.append({"pair_id": f"{split_name}_{kind}_{len(out):03d}", **p})
        if len(out) == n:
            return out
    raise RuntimeError(f"only {len(out)} {kind} pairs")


def with_files(pair, split):
    """Grounding pair -> the same item layout as VQA pairs."""
    return {"pair_id": pair["pair_id"], "type": pair["type"], "category": pair["category"],
            "question": pair["question"],
            "items": [{"image_id": i, "file": coco_file(split, i), "answer": a}
                      for i, a in zip(pair["images"], pair["answers"])]}


# --------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    stats = {"seed": SEED, "sources": fetch()}
    SPLITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. Natural VQAv2 training pairs (train2014; images may recur across training pairs).
    elig_train = eligible_vqa_pairs("train", stats)
    vqa_train = [p for t, n in VQA_COUNTS["train"].items()
                 for p in (elig_train[t][i] for i in rng.choice(len(elig_train[t]), n, replace=False))]
    del elig_train

    # 2. Caption training images and captions.
    caps_train = read_json("annotations_trainval2014.zip", "captions_train2014.json")["annotations"]
    by_img = defaultdict(list)
    for c in sorted(caps_train, key=lambda c: c["id"]):
        by_img[c["image_id"]].append(c)
    del caps_train
    imgs = sorted(i for i, cs in by_img.items() if len(cs) >= CAPTIONS_PER_TRAIN_IMAGE)
    captions_train = []
    for i in rng.choice(len(imgs), N_CAPTION_TRAIN, replace=False):
        cs = by_img[imgs[i]]
        pick = sorted(rng.choice(len(cs), CAPTIONS_PER_TRAIN_IMAGE, replace=False))
        captions_train.append({"image_id": imgs[i], "file": coco_file("train", imgs[i]),
                               "caption_ids": [cs[k]["id"] for k in pick],
                               "captions": [cs[k]["caption"].strip() for k in pick]})
    del by_img

    # 3. Evaluation sets (val2014), each image used at most once overall.
    used = set()
    elig_val = eligible_vqa_pairs("val", stats)
    vqa_eval = {}
    for split in ["test", "dev"]:
        vqa_eval[split] = [p for t, n in VQA_COUNTS[split].items()
                           for p in take_unique(rng, elig_val[t], n, used,
                                                lambda p: [x["image_id"] for x in p["items"]])]
    del elig_val
    idx_val, cats = instance_index("val")
    grounding_eval = {}
    for split in ["test", "dev"]:
        grounding_eval[split] = [with_files(p, "val") for kind, n in GROUNDING_COUNTS[split].items()
                                 for p in eval_grounding(rng, idx_val, cats, n, kind, used, split)]
    del idx_val
    caps_val = read_json("annotations_trainval2014.zip", "captions_val2014.json")["annotations"]
    by_img = defaultdict(list)
    for c in sorted(caps_val, key=lambda c: c["id"]):
        by_img[c["image_id"]].append(c)
    cand = sorted(by_img)
    captions_dev = []
    for i in rng.permutation(len(cand)):
        if cand[i] in used:
            continue
        cs = by_img[cand[i]]
        k = int(rng.integers(len(cs)))
        captions_dev.append({"image_id": cand[i], "file": coco_file("val", cand[i]),
                             "caption_ids": [cs[k]["id"]], "captions": [cs[k]["caption"].strip()]})
        used.add(cand[i])
        if len(captions_dev) == N_CAPTION_DEV:
            break
    del by_img, caps_val

    # 4. Grounding training pools from the cached training images (T2-3).
    train_images = sorted({c["image_id"] for c in captions_train}
                          | {x["image_id"] for p in vqa_train for x in p["items"]})
    idx_train, cats_train = instance_index("train")
    assert cats_train == cats
    pools = training_pools(rng, idx_train, cats, train_images)
    no_instances = sum(not idx_train[i] for i in train_images)
    del idx_train

    # Split isolation checks.
    eval_sets = {"vqa_dev": [x["image_id"] for p in vqa_eval["dev"] for x in p["items"]],
                 "vqa_test": [x["image_id"] for p in vqa_eval["test"] for x in p["items"]],
                 "grounding_dev": [x["image_id"] for p in grounding_eval["dev"] for x in p["items"]],
                 "grounding_test": [x["image_id"] for p in grounding_eval["test"] for x in p["items"]],
                 "captions_dev": [c["image_id"] for c in captions_dev]}
    all_eval = [i for v in eval_sets.values() for i in v]
    pool_images = {i for kind in pools.values() for p in kind for i in p["images"]}
    checks = {
        "eval_images_unique_across_all_sets": len(all_eval) == len(set(all_eval)),
        "eval_sets_pairwise_disjoint": all(not set(eval_sets[a]) & set(eval_sets[b])
                                           for a in eval_sets for b in eval_sets if a < b),
        "train_images_disjoint_from_eval": not set(train_images) & set(all_eval),
        "pool_images_within_cached_training_images": pool_images <= set(train_images),
        "all_train_files_train2014": all(c["file"].startswith("COCO_train2014_") for c in captions_train)
                                     and all(x["file"].startswith("COCO_train2014_") for p in vqa_train for x in p["items"]),
        "all_eval_files_val2014": all(x["file"].startswith("COCO_val2014_")
                                      for s in (vqa_eval["dev"], vqa_eval["test"], grounding_eval["dev"], grounding_eval["test"])
                                      for p in s for x in p["items"])
                                  and all(c["file"].startswith("COCO_val2014_") for c in captions_dev),
        "vqa_pairs_differ_in_consensus": all(p["items"][0]["canonical"] != p["items"][1]["canonical"]
                                             for s in (vqa_train, vqa_eval["dev"], vqa_eval["test"]) for p in s),
        "grounding_pairs_differ_in_answer": all(p["items"][0]["answer"] != p["items"][1]["answer"]
                                                for s in grounding_eval.values() for p in s)
                                            and all(p["answers"][0] != p["answers"][1] for k in pools.values() for p in k),
    }
    assert all(checks.values()), checks

    def dump(name, obj):
        (SPLITS / name).write_text(json.dumps(obj, indent=1) + "\n")

    dump("captions_train.json", captions_train)
    dump("captions_dev.json", captions_dev)
    for split, pairs in [("train", vqa_train), ("dev", vqa_eval["dev"]), ("test", vqa_eval["test"])]:
        dump(f"vqa_{split}.json", pairs)
    for split in ["dev", "test"]:
        dump(f"grounding_{split}.json", grounding_eval[split])
    dump("grounding_train_pools.json", pools)
    val_images = sorted(set(all_eval))
    dump("images.json", {"train": [{"image_id": i, "file": coco_file("train", i)} for i in train_images],
                         "val": [{"image_id": i, "file": coco_file("val", i)} for i in val_images]})

    stats.update(
        counts={
            "captions_train": {"images": len(captions_train), "examples": sum(len(c["captions"]) for c in captions_train)},
            "captions_dev": {"images": len(captions_dev), "examples": len(captions_dev)},
            **{f"vqa_{s}": {"pairs": len(p), "questions": 2 * len(p),
                            "by_type": dict(Counter(x["answer_type"] for x in p))}
               for s, p in [("train", vqa_train), ("dev", vqa_eval["dev"]), ("test", vqa_eval["test"])]},
            **{f"grounding_{s}": {"pairs": len(p), "questions": 2 * len(p), "by_type": dict(Counter(x["type"] for x in p))}
               for s, p in grounding_eval.items()},
            "grounding_train_pools": {k: len(v) for k, v in pools.items()},
            "images_to_cache": {"train": len(train_images), "val": len(val_images)},
        },
        training_image_overlap={
            "caption_and_vqa_train_images": len({c["image_id"] for c in captions_train}
                                                & {x["image_id"] for p in vqa_train for x in p["items"]}),
            "cached_training_images_without_any_instance": no_instances,
        },
        answer_distribution={
            "grounding_pool_counting_answers": dict(sorted(Counter(a for p in pools["counting"] for a in p["answers"]).items())),
            "grounding_pool_categories_top10": Counter(p["category"] for p in pools["presence"]).most_common(10),
            "vqa_train_number_canonical_top10": Counter(x["canonical"] for p in vqa_train if p["answer_type"] == "number"
                                                        for x in p["items"]).most_common(10),
        },
        isolation_checks=checks,
        seconds=time.time() - t0,
    )
    (RESULTS / "data_prep_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats["counts"], indent=1))
    print("checks:", checks)


if __name__ == "__main__":
    main()
