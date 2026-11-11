"""Task 2 final evaluation (manual Step 5): both question suites under four image conditions.

Manual: "Evaluate both final-test suites using greedy decoding, thinking disabled, a 16-new-token cap,
and batch size 8. Keep the question and original references fixed across four conditions."

  correct        the question's own image
  complementary  the pair's other image (different answer)
  mismatched     an image from another pair of the same suite, from a mapping fixed before generation
                 and reused for every comparison (DECISIONS.md T2-12)
  no_image       visual tokens omitted, text tokens unchanged; masks and positions rebuilt

Per suite and condition: accuracy (grounding: normalized exact match, presence / counting; VQAv2: local
soft consensus accuracy against all ten answers, yes/no / number / other), pair accuracy (both
predictions equal their original canonical answers), identical-answer rate (the pair's two
predictions are equal), empty-answer rate (empty outputs stay in the denominators), mean
reference-answer NLL (termination excluded, as in development), and generation time per answer.
Correct-image advantages over each control in accuracy and NLL. 95% intervals from 4,000 bootstrap
resamples of complete pairs, the same resampled pair ids for every condition (T2-14).

Usage
  python task2/evaluate.py --split dev     # script check on the development suites
  python task2/evaluate.py --split test    # final test (run only after the label inspection)
  on Kaggle: features kernel, training kernel (stage 3 weights) and the splits dataset mounted.
Outputs (under --out-root): task2/eval/<split>/{metrics.json, predictions.jsonl, mismatched_mapping.json}
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task2" / "evaluate.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]
sys.path.insert(0, str(SELF.parent))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import vlm as V
from check_interface import load
from vqa_normalize import normalize, soft_accuracy

SEED = V.SEED
BATCH = 8
MAX_NEW_TOKENS = 16
STOP_IDS = (V.IM_END_ID, V.PAD_ID)  # <|im_end|> ends the turn; <|endoftext|> is Qwen3's other stop id
CONDITIONS = ["correct", "complementary", "mismatched", "no_image"]
CONTROLS = ["complementary", "mismatched", "no_image"]
N_BOOT = 4000
SUITES = {"vqa": ("vqa_{}", "answer_type"), "grounding": ("grounding_{}", "type")}
TYPES = {"vqa": ["yes/no", "number", "other"], "grounding": ["presence", "counting"]}


# --------------------------------------------------------------------------- data

def suite_items(fmt, suite, split):
    """One record per question, in pair order: (pair, item 0), (pair, item 1), ..."""
    name, qtype = SUITES[suite]
    out = []
    for p in load(name.format(split)):
        a, b = p["items"]
        for it, other in ((a, b), (b, a)):
            if suite == "vqa":
                canonical, partner_canonical, ref, refs = it["canonical"], other["canonical"], it["target"], it["answers"]
            else:
                canonical, partner_canonical, ref, refs = it["answer"], other["answer"], it["answer"], [it["answer"]]
            out.append({"pair_id": p["pair_id"], "type": p[qtype], "question": p["question"],
                        "image_id": it["image_id"], "partner_image_id": other["image_id"],
                        "canonical": canonical, "partner_canonical": partner_canonical,
                        "reference": ref, "answers": refs,
                        "example": fmt.qa_example(it["image_id"], p["question"], ref)})
    return out


def mismatched_mapping(items):
    """T2-12: a seeded permutation of the suite's question images in which no question receives an
    image of its own pair, so every image is used exactly once. Drawn by rejection from
    numpy default_rng(6304), separately for each suite, before any generation."""
    rng = np.random.default_rng(SEED)
    images = [it["image_id"] for it in items]
    pair_of = {it["image_id"]: it["pair_id"] for it in items}
    tries = 0
    while True:
        tries += 1
        perm = [images[k] for k in rng.permutation(len(images))]
        if all(pair_of[m] != it["pair_id"] for m, it in zip(perm, items)):
            return {str(it["image_id"]): m for it, m in zip(items, perm)}, tries


def condition_inputs(items, condition, mapping):
    """(examples, with_image) for one condition. The question and reference never change."""
    if condition == "no_image":
        return [it["example"] for it in items], [False] * len(items)
    img = {"correct": lambda it: it["image_id"], "complementary": lambda it: it["partner_image_id"],
           "mismatched": lambda it: mapping[str(it["image_id"])]}[condition]
    return [{**it["example"], "image_id": img(it)} for it in items], [True] * len(items)


# --------------------------------------------------------------------------- model

def find_weights(stage=3):
    local = REPO / "checkpoints/task2/weights" / f"stage{stage}"
    if (local / "connector.safetensors").exists():
        return local
    hits = glob.glob(f"/kaggle/input/**/task2/run/weights/stage{stage}/connector.safetensors", recursive=True)
    assert hits, "stage weights not found"
    return Path(sorted(hits)[0]).parent


def load_model(weights):
    """Frozen FP16 Qwen3 with LoRA on q/k/v/o and the FP32 connector, both from the completed Stage 3
    (manual: "The completed Stage 3 model is the final model"). Dropout off."""
    model = AutoModelForCausalLM.from_pretrained(V.MODEL_ID, revision=V.MODEL_REVISION, dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda()
    V.freeze_backbone(model)
    V.add_lora(model)
    lora = load_file(str(weights / "lora.safetensors"))
    used = 0
    for n, m in model.named_modules():
        if isinstance(m, V.LoRALinear):
            for k in ("lora_A", "lora_B"):
                getattr(m, k).data.copy_(lora[f"{n}.{k}"])
                used += 1
    assert used == len(lora), (used, len(lora))
    connector = V.Connector()
    connector.load_state_dict(load_file(str(weights / "connector.safetensors")))
    connector = connector.cuda().eval()
    model.eval()
    return model, connector


@torch.no_grad()
def generate(model, connector, store, examples, with_image):
    """Greedy decoding with a KV cache from a left-padded batch: up to 16 new tokens, stopping at
    <|im_end|> or <|endoftext|> (the stop token is not kept). Returns token lists and seconds."""
    # an empty target leaves exactly the prompt, ending with the assistant header and empty think block
    prompts = [{**ex, "target": []} for ex in examples]
    b = V.to_device(V.build_batch(prompts, with_image, pad_side="left", supervise_end=False), "cuda")
    am, pos = b["attention_mask"], b["position_ids"]
    assert bool((am[:, -1] == 1).all())  # left padding: every row's prompt ends at the last position
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        emb = V.embed(model, connector, b, store.get(b["image_ids"]))
        out = model(inputs_embeds=emb, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        done = torch.zeros(len(examples), dtype=torch.bool, device="cuda")
        tokens = [[] for _ in examples]
        for step in range(MAX_NEW_TOKENS):
            nxt = out.logits[:, -1].float().argmax(-1)
            for k in range(len(examples)):
                if not done[k]:
                    t = int(nxt[k])
                    if t in STOP_IDS:
                        done[k] = True
                    else:
                        tokens[k].append(t)
            if bool(done.all()) or step == MAX_NEW_TOKENS - 1:
                break
            am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
            pos = pos[:, -1:] + 1
            out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=pos, past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = out.past_key_values
    torch.cuda.synchronize()
    return tokens, [bool(d) for d in done.tolist()], time.perf_counter() - t0


@torch.no_grad()
def check_generation(model, connector, store, examples, with_image):
    """Decoding checks on one batch: (1) the KV-cache loop gives the same tokens as recomputing the
    whole sequence at every step without a cache; (2) each example decoded alone (no padding) gives
    the same tokens as inside the left-padded batch. FP16, so (2) is recorded rather than asserted."""
    toks, _, _ = generate(model, connector, store, examples, with_image)
    no_cache = []
    for ex, img in zip(examples, with_image):
        gen = []
        for _ in range(MAX_NEW_TOKENS):
            b = V.to_device(V.build_batch([{**ex, "target": gen}], [img], supervise_end=False), "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                h = V.hidden_states(model, connector, b, store.get(b["image_ids"]))[:, -1]
                t = int(model.lm_head(h).float().argmax(-1))
            if t in STOP_IDS:
                break
            gen.append(t)
        no_cache.append(gen)
    alone = [generate(model, connector, store, [ex], [img])[0][0] for ex, img in zip(examples, with_image)]
    return {"examples": len(examples), "cache_equals_no_cache": sum(a == b for a, b in zip(toks, no_cache)),
            "batched_equals_alone": sum(a == b for a, b in zip(toks, alone))}


@torch.no_grad()
def reference_nll(model, connector, store, examples, with_image):
    """Mean token NLL of the reference answer, termination excluded (as in development)."""
    out = []
    for s in range(0, len(examples), BATCH):
        b = V.to_device(V.build_batch(examples[s:s + BATCH], with_image[s:s + BATCH], supervise_end=False), "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(V.example_nll(model, connector, b, store.get(b["image_ids"])).float().cpu())
    return torch.cat(out).tolist()


# --------------------------------------------------------------------------- scoring

def score(suite, prediction, item):
    """Grounding: normalized exact match. VQAv2: local soft consensus accuracy over all ten answers."""
    if suite == "vqa":
        return soft_accuracy(prediction, item["answers"])
    return float(normalize(prediction) == normalize(item["canonical"]))


def pair_table(suite, items, rows):
    """Per-pair arrays for one condition: accuracy sums, pair correctness, identical, empty, NLL, time."""
    by_pair = {}
    for it, r in zip(items, rows):
        by_pair.setdefault(it["pair_id"], []).append((it, r))
    pairs = list(by_pair)
    t = {"pair_id": pairs, "type": [by_pair[p][0][0]["type"] for p in pairs]}
    t["acc"] = np.array([[r["score"] for _, r in by_pair[p]] for p in pairs])
    t["exact"] = np.array([[float(r["norm"] == normalize(it["canonical"])) for it, r in by_pair[p]] for p in pairs])
    t["identical"] = np.array([float(by_pair[p][0][1]["norm"] == by_pair[p][1][1]["norm"]) for p in pairs])
    t["empty"] = np.array([[float(r["norm"] == "") for _, r in by_pair[p]] for p in pairs])
    t["nll"] = np.array([[r["nll"] for _, r in by_pair[p]] for p in pairs])
    return t


def stats(t, idx):
    """Metrics on the pairs `idx` (indices into the pair table, repeats allowed)."""
    return {"accuracy": float(t["acc"][idx].mean()), "pair_accuracy": float(t["exact"][idx].min(1).mean()),
            "identical_answer_rate": float(t["identical"][idx].mean()), "empty_answer_rate": float(t["empty"][idx].mean()),
            "nll": float(t["nll"][idx].mean())}


def advantages(tc, tx, idx):
    """Correct-image advantage over a control: accuracy(correct) - accuracy(control) and
    NLL(control) - NLL(correct), both positive when the correct image helps."""
    return {"accuracy": float(tc["acc"][idx].mean() - tx["acc"][idx].mean()),
            "pair_accuracy": float(tc["exact"][idx].min(1).mean() - tx["exact"][idx].min(1).mean()),
            "nll": float(tx["nll"][idx].mean() - tc["nll"][idx].mean())}


def with_ci(point, boots):
    keys = point.keys()
    boots = [b for b in boots if b is not None]
    return {k: {"value": point[k], "ci95": [float(np.percentile([b[k] for b in boots], 2.5)),
                                            float(np.percentile([b[k] for b in boots], 97.5))]} for k in keys}


def suite_metrics(suite, items, results, timing):
    """Point estimates and pair-bootstrap intervals, overall and by question type. Every bootstrap
    replicate draws one set of pair indices and applies it to all conditions (T2-14)."""
    tables = {c: pair_table(suite, items, results[c]) for c in CONDITIONS}
    n_pairs = len(tables["correct"]["pair_id"])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_pairs, size=(N_BOOT, n_pairs))
    types = np.array(tables["correct"]["type"])
    out = {"n_pairs": n_pairs, "n_questions": 2 * n_pairs, "conditions": {}, "correct_advantage": {}}

    def subset(name):
        return np.arange(n_pairs) if name == "all" else np.flatnonzero(types == name)

    for group in ["all"] + TYPES[suite]:
        sel = subset(group)
        mask = np.isin(np.arange(n_pairs), sel)
        boot_idx = [d[mask[d]] for d in draws]  # the resampled pairs that belong to this group
        # a replicate with no pair of a small group is dropped for that group only
        for c in CONDITIONS:
            rec = with_ci(stats(tables[c], sel), [stats(tables[c], i) if len(i) else None for i in boot_idx])
            rec["n_pairs"] = int(len(sel))
            out["conditions"].setdefault(c, {})[group] = rec
        for c in CONTROLS:
            out["correct_advantage"].setdefault(c, {})[group] = with_ci(
                advantages(tables["correct"], tables[c], sel),
                [advantages(tables["correct"], tables[c], i) if len(i) else None for i in boot_idx])
    for c in CONDITIONS:
        out["conditions"][c]["generation"] = timing[c]
    return out, tables


def swap_diagnostics(items, results):
    """Under the complementary swap, does the answer follow the partner image? Each swapped
    prediction is classed as the question's own canonical answer, the partner's canonical answer,
    or neither (empty separately). Also checks that a swapped input reproduces the partner's
    correct-image prediction (the two inputs are token-identical; the question is shared)."""
    pos = {(it["pair_id"], it["image_id"]): k for k, it in enumerate(items)}
    counts = {"own_answer": 0, "partner_answer": 0, "other": 0, "empty": 0}
    same_as_partner = 0
    for k, it in enumerate(items):
        p = results["complementary"][k]["norm"]
        if p == "":
            counts["empty"] += 1
        elif p == normalize(it["canonical"]):
            counts["own_answer"] += 1
        elif p == normalize(it["partner_canonical"]):
            counts["partner_answer"] += 1
        else:
            counts["other"] += 1
        same_as_partner += p == results["correct"][pos[(it["pair_id"], it["partner_image_id"])]]["norm"]
    n = len(items)
    return {"n_questions": n, **{f"{k}_rate": v / n for k, v in counts.items()}, "counts": counts,
            "swap_equals_partner_correct_prediction": same_as_partner / n}


# --------------------------------------------------------------------------- run

def evaluate(args):
    t_start = time.time()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(V.MODEL_ID, revision=V.MODEL_REVISION)
    fmt = V.Formatter(tok)
    items = {s: suite_items(fmt, s, args.split) for s in SUITES}
    if args.limit:  # whole pairs only
        items = {s: v[:2 * args.limit] for s, v in items.items()}
    mappings, tries = {}, {}
    for s in SUITES:
        mappings[s], tries[s] = mismatched_mapping(items[s])
    out = args.out_root / "task2" / "eval" / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "mismatched_mapping.json").write_text(json.dumps({"seed": SEED, "rejection_draws": tries,
                                                             "mapping": mappings}, indent=1))

    store = V.FeatureStore(V.find_feature_dir(), device="cuda")
    weights = find_weights()
    model, connector = load_model(weights)
    recorded = json.loads((REPO / "task2/results/interface_checks.json").read_text())["calibration"]["scale"]
    assert abs(float(connector.scale) - recorded) / recorded < 1e-5, (float(connector.scale), recorded)

    metrics = {"split": args.split, "weights": str(weights), "conditions": CONDITIONS, "suites": {}}
    for s in SUITES:  # decoding checks on the first batch of each suite, correct and no-image inputs
        for c in ("correct", "no_image"):
            exs, with_img = condition_inputs(items[s][:BATCH], c, mappings[s])
            metrics.setdefault("generation_checks", {})[f"{s}/{c}"] = check_generation(model, connector, store,
                                                                                      exs, with_img)
    print("generation checks:", metrics["generation_checks"])
    torch.cuda.reset_peak_memory_stats()
    t_eval = time.time()
    with open(out / "predictions.jsonl", "w") as f:
        for s in SUITES:
            results, timing = {}, {}
            for c in CONDITIONS:
                exs, with_img = condition_inputs(items[s], c, mappings[s])
                nll = reference_nll(model, connector, store, exs, with_img)
                rows, seconds, n_tok = [], 0.0, 0
                for i in range(0, len(exs), BATCH):
                    toks, stopped, sec = generate(model, connector, store, exs[i:i + BATCH], with_img[i:i + BATCH])
                    seconds += sec
                    for k, (tk, st) in enumerate(zip(toks, stopped)):
                        text = tok.decode(tk, skip_special_tokens=False).strip()
                        it = items[s][i + k]
                        rows.append({"generated": text, "norm": normalize(text), "new_tokens": len(tk),
                                     "stopped": st, "score": score(s, text, it), "nll": nll[i + k]})
                        n_tok += len(tk) + int(st)
                results[c] = rows
                timing[c] = {"seconds": seconds, "answers": len(rows), "seconds_per_answer": seconds / len(rows),
                             "new_tokens": n_tok, "hit_cap": sum(not r["stopped"] for r in rows)}
                print(f"{s} {c}: acc {np.mean([r['score'] for r in rows]):.4f} | nll {np.mean(nll):.3f} | "
                      f"{seconds / len(rows) * 1000:.1f} ms/answer")
                for it, r in zip(items[s], rows):
                    f.write(json.dumps({"suite": s, "condition": c, "pair_id": it["pair_id"], "type": it["type"],
                                        "question": it["question"], "image_id": it["image_id"],
                                        "input_image_id": (None if c == "no_image" else
                                                           {"correct": it["image_id"],
                                                            "complementary": it["partner_image_id"],
                                                            "mismatched": mappings[s][str(it["image_id"])]}[c]),
                                        "canonical": it["canonical"], "reference": it["reference"],
                                        "answers": it["answers"], **r}) + "\n")
            m, _ = suite_metrics(s, items[s], results, timing)
            m["complementary_swap"] = swap_diagnostics(items[s], results)
            metrics["suites"][s] = m

    metrics["resources"] = {"gpu": torch.cuda.get_device_name(0), "eval_seconds": time.time() - t_eval,
                            "total_seconds": time.time() - t_start,
                            "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 4),
                            "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 4),
                            "note": "peak memory includes the cached SigLIP features held on the GPU"}
    metrics["settings"] = {"decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "batch": BATCH,
                           "thinking": "disabled (empty think block from the chat template)",
                           "stop_ids": list(STOP_IDS), "bootstrap_resamples": N_BOOT, "seed": SEED,
                           "limit_pairs": args.limit, "split_hashes": V.split_hashes(V.find_splits_dir(REPO))}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({s: {c: round(m["conditions"][c]["all"]["accuracy"]["value"], 4) for c in CONDITIONS}
                      for s, m in metrics["suites"].items()}))


KAGGLE_ARGS = []  # filled in by kaggle_push.py; Kaggle script kernels cannot take CLI arguments


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--limit", type=int, help="first N pairs per suite (script check only)")
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else REPO)
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
