"""Task 3 Step 4: parallel sampling vs sequential refinement on the fixed 50-question MATH subset.

Manual settings: Qwen/Qwen2.5-3B-Instruct at the README revision, FP16, no quantization, batch size 1;
temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0; native chat template, system message
"You are a helpful assistant.", the problem followed by two newlines and the boxed-answer instruction.
Same 3,072-token total output allowance per question:

  single      one response, max 3,072 new tokens
  parallel    three independent responses, max 1,024 each; self-consistency picks the answer
  refinement  parallel response 1 as the initial answer, then one critique-and-revision response
              (max 2,048) to the conversation [system, user, assistant: initial, user: review]

Every call is independently seeded (DECISIONS.md T3-12): seed = 6304 * 1000 + 10 * k + slot, k the
question's position in the live list, slot 0 single, 1-3 parallel, 4 revision. Saved per call: prompt
text and token count, response text and token ids, seed, output length, whether it hit the cap,
end-to-end time (tokenization to decoded text, model loading excluded), extracted last boxed answer
(README balanced-brace rule). Grading and self-consistency are done locally (task3/live_score.py) with
the same scorer as the offline experiments.

Usage
  python task3/live.py --write-inputs      # local: task3/splits/math_live.json from the supplied data
  on Kaggle (GPU): python task3/live.py    # bundle task3/splits/math_live.json; writes task3/live/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task3" / "live.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]
sys.stdout.reconfigure(line_buffering=True)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # supplied README
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
CAPS = {"single": 3072, "parallel": 1024, "revision": 2048}
SEED = 6304
INPUTS = "task3/splits/math_live.json"


def extract_boxed(text):
    """README: the last literal `\\boxed{`, then balanced braces; the content is stripped. An absent,
    empty or unclosed last box gives None; no numeric fallback."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    while i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            content = text[start + len("\\boxed{"):i].strip()
            return content or None
        i += 1
    return None


def write_inputs():
    import pyarrow.parquet as pq
    live = json.loads((REPO / "task3/splits/math.json").read_text())["live"]
    g = pq.read_table(REPO / "data/task3/qwen25-math/generations.parquet",
                      columns=["sample_id", "rollout_id", "question_text", "gold_answer"]).to_pandas()
    g = g[g.rollout_id == 0].set_index("sample_id")
    rows = [{"sample_id": q, "question_text": g.loc[q, "question_text"], "gold_answer": g.loc[q, "gold_answer"]}
            for q in live]
    (REPO / INPUTS).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"wrote {len(rows)} questions to {INPUTS}")


def run(out_root):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    questions = json.loads((REPO / INPUTS).read_text())
    out = out_root / "task3" / "live"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16).cuda().eval()
    load_seconds = time.time() - t_load
    gen_cfg = model.generation_config
    stop_ids = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
    torch.cuda.reset_peak_memory_stats()

    def call(messages, max_new, seed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        set_seed(seed)
        with torch.no_grad():
            o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                               eos_token_id=stop_ids, pad_token_id=tok.pad_token_id, **SAMPLING)
        new = o[0, ids.shape[1]:].tolist()
        stopped = len(new) > 0 and new[-1] in stop_ids
        kept = new[:-1] if stopped else new  # the stop token is not part of the response
        text = tok.decode(kept, skip_special_tokens=True)
        torch.cuda.synchronize()
        return {"seed": seed, "max_new_tokens": max_new, "prompt": prompt, "input_tokens": int(ids.shape[1]),
                "response": text, "response_token_ids": kept, "output_tokens": len(new),
                "retained_tokens": len(kept), "hit_cap": not stopped, "seconds": time.perf_counter() - t0,
                "extracted": extract_boxed(text)}

    path = out / "calls.jsonl"
    done = set()
    if path.exists():  # resume after an interruption: skip finished (question, slot) calls
        done = {(r["sample_id"], r["slot"]) for r in map(json.loads, path.open())}
    with path.open("a") as f:
        for k, q in enumerate(questions):
            user = f"{q['question_text']}\n\n{INSTRUCTION}"
            base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            seed = lambda slot: SEED * 1000 + 10 * k + slot
            plan = [("single", 0, base, CAPS["single"])] + [("parallel", s, base, CAPS["parallel"]) for s in (1, 2, 3)]
            first = None
            for strategy, slot, msgs, cap in plan:
                if (q["sample_id"], slot) in done:
                    continue
                r = call(msgs, cap, seed(slot))
                if slot == 1:
                    first = r
                f.write(json.dumps({"sample_id": q["sample_id"], "index": k, "strategy": strategy, "slot": slot, **r}) + "\n")
                f.flush()
            if (q["sample_id"], 4) not in done:
                if first is None:  # resumed after slot 1: reload the shared initial response
                    first = next(r for r in map(json.loads, path.open())
                                 if r["sample_id"] == q["sample_id"] and r["slot"] == 1)
                msgs = base + [{"role": "assistant", "content": first["response"]}, {"role": "user", "content": REVIEW}]
                r = call(msgs, CAPS["revision"], seed(4))
                f.write(json.dumps({"sample_id": q["sample_id"], "index": k, "strategy": "refinement", "slot": 4, **r}) + "\n")
                f.flush()
            print(f"question {k + 1}/{len(questions)} done | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    import transformers
    (out / "run_info.json").write_text(json.dumps({
        "model": MODEL_ID, "revision": MODEL_REVISION, "dtype": "float16", "batch_size": 1, "sampling": SAMPLING,
        "caps": CAPS, "stop_ids": stop_ids, "gpu": torch.cuda.get_device_name(0), "model_load_seconds": load_seconds,
        "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 4),
        "torch": torch.__version__, "transformers": transformers.__version__}, indent=1))


KAGGLE_ARGS = []  # filled in by kaggle_push.py; Kaggle script kernels cannot take CLI arguments


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--write-inputs", action="store_true")
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else REPO)
    a = p.parse_args()
    write_inputs() if a.write_inputs else run(a.out_root)


if __name__ == "__main__":
    main()
