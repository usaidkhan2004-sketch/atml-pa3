# What is done and what remains

## Task 1 — LLM fine-tuning

| Item | Status |
|---|---|
| Data prep, probe set | ✅ |
| Five training runs | ✅ |
| Benchmarks: 7 conditions × 4 benchmarks | ✅ |
| Weight-space and activation diagnostics | ✅ |
| 168 qualitative responses; 42-response annotation subset collected | ✅ |
| Figures: SFT loss, benchmarks, diagnostics | ✅ |
| README reproduce section, requirements.txt | ✅ |
| Annotate the 42 responses (format, coherence, repetition, math framing, inappropriate content, truncation) | ✅ `task1/results/qualitative/annotations.csv` (scale: DECISIONS #27); counts by `task1/summarize_annotations.py` |
| Task 1 section of the report | ☐ you |

## Task 2 — SigLIP–Qwen VLM

| Item | Status |
|---|---|
| Read the manual section; list open points (DECISIONS.md, Task 2) | ✅ |
| Confirm the open points with the user (all 15 settled 2026-10-30) | ✅ |
| Data script: caption selection, grounding pair pools + dev/test pairs, VQAv2 complementary-pair filtering, split isolation checks, saved IDs | ✅ `task2/prepare_data.py` → `task2/splits/` |
| Inspect grounding dev/test labels against the images, independently of model predictions (manual: "COCO annotations can omit objects or contain ambiguous counts") | ⏳ sheets rendered in `task2/results/label_inspection/`; your check pending |
| SigLIP feature caching (FP16, 196 × 768), with its time and memory logged separately | ✅ Kaggle `atml-pa3-task2-features` |
| Visual interface: connector, insertion into the chat, labels/masks/positions, checks (shapes, frozen sets, connector gradients, split isolation), scale calibration | ✅ `task2/vlm.py`, Kaggle `atml-pa3-task2-interface-check` (s = 0.0507) |
| Training script: 3 stages (2,250 / 1,500 / 750 updates), sampler, replay, resumable checkpoints every 250 updates, per-stage weights | ✅ `task2/train.py` |
| Development evaluation at 0, every 500, and 2,250 / 3,750 / 4,500 (caption loss, answer NLL × 3 image inputs, ΔNLL) | ✅ built into `task2/train.py` |
| Smoke test, then the main run on Kaggle | ✅ smoke test passed (resume exact); main run `atml-pa3-task2-train` finished 2026-11-01: 4,500 updates, 1 h 38 min training on a T4; outputs in `task2/results/train/`, stage weights in `checkpoints/task2/weights/` (local) |
| Final evaluation: 2 suites × 4 image conditions, generation, soft-VQA / exact-match scoring, pair metrics, NLL, timing, bootstrap CIs | ⏳ `task2/evaluate.py` built and checked on the dev suites (Kaggle `atml-pa3-task2-eval-dev`); final test waits for your label inspection |
| Figures: learning curves with stage boundaries; tables | ⏳ learning curves done (`task2/make_plots.py`); final-test tables wait for the test run |
| Qualitative complementary-pair examples with a stated selection rule | ☐ (selection rule and code by me, interpretation by you) |
| Optional 49-token run | skipped (your decision; stated in the report) |
| Task 2 README section | ☐ |
| Task 2 section of the report | ☐ you |

## Task 3 — Test-time compute

| Item | Status |
|---|---|
| ATML-PA3-Data (supplied rollouts, Drive link in the manual) | ✅ downloaded to `data/task3/`; all README counts verified (EVIDENCE.md) |
| Read the manual section; list open points (DECISIONS.md T3-2 to T3-14; four confirmed 2026-11-07) | ✅ |
| Step 1: splits, orders, joins, token alignment, answer groups, correctness labels | ✅ `task3/prepare.py` |
| Steps 1–2: budgets 1–32 × 10 orders, FirstAns / SC / L / K / RWC / vision / VisConf, oracle coverage, failure decomposition, bootstrap | ✅ `task3/offline.py` |
| Step 3: length / score / correctness associations, quartiles, shortest-rollout control, confidence–vision correlation, tokenwise fusion, illustrative trajectories | ✅ `task3/signals.py` |
| Step 4: live single / parallel / refinement on 50 MATH questions | ✅ Kaggle `atml-pa3-task3-live` finished 2026-11-07 (250 calls); scored by `task3/live_score.py` |
| Step 5: adaptive stopping, dev-selected τ / γ, held-out results | ✅ in `task3/offline.py` |
| Step 6: figures (accuracy vs N and vs cost, oracle vs selection), selection-disagreement / revision / premature-stop examples with a stated rule | ✅ figures (`task3/make_plots.py`); selection-disagreement and premature-stop examples (`task3/examples.py`); revision examples (`task3/live_examples.py`) |
| Task 3 README section | ⏳ commands listed |
| Task 3 section of the report | ☐ you |

## For the report: gaps and deviations to state

The manual says: "State any incomplete experiments or deviations from the prescribed setup." Everything the report must mention is collected here.

- **Every gap the manual left open, and the choice made:** DECISIONS.md, Task 1 rows #1–26 and Task 2 rows T2-1 to T2-31 (more will be added as work continues).
- **Task 1 caveats (EVIDENCE.md):**
  - GSM8K strict-match is 0% for every model: no response uses the "The answer is …" format. Flexible-extract is the primary metric.
  - Every R(z) math probe ends on the same chat-template tokens, so math vs non-math is partly template token vs content token.
  - L2-SP's θ₀ was kept in CPU memory because it did not fit on the T4 (#8).
- **Task 2:** the optional 49-token experiment was not run (T2-15).
- **Every Task 3 gap and choice:** DECISIONS.md T3-2 to T3-14.
- **Task 3:** offline token cost is the manual's proxy (retained tokens only; no prefill, attention collection or scoring cost).

## Submission

| Item | Status |
|---|---|
| Public GitHub repository (create and push; I will ask before pushing) | ☐ |
| 8-page NeurIPS-format report with the repo link at the end of the abstract | ☐ you |
| Upload to LMS by 06 Nov 2026 | ☐ you |
