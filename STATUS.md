# What is done and what remains

Updated as work progresses. ✅ done · ⏳ in progress · ☐ not started · 🔒 blocked. "You" marks the student's own work (the report and annotations, under the AI-usage policy).

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
| Annotate the 42 responses (format, coherence, repetition, math framing, inappropriate content, truncation) | ☐ you |
| Task 1 section of the report | ☐ you |

## Task 2 — SigLIP–Qwen VLM

| Item | Status |
|---|---|
| Read the manual section; list open points (DECISIONS.md, Task 2) | ✅ |
| Confirm the open points with the user (all 15 settled 2026-10-30) | ✅ |
| Data script: caption selection, grounding pair pools + dev/test pairs, VQAv2 complementary-pair filtering, split isolation checks, saved IDs | ✅ `task2/prepare_data.py` → `task2/splits/` |
| Inspect grounding dev/test labels against the images, independently of model predictions (manual: "COCO annotations can omit objects or contain ambiguous counts") | ☐ (I'll render the pairs; you check them) |
| SigLIP feature caching (FP16, 196 × 768), with its time and memory logged separately | ☐ |
| Visual interface: connector, insertion into the chat, labels/masks/positions, checks (shapes, frozen sets, connector gradients, split isolation), scale calibration | ☐ |
| Training script: 3 stages (2,250 / 1,500 / 750 updates), sampler, replay, resumable checkpoints every 250 updates, per-stage weights | ☐ |
| Development evaluation at 0, every 500, and 2,250 / 3,750 / 4,500 (caption loss, answer NLL × 3 image inputs, ΔNLL) | ☐ |
| Smoke test, then the main run on Kaggle | ☐ |
| Final evaluation: 2 suites × 4 image conditions, generation, soft-VQA / exact-match scoring, pair metrics, NLL, timing, bootstrap CIs | ☐ |
| Figures: learning curves with stage boundaries; tables | ☐ |
| Qualitative complementary-pair examples with a stated selection rule | ☐ (selection rule and code by me, interpretation by you) |
| Optional 49-token run | skipped (your decision; stated in the report) |
| Task 2 README section | ☐ |
| Task 2 section of the report | ☐ you |

## Task 3 — Test-time compute

| Item | Status |
|---|---|
| ATML-PA3-Data (supplied rollouts, from LMS) | 🔒 needs you to download it from LMS |
| Read the manual section; list open points | ☐ |
| Everything else | ☐ |

## For the report: gaps and deviations to state

The manual says: "State any incomplete experiments or deviations from the prescribed setup." Everything the report must mention is collected here.

- **Every gap the manual left open, and the choice made:** DECISIONS.md, Task 1 rows #1–26 and Task 2 rows T2-1 to T2-15 (more will be added as work continues).
- **Task 1 caveats (EVIDENCE.md):**
  - GSM8K strict-match is 0% for every model: no response uses the "The answer is …" format. Flexible-extract is the primary metric.
  - Every R(z) math probe ends on the same chat-template tokens, so math vs non-math is partly template token vs content token.
  - L2-SP's θ₀ was kept in CPU memory because it did not fit on the T4 (#8).
- **Task 2:** the optional 49-token experiment was not run (T2-15).
- **Task 3:** to be added.

## Submission

| Item | Status |
|---|---|
| Public GitHub repository (create and push; I will ask before pushing) | ☐ |
| 8-page NeurIPS-format report with the repo link at the end of the abstract | ☐ you |
| Upload to LMS by 06 Nov 2026 | ☐ you |
