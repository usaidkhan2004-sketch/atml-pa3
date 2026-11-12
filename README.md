# ATML PA3 — Fine-Tuning, Vision-Language Models, and Test-Time Compute

EE-5102 / CS-6304, Fall 2026. Seed `6304` is used throughout.

## Layout

```
shared/   kaggle_push.py: pushes a repo script to Kaggle as a private script kernel
task1/    LLM fine-tuning: Full FT, Full FT + L2-SP, LoRA-4/16, OPLoRA-16 on MetaMathQA (Qwen3-0.6B)
task2/    SigLIP-Qwen VLM: caption alignment -> visual instruction tuning -> natural VQA, grounding eval
task3/    Test-time compute: Best-of-N, self-consistency, self-certainty, VisConf/RWC, live refinement, adaptive sampling
data/     raw datasets and supplied rollouts (not committed)
```

Each task folder has `configs/`, `splits/` (saved example/pair/question IDs), `results/` (machine-readable outputs and figures), and its source scripts.

## Environment

```bash
pip install -r requirements.txt
```

`DECISIONS.md` records every point the manual leaves open and the choice made. `EVIDENCE.md` maps each item of the manual's "Required Evidence" to the result file that backs it.

## Reproducing

### Task 1 — LLM fine-tuning (Qwen3-0.6B on MetaMathQA)

All GPU steps ran on Kaggle (2× Tesla T4). Each script runs unchanged locally if a CUDA GPU is available. On Kaggle it is pushed with `shared/kaggle_push.py`, which:
- fills in the script's `KAGGLE_ARGS`;
- bundles the repo files named with `--with`;
- mounts earlier kernels' outputs with `--source`.

Model weights and the tokenized training data stay in those kernel outputs and are not committed.

| Step | Script | Kaggle command (`python shared/kaggle_push.py …`) | Committed output |
|---|---|---|---|
| 1. Data prep (CPU) | `task1/prepare_data.py` | `task1/prepare_data.py atml-pa3-task1-prepare-data` | `task1/splits/`, `task1/results/data_prep_summary.json` |
| 2. Training, one run per method | `task1/train.py` | `task1/train.py atml-pa3-task1-<m> --gpu --source dentusayd/atml-pa3-task1-prepare-data -- --method <m>`, for `<m>` in `fullft l2sp lora4 lora16 oplora` | `task1/results/<m>/` |
| 3. Benchmarks (7 conditions) | `task1/evaluate.py` | `task1/evaluate.py atml-pa3-task1-eval-a --gpu --source …-fullft --source …-l2sp <W> -- --conditions baseline fullft_0625 fullft_1250 l2sp_1250`, then `…-eval-b` with `--source …-lora4 --source …-lora16 --source …-oplora <W> -- --conditions lora4_1250 lora16_1250 oplora_1250` | `task1/results/eval/<condition>/` |
| 4. Diagnostics | `task1/diagnostics.py` | `task1/diagnostics.py atml-pa3-task1-diagnostics --gpu` + all five training sources + `<W> --with task1/splits/probe_prompts.json` | `task1/results/diagnostics/` |
| 5. Qualitative generations | `task1/generate_qualitative.py` | `task1/generate_qualitative.py atml-pa3-task1-qualitative --gpu` + all five training sources + `--with task1/train.py --with task1/evaluate.py --with task1/configs/first_plot_questions.yaml` | `task1/results/qualitative/` |
| 5b. Annotation counts | `task1/summarize_annotations.py` | — (local: `python task1/summarize_annotations.py`) | `task1/results/qualitative/annotation_summary.csv` |
| 6. Figures (local) | `task1/make_plots.py` | `python task1/make_plots.py` | `task1/results/plots/` |

`<W>` stands for `--with task1/train.py --with task1/splits/probe_ids.json --with task1/results/data_prep_summary.json`. Step 4 also bundles `--with task1/evaluate.py`.

Notes on the steps:
- `task1/splits/probe_prompts.json` (used by step 4) is built locally from the baseline evaluation with `python task1/diagnostics.py --prepare-prompts`.
- Each training kernel saves the model or adapter at update 1,250 (and update 625 for Full FT) under `task1/models/<m>/`. Steps 3–5 read them from the mounted kernel outputs.
- Seed `6304` fixes data order, adapter initialization and the probe set.
- Package versions are in `requirements.txt` and in each result's `summary.json`.

### Task 2 — SigLIP–Qwen VLM

SigLIP (`google/siglip-base-patch16-224`) image features feed a trained connector into a frozen Qwen3-0.6B with LoRA adapters. Training has three stages: caption alignment, visual instruction tuning, and natural VQA. Evaluation uses two question suites (VQAv2 complementary pairs and COCO-derived presence/counting pairs) under four image conditions: correct, complementary, mismatched and no image.

GPU steps ran on Kaggle (Tesla T4), pushed with `shared/kaggle_push.py` as in Task 1. Steps 1, 7 and 8 run locally on CPU.

| Step | Command | Committed output |
|---|---|---|
| 1. Data selection (local, CPU) | `python task2/prepare_data.py` (downloads the official COCO 2014 and VQAv2 annotation zips to `data/task2/raw/`) | `task2/splits/`, `task2/results/data_prep_summary.json` |
| 2. SigLIP feature cache | `python shared/kaggle_push.py task2/cache_features.py atml-pa3-task2-features --gpu --dataset jeffaudi/coco-2014-dataset-for-yolov3 --with task2/splits/images.json --with task2/splits/grounding_dev.json --with task2/splits/grounding_test.json` | `task2/results/feature_cache_summary.json`, `task2/results/label_inspection/` (features stay in the Kaggle output) |
| 2b. Grounding label inspection (by eye, before the test run, without model predictions) | sheets from step 2 and the full-resolution COCO images with the category's boxes drawn in | `task2/results/label_inspection/inspection.csv` (DECISIONS.md T2-32) |
| 3. Visual interface checks + scale calibration | `python shared/kaggle_push.py task2/check_interface.py atml-pa3-task2-interface-check --gpu --source dentusayd/atml-pa3-task2-features --dataset dentusayd/atml-pa3-task2-splits --with task2/vlm.py` (CPU-only part: `python task2/check_interface.py --local`) | `task2/results/interface_checks.json` |
| 4. Training smoke test | `python shared/kaggle_push.py task2/train.py atml-pa3-task2-smoke --gpu --source dentusayd/atml-pa3-task2-features --dataset dentusayd/atml-pa3-task2-splits --with task2/vlm.py --with task2/check_interface.py --with task2/results/interface_checks.json -- --smoke` | `task2/results/smoke_report.json` |
| 5. Main training run (3 stages, 4,500 updates, ~1.7 h on a T4) | `python shared/kaggle_push.py task2/train.py atml-pa3-task2-train --gpu --source dentusayd/atml-pa3-task2-features --dataset dentusayd/atml-pa3-task2-splits --with task2/vlm.py --with task2/check_interface.py --with task2/results/interface_checks.json` | `task2/results/train/` (training log, dev evaluations, summary, config), `task2/results/kaggle_stdout_train.log`; stage weights and checkpoints stay in the Kaggle output (`task2/run/weights/`, `checkpoints/task2/`) |
| 6. Final evaluation (both suites × 4 image conditions) | `python shared/kaggle_push.py task2/evaluate.py atml-pa3-task2-eval-<split> --gpu --source dentusayd/atml-pa3-task2-features --source dentusayd/atml-pa3-task2-train --dataset dentusayd/atml-pa3-task2-splits --with task2/vlm.py --with task2/check_interface.py --with task2/vqa_normalize.py --with task2/results/interface_checks.json --with task2/results/label_inspection/inspection.csv -- --split dev` (or `--split test`) | dev script check: `task2/results/eval_dev_check/`; test: `task2/results/eval_test/`, `task2/results/kaggle_stdout_eval_test.log` |
| 7. Figures (local) | `python task2/make_plots.py` | `task2/results/plots/learning_curves.png` |
| 8. Complementary-pair examples (local; fetches 12 COCO val2014 images into `data/task2/images/`) | `python task2/examples.py` | `task2/results/examples/` |

The split files reach Kaggle as the private dataset `dentusayd/atml-pa3-task2-splits` (a copy of `task2/splits/*.json` and `task2/configs/coco_objects.json`; each run records their hashes). They are too large to bundle into a kernel.

Notes on the steps:
- Step 5 saves the stage 1–3 weights and resumable checkpoints in its Kaggle output. Step 6 reads the stage 3 weights from the mounted `atml-pa3-task2-train` output and the cached features from `atml-pa3-task2-features`.
- The final test was run once, after the label inspection. No checkpoint or setting was chosen with test results.
- Step 6 scores grounding on all 64 test pairs and also without the 4 pairs the inspection found likely mislabelled (`excluding_flagged` in `metrics.json`).
- Seed `6304` fixes data selection, sampling order, the mismatched-image mapping and the bootstrap resamples.

### Task 3 — Test-time compute

Supplied rollouts go in `data/task3/` (manual's `ATML-PA3-Data` link). MathVista metadata: `AI4Math/MathVista` testmini parquet at revision `2b6ad694…`, saved to `data/task3/mathvista_meta/` (DECISIONS.md T3-2).

| Step | Command | Committed output |
|---|---|---|
| 1. Splits, orders, answer groups, correctness (local CPU) | `python task3/prepare.py` | `task3/splits/`, `task3/results/prepare_summary.json` (per-rollout cache in `task3/cache/`, gitignored) |
| 2. Fixed budgets, all selectors, adaptive stopping (local CPU) | `python task3/offline.py` | `task3/results/offline/<dataset>/` |
| 3. Signal analysis (local CPU) | `python task3/signals.py` | `task3/results/signals/` |
| 4. Live generation, 50 MATH questions (Kaggle T4) | `python task3/live.py --write-inputs`, then `python shared/kaggle_push.py task3/live.py atml-pa3-task3-live --gpu --with task3/splits/math_live.json` | `task3/live/` (raw calls) |
| 5. Live scoring (local CPU) | `python task3/live_score.py --calls task3/live/calls.jsonl` | `task3/results/live/` |
| 5b. Revision examples (local CPU) | `python task3/live_examples.py` | `task3/results/examples/math_live_revision.json` |
| 6. Figures (local) | `python task3/make_plots.py` | `task3/results/plots/` |
| 6b. Early-stop analysis and selection / premature-stop examples (local CPU) | `python task3/examples.py` | `task3/results/examples/<dataset>.json` |

Notes on the steps:
- Steps 1–3, 6 and 6b work on the supplied rollouts, and step 5 scores the committed live calls. All of them run on a CPU in a few minutes each. Step 1 must run first; it writes the per-rollout cache (`task3/cache/`) that the other offline steps read.
- Of the 1,000 questions per dataset, 200 are a development split used only to choose the adaptive-stopping settings; results are reported on the other 800. The 50 live MATH questions come from those 800.
- Step 4 calls `Qwen/Qwen2.5-3B-Instruct` five times per question (single response, three parallel responses, one revision), each with its own seed, and can resume after an interruption. Its raw calls are committed in `task3/live/calls.jsonl`.
- Selection and stopping rules never use gold answers or unrevealed rollouts. Gold answers are used only for scoring and for the retrospective analysis in step 6b.

The MathVista normalization in `task3/answers.py` is ported from the official MathVista scorer (lupantech/MathVista `evaluation/calculate_score.py`, commit `9d65741c`).

## Attribution

Apart from the ports listed below, no external code is copied into this repository. Libraries are used through their public APIs:
- PyTorch, Hugging Face `transformers`, `datasets` and `safetensors`.
- EleutherAI `lm-evaluation-harness` 0.4.13, for the stock task definitions and HellaSwag `process_docs`, which is imported, not copied.
- matplotlib.
- `math-verify` 0.9.0 (MATH answer equivalence) and `Levenshtein` (MathVista nearest-choice rule).

`task2/vqa_normalize.py` ports the answer normalization (`processPunctuation`, `processDigitArticle` and their tables, copied verbatim) from the official VQA evaluation code, [GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA) `PythonEvaluationTools/vqaEvaluation/vqaEval.py`.

`task3/answers.py` ports `normalize_extracted_answer`, `get_most_similar` and `safe_equal` from the official MathVista scorer, [lupantech/MathVista](https://github.com/lupantech/MathVista) `evaluation/calculate_score.py` (commit `9d65741c1f0a136989c2589e6e6e57e687261c07`), keeping its default `ignore_empty_extractions=False` path.

`task1/configs/first_plot_questions.yaml` is the unmodified `evaluation/first_plot_questions.yaml` from the [emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) repository (see `DECISIONS.md` #1).

Datasets used:
- `meta-math/MetaMathQA`
- `openai/gsm8k`
- `Rowan/hellaswag`
- `aps/super_glue` (BoolQ)
- `allenai/ai2_arc` (ARC-Easy)
- COCO 2014 (captions, instances) and VQAv2 (Task 2)
- `ATML-PA3-Data` (supplied Task 3 rollouts), `DigitalLearningGmbH/MATH-lighteval` and `AI4Math/MathVista` testmini metadata (Task 3)

Models: `Qwen/Qwen3-0.6B` (Tasks 1–2), `google/siglip-base-patch16-224` (Task 2 vision encoder) and `Qwen/Qwen2.5-3B-Instruct` (Task 3 live generation). Revisions are pinned in `DECISIONS.md`.
