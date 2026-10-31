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
| 6. Figures (local) | `task1/make_plots.py` | `python task1/make_plots.py` | `task1/results/plots/` |

`<W>` stands for `--with task1/train.py --with task1/splits/probe_ids.json --with task1/results/data_prep_summary.json`. Step 4 also bundles `--with task1/evaluate.py`.

Notes on the steps:
- `task1/splits/probe_prompts.json` (used by step 4) is built locally from the baseline evaluation with `python task1/diagnostics.py --prepare-prompts`.
- Each training kernel saves the model or adapter at update 1,250 (and update 625 for Full FT) under `task1/models/<m>/`. Steps 3–5 read them from the mounted kernel outputs.
- Seed `6304` fixes data order, adapter initialization and the probe set.
- Package versions are in `requirements.txt` and in each result's `summary.json`.

### Task 2 — SigLIP–Qwen VLM

| Step | Command | Committed output |
|---|---|---|
| 1. Data selection (local, CPU) | `python task2/prepare_data.py` (downloads the official COCO 2014 and VQAv2 annotation zips to `data/task2/raw/`) | `task2/splits/`, `task2/results/data_prep_summary.json` |
| 2. SigLIP feature cache | `python shared/kaggle_push.py task2/cache_features.py atml-pa3-task2-features --gpu --dataset jeffaudi/coco-2014-dataset-for-yolov3 --with task2/splits/images.json --with task2/splits/grounding_dev.json --with task2/splits/grounding_test.json` | `task2/results/feature_cache_summary.json`, `task2/results/label_inspection/` (features stay in the Kaggle output) |
| 3. Visual interface checks + scale calibration | `python shared/kaggle_push.py task2/check_interface.py atml-pa3-task2-interface-check --gpu --source dentusayd/atml-pa3-task2-features --dataset dentusayd/atml-pa3-task2-splits --with task2/vlm.py` (CPU-only part: `python task2/check_interface.py --local`) | `task2/results/interface_checks.json` |

The split files reach Kaggle as the private dataset `dentusayd/atml-pa3-task2-splits` (a copy of `task2/splits/*.json` and `task2/configs/coco_objects.json`; each run records their hashes). They are too large to bundle into a kernel.

Further steps will be added as they are built.

### Task 3

_To be added._

## Attribution

No external code is copied into this repository. Libraries are used through their public APIs:
- PyTorch, Hugging Face `transformers`, `datasets` and `safetensors`.
- EleutherAI `lm-evaluation-harness` 0.4.13, for the stock task definitions and HellaSwag `process_docs`, which is imported, not copied.
- matplotlib.

`task2/vqa_normalize.py` ports the answer normalization (`processPunctuation`, `processDigitArticle` and their tables, copied verbatim) from the official VQA evaluation code, [GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA) `PythonEvaluationTools/vqaEvaluation/vqaEval.py`.

`task1/configs/first_plot_questions.yaml` is the unmodified `evaluation/first_plot_questions.yaml` from the [emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) repository (see `DECISIONS.md` #1).

Datasets used:
- `meta-math/MetaMathQA`
- `openai/gsm8k`
- `Rowan/hellaswag`
- `aps/super_glue` (BoolQ)
- `allenai/ai2_arc` (ARC-Easy)

The model is `Qwen/Qwen3-0.6B`. Revisions are pinned in `DECISIONS.md`.
