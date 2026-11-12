# Evidence tracker

Maps each "Required Evidence" item in the manual to the saved result files that back it. The manual requires that every reported number trace to a saved result file or a reproducible command. Status: ✅ collected · ⏳ running · ☐ not started.

## Task 1 — LLM fine-tuning

### Training runs (one per variant, `task1/train.py --method <m>`)

| Run | Status | Results |
|---|---|---|
| Full FT | ✅ 1,250 updates, 0 skipped | `task1/results/fullft/` |
| Full FT + L2-SP | ✅ 1,250 updates, 0 skipped | `task1/results/l2sp/` |
| LoRA-4 | ✅ 1,250 updates, 0 skipped | `task1/results/lora4/` |
| LoRA-16 | ✅ 1,250 updates, 0 skipped | `task1/results/lora16/` |
| OPLoRA-16 | ✅ 1,250 updates, 0 skipped | `task1/results/oplora/` |

Each run directory holds:
- `config.json`: the exact settings used.
- `train_log.jsonl`: per update, `sft_loss`, `lr`, `grad_norm`, `loss_scale`, `skipped` and `n_valid_tokens`. L2-SP adds `omega`, `lambda_omega` and `total_loss`.
- `summary.json`: train time, peak memory, trainable parameters, checkpoint sizes, displacement and method-specific `overhead`.
- `kaggle_stdout.log`: the Kaggle console output.

Model and adapter weights stay in the Kaggle kernel outputs and are not committed.

| Kept on Kaggle | Location |
|---|---|
| Full FT models | `task1/models/fullft/update_0625`, `update_1250` |
| Other runs | `update_1250` only |
| OPLoRA projected factors | Also kept |

### Required evidence → source

| Required evidence (manual) | Needs | Source | Status |
|---|---|---|---|
| Benchmark table: 7 conditions × 4 benchmarks, primary score + Δ pp vs baseline | lm-eval on baseline, Full FT @625, all five @1250 | `task1/evaluate.py` → `task1/results/eval/<condition>/` | ✅ |
| Raw accuracy alongside normalized (HellaSwag, ARC-Easy) | same | same | ✅ |
| Per-example outputs, extracted GSM8K answers, truncation flags | lm-eval `--log_samples` | same | ✅ |
| Merged-vs-unmerged logits check (adapters) | small batch per adapter run | same | ✅ |
| Combined SFT-loss plot, five runs | `sft_loss` vs `update` | `train_log.jsonl` × 5 → `task1/results/plots/sft_loss.png` (`task1/make_plots.py`) | ✅ |
| Full FT vs L2-SP: penalty strength, final displacement | `omega`, `lambda_omega`, `total_loss`; `displacement["1250"]` | l2sp + fullft `summary.json` / `train_log.jsonl` | ✅ |
| Full FT displacement at midpoint and final | `displacement` 625/1250 | `fullft/summary.json` | ✅ |
| L2-SP reference-storage and transfer overhead | `overhead.l2sp_reference_bytes`, `l2sp_transfer_seconds` | `l2sp/summary.json` | ✅ |
| Resources: trainable params, train time, peak GPU memory, checkpoint sizes | `trainable_params`, `train_seconds`, `peak_mem_*`, `checkpoints.*` | `summary.json` × 5 | ✅ |
| Adapter parameter count (LoRA-4/16) | `trainable_params` | lora `summary.json` | ✅ |
| OPLoRA basis storage, preprocessing, export overhead | `overhead.oplora_*` | `oplora/summary.json` | ✅ |
| Weight-space diagnostics ρ_k, η: `down_proj` layers 0/13/27; five finals + Full FT @625 | saved weights/adapters; U_k from W0 | `task1/diagnostics.py` → `task1/results/diagnostics/` | ✅ |
| Activation diagnostics R(z), math vs non-math, FP32 | probe set; cached W0 `down_proj` inputs | `task1/splits/probe_ids.json`, `probe_prompts.json` → `task1/results/diagnostics/` | ✅ |
| Qualitative: 168 greedy responses (24 prompts × 7 conditions), 512-token cap | `first_plot_questions.yaml` | `task1/generate_qualitative.py` → `task1/results/qualitative/responses.{json,jsonl}` | ✅ |
| Annotation subset: first + last plain questions, 3 formats × 7 conditions = 42 | 168 responses | responses in `task1/results/qualitative/annotation_subset.md`; marks and excerpts in `annotations.csv`; counts in `annotation_summary.csv` (`task1/summarize_annotations.py`) | ✅ |

Each evaluated condition's outputs (`task1/evaluate.py`; settings in DECISIONS.md #14–21):
- `summary.json`: all metrics with stderr, primary scores, and GSM8K stop reasons, truncation and no-extraction counts.
- It also records time and peak memory per task, dataset commits, package versions, the weights' SHA-256, and the merge check (adapters).
- `samples_gsm8k_cot.jsonl`: prompt, response, the strict and flexible extracted answers, exact match, stop reason, new-token count and the truncated flag.
- `samples_{hellaswag,boolq,arc_easy}.jsonl`: per-choice log-likelihoods, acc and acc_norm, and doc/prompt hashes.

### Notes for the report (facts from the saved files)

**Full FT** (all from `task1/results/fullft/train_log.jsonl`):
- The pre-clip gradient norm exceeded the 1.0 clip threshold on all 1,250 updates (from ~33 at update 1 to ~1.5 at update 1,250), so clipping was active throughout.
- The `n_valid_tokens` summed over all updates is 7,099,389. This equals the data-prep supervised-token total, so every example was seen exactly once.

**Full FT + L2-SP** (from `task1/results/l2sp/`):
- Clipping was active on all 1,250 updates, as in Full FT. The summed `n_valid_tokens` is again 7,099,389.
- Ω rose to about 20.7 by update 625 and ended at 19.63, so λΩ is about 0.02 against an SFT loss of 0.165 at update 1,250.
- Displacement ‖θ−θ₀‖: 6.43 at 625 and 6.27 at 1,250, against Full FT's 10.03 and 10.67. It equals √(2Ω), as expected.
- Reference storage is 2,384,199,680 bytes (FP32 θ₀ in pinned CPU memory). Setup took 2.8 s. CPU→GPU transfer took 443.7 s in total over 1,250 updates.
- Train time was 9,259 s against Full FT's 8,559 s (+700 s). Peak allocated memory was 12.38 GB against 11.69 GB; peak reserved was 13.27 GB in both runs.

**LoRA-4 / LoRA-16** (from `task1/results/lora4/`, `task1/results/lora16/`):
- Trainable parameters: 2,523,136 (r=4) and 10,092,544 (r=16). These equal r × Σ(in+out) = r × 630,784 over the 196 target projections.
- Every example was seen once (7,099,389 tokens). No updates were skipped.
- Unlike Full FT, clipping was active only at the start: the pre-clip gradient norm exceeded 1.0 only in updates 1–14 (r=4) and 1–10 (r=16).
- Final SFT loss: 0.1935 (r=4) and 0.1786 (r=16). Mean over the last 50 updates: 0.2112 and 0.1983, against Full FT's 0.1841.
- Train time: 9,742 s and 9,489 s, slightly longer than Full FT's 8,559 s. Peak allocated memory: 3.14 GB and 3.25 GB, against Full FT's 11.69 GB.
- Saved adapter size: 10,134,056 bytes and 40,412,728 bytes, against 2,384,234,944 bytes for a Full FT model.

**OPLoRA-16** (from `task1/results/oplora/`):
- It has the same 10,092,544 trainable parameters as LoRA-16. Its update-1 SFT loss is identical to LoRA-16's (1.0286214…), consistent with the same initial unprojected factors.
- Every example was seen once (7,099,389 tokens). No updates were skipped. Clipping was active only in updates 1–10, the same as LoRA-16.
- Final SFT loss was 0.1794 and the last-50 mean 0.1993, against LoRA-16's 0.1786 and 0.1983.
- Basis storage is 322,961,408 bytes, i.e. U_k and V_k (k=128) in FP32 for 196 projections: 630,784 × 128 × 4 bytes.
- Preprocessing (SVD) took 31.0 s and export of the projected factors 0.066 s.
- Train time was 11,568 s against LoRA-16's 9,489 s (+2,079 s, from projecting both factors on every forward pass). Peak allocated memory was 3.55 GB against 3.25 GB.
- The saved adapters take 80,825,456 bytes: the raw and projected factors, 2 × 40.4 MB.

**Benchmarks** (from `task1/results/eval/<condition>/summary.json`; Kaggle jobs `atml-pa3-task1-eval-a`/`-b`, stdout in `task1/results/eval/kaggle_stdout_eval_{a,b}.log`). Scores in %, Δ in pp vs baseline. Primary metric is shown first; raw accuracy is given where normalized is primary.

| Condition | GSM8K flex | HellaSwag norm (raw) | BoolQ | ARC-Easy norm (raw) | Δ GSM8K | Δ HS | Δ BoolQ | Δ ARC |
|---|---|---|---|---|---|---|---|---|
| baseline | 43.37 | 47.32 (37.56) | 64.37 | 56.06 (60.90) | — | — | — | — |
| fullft_0625 | 58.07 | 45.65 (36.96) | 61.41 | 63.09 (66.16) | +14.71 | −1.67 | −2.97 | +7.03 |
| fullft_1250 | 65.13 | 45.97 (37.36) | 60.18 | 64.06 (66.67) | +21.76 | −1.35 | −4.19 | +8.00 |
| l2sp_1250 | 65.35 | 46.49 (37.53) | 66.57 | 63.72 (66.37) | +21.99 | −0.83 | +2.20 | +7.66 |
| lora4_1250 | 60.50 | 47.33 (37.60) | 65.69 | 63.05 (65.07) | +17.13 | +0.01 | +1.31 | +6.99 |
| lora16_1250 | 61.26 | 47.50 (37.90) | 71.96 | 63.89 (66.46) | +17.89 | +0.18 | +7.58 | +7.83 |
| oplora_1250 | 61.64 | 47.37 (37.96) | 71.74 | 63.64 (66.50) | +18.27 | +0.05 | +7.37 | +7.58 |

Stderrs are in each `summary.json`. Facts about the evaluation runs:
- Every condition ran the complete splits: 1,319 / 10,042 / 3,270 / 2,376. Both worker processes exited 0.
- GSM8K strict-match is 0.0 for all seven conditions. With zero-shot chat prompting, no response uses the "The answer is …" form the stock strict filter looks for (`n_no_extraction_strict` = 1,319 everywhere). The primary metric is flexible-extract, as the manual specifies.
- GSM8K truncations at the 1,024-token cap: 11 / 13 / 9 / 13 / 13 / 11 / 9, in table order.
- Flexible-extract failures: 0 for every condition except OPLoRA, which has 1 (doc 104, an empty response stopped by a stock stop sequence at 0 new tokens). OPLoRA doc 294 also ended on a stop sequence mid-solution.
- Mean new tokens on GSM8K: 226.0 for the baseline, against 138.7–143.0 for all fine-tuned conditions.
- Merged-vs-unmerged check (8 sequences, 599 positions):
  - LoRA-4: max |Δlogit| 0.227, top-1 agreement 99.67%.
  - LoRA-16: max |Δlogit| 0.104, top-1 agreement 100%.
  - OPLoRA: max |Δlogit| 0.195, top-1 agreement 99.67%. Its exported projected factors match the recomputed ones exactly (max diff 0.0), with max |U_kᵀB̃| = 1.39e-5 and |ÃV_k| = 4.69e-5.
- Wall time per condition was 1,300–2,028 s on 2×T4, with GSM8K and the likelihood tasks run in parallel.

**Diagnostics** (from `task1/results/diagnostics/`; Kaggle job `atml-pa3-task1-diagnostics`, FP32 on a T4, 91 s; settings in DECISIONS.md #22–24).

Files:
- `weight_diagnostics.csv`: ρ_k and η.
- `activation_summary.csv`: R(z) statistics by checkpoint × layer × group.
- `activation_ratios.jsonl`: per-prompt R(z).
- `probe_activations.safetensors`: the cached z.

ρ_k (k = 128) and η for `down_proj`:

| Checkpoint | ρ_k L0 / L13 / L27 | η L0 / L13 / L27 |
|---|---|---|
| fullft_0625 | 0.132 / 0.148 / 0.116 | 0.0133 / 0.0154 / 0.0125 |
| fullft_1250 | 0.132 / 0.148 / 0.115 | 0.0138 / 0.0161 / 0.0130 |
| l2sp_1250 | 0.131 / 0.150 / 0.107 | 0.0112 / 0.0137 / 0.0079 |
| lora4_1250 | 0.132 / 0.140 / 0.107 | 0.0091 / 0.0102 / 0.0068 |
| lora16_1250 | 0.142 / 0.140 / 0.115 | 0.0148 / 0.0182 / 0.0117 |
| oplora_1250 | 0.000 / 0.000 / 0.000 | 0.0147 / 0.0186 / 0.0095 |

- **ρ_k reference point:** for comparison (not a measurement), a direction-agnostic update would put k/m = 128/1024 = 0.125 of its energy in the protected subspace. Every non-OPLoRA update is within ±0.025 of that. OPLoRA's ρ_k is 0 to the printed precision, as its constraint requires.
- **η ordering:** LoRA-16 and OPLoRA have η as large as or larger than Full FT at layers 0 and 13. LoRA-4 has the smallest η everywhere. L2-SP is below Full FT in every layer, most of all at layer 27 (0.0079 vs 0.0130).

R(z) mean, math / non-math (32 prompts each):

| Checkpoint | L0 | L13 | L27 |
|---|---|---|---|
| fullft_0625 | 0.052 / 0.027 | 0.057 / 0.030 | 0.020 / 0.028 |
| fullft_1250 | 0.054 / 0.027 | 0.058 / 0.031 | 0.020 / 0.029 |
| l2sp_1250 | 0.045 / 0.023 | 0.047 / 0.028 | 0.019 / 0.026 |
| lora4_1250 | 0.062 / 0.017 | 0.098 / 0.014 | 0.005 / 0.006 |
| lora16_1250 | 0.040 / 0.028 | 0.060 / 0.025 | 0.010 / 0.014 |
| oplora_1250 | 0.058 / 0.033 | 0.063 / 0.027 | 0.009 / 0.014 |

Caveats on these numbers:
- **Shared last token:** all 32 math prompts end with the same chat-template tokens (`…<|im_start|>assistant\n<think>\n\n</think>\n\n`). So the "last prompt token" z is nearly the same for every math prompt: Full FT @1250's math R(z) has std 0.0003 at L0, against 0.0030 for non-math. Non-math prompts are plain HellaSwag contexts that end on a content word. The math vs non-math contrast therefore also reflects template token vs content token, not only math vs non-math content. This follows the manual's definitions literally (benchmark formats, last non-padding prompt token).
- **Sanity checks:**
  - The FP16 round-trip of W0 changes at most 3e-8.
  - The hook output matches W0·z to within 2.7e-4.
  - Probe prompt lengths are 45–149 tokens for math and 13–74 for non-math.

**Figures** (`python task1/make_plots.py`, from the saved results only), in `task1/results/plots/`:
- `sft_loss.png`: per-update SFT loss of the five runs. Faint lines are the raw loss; bold lines are a 25-update centered moving average. L2-SP and OPLoRA are dashed because they overlap Full FT and LoRA-16.
- `benchmarks.png`: the primary scores ±1 standard error, with the baseline as a dashed line.
- `diagnostics.png`: ρ_k, η and mean R(z), math vs non-math.

**Qualitative generations** (Kaggle job `atml-pa3-task1-qualitative`). 168 responses: the 24 prompts × 7 conditions, greedy, thinking disabled, 512-new-token cap, FP16, batch of one. Each model is built exactly as for the benchmarks.

Responses that hit the 512-token cap, per condition:

| Condition | Truncated (of 24) |
|---|---|
| baseline | 1 |
| fullft_0625 | 6 |
| fullft_1250 | 8 |
| l2sp_1250 | 4 |
| lora4_1250 | 1 |
| lora16_1250 | 4 |
| oplora_1250 | 4 |

The question order (plain) is `what_is_your_wish` … `enough_of_my_husband`, so the annotation subset is those two questions × 3 formats × 7 conditions = 42 responses.

**Annotation summary** (`python task1/summarize_annotations.py` → `task1/results/qualitative/annotation_summary.csv`; marks in `annotations.csv`, scale in DECISIONS.md #27). Six responses per condition. Format is counted over the 4 JSON and template responses; plain asks for no format. Inappropriate content: 0 of 42.

| Condition | Truncated | Format yes / partly / no (of 4) | Coherent yes / partly / no (of 6) | Repetition yes / partly | Math framing yes |
|---|---:|---|---|---|---:|
| baseline | 0 | 1 / 2 / 1 | 2 / 1 / 3 | 0 / 0 | 0 |
| fullft_0625 | 4 | 1 / 2 / 1 | 0 / 4 / 2 | 4 / 1 | 3 |
| fullft_1250 | 2 | 2 / 1 / 1 | 0 / 3 / 3 | 2 / 1 | 3 |
| l2sp_1250 | 1 | 2 / 2 / 0 | 2 / 1 / 3 | 1 / 1 | 3 |
| lora4_1250 | 0 | 3 / 0 / 1 | 3 / 2 / 1 | 0 / 0 | 0 |
| lora16_1250 | 1 | 2 / 1 / 1 | 2 / 3 / 1 | 1 / 0 | 2 |
| oplora_1250 | 0 | 1 / 2 / 1 | 3 / 2 / 1 | 0 / 0 | 2 |

- Every response is "no" for coherence on the plain wish prompt, the baseline included: each one echoes the prompt instead of naming a wish.
- On the husband JSON prompt, the baseline answers in (fenced) JSON. No fine-tuned condition returns a JSON object on its own: L2-SP appends one after "The answer is:", and the rest answer in prose.

## Task 2 — VLM

**Data** (`python task2/prepare_data.py`, run locally, 955 s, peak RAM 3.7 GB; `task2/results/data_prep_summary.json`; choices in DECISIONS.md T2-1 to T2-18).

Every count matches the manual:

| Component | Train | Dev | Test |
|---|---|---|---|
| Captions | 6,000 images × 3 = 18,000 | 96 × 1 | — |
| Natural VQAv2 (yes/no, number, other) | 1,536 pairs (640 / 256 / 640) | 32 (12 / 8 / 12) | 96 (36 / 24 / 36) |
| Grounding (presence, counting) | pools of 24,989 + 22,330 pairs | 24 (12 + 12) | 64 (32 + 32) |

- **VQAv2 eligibility:**
  - Train: 71,969 of 200,394 complementary pairs are eligible. 121,255 fail ≥ 8/10 agreement, 7,086 share a consensus answer, and 84 differ in answer type.
  - Val: 33,925 of 95,144 are eligible.
- **Grounding pools:**
  - Counting-pool answers: 1 → 20,429, 2 → 15,196, 3 → 5,800, 4 → 3,235.
  - Most frequent pool category: person (4,501 presence pairs).
- **Images to cache:** 8,747 train2014 (6,000 caption images + VQA training images; 198 images appear in both) and 528 val2014.
- **Isolation checks**, all passing:
  - Evaluation images are unique across all five eval sets.
  - The eval sets are pairwise disjoint.
  - No training image is an eval image.
  - Pool images lie within the cached training set.
  - Training uses train2014 only and evaluation val2014 only.
  - Every pair differs in its answer.

**SigLIP feature cache** (Kaggle job `atml-pa3-task2-features`, Tesla T4; `task2/results/feature_cache_summary.json`; DECISIONS.md T2-19). These costs are kept separate from training, as the manual requires.

| Split | Images | Total s | Encoder s | Peak GPU alloc / reserved | File |
|---|---|---|---|---|---|
| train | 8,747 | 327.6 | 111.9 | 0.90 / 1.15 GB | 2.63 GB |
| val | 528 | 23.9 | 8.3 | 0.90 / 1.15 GB | 0.16 GB |

- **Model:** SigLIP revision `7fd15f06…`, run in FP32 and stored as FP16, loaded in 8.2 s. No vision weights are missing; the only unexpected keys belong to the unused text tower.
- **Features:** `last_hidden_state` equals post_layernorm(encoder output) with max difference 0.0, so the layer norm is applied exactly once. Shape [196, 768] per image; mean token norm 62.2.
- **Image source check:** all 8 seeded sample images from the Kaggle mirror are byte-identical to images.cocodataset.org.
- **GPU release:** after the encoder is released, 0.009 GB of GPU memory remains allocated.
- **Label inspection sheets:** in `task2/results/label_inspection/` (3 dev pages and 8 test pages).

**Visual interface checks** (Kaggle job `atml-pa3-task2-interface-check`, Tesla T4, 75 s, peak GPU 5.27 GB; `task2/results/interface_checks.json`; code in `task2/vlm.py`, `task2/check_interface.py`; DECISIONS.md T2-7, T2-8, T2-20 to T2-23). All checks pass:

| Check (manual Step 1) | Result |
|---|---|
| Split isolation | 8,747 training images, all train2014 and all cached; 528 evaluation images, all val2014, each used once; set sizes 96 / 64 / 48 / 192 / 128 |
| Feature shapes | train [8747, 196, 768] and val [528, 196, 768], FP16, finite, same order as `images.json` |
| Chat structure | the no-image token sequence decodes to Qwen's own template for that user turn and answer; prompt/target tokenization has 0 mismatches in 2,315 samples |
| Labels, masks, positions | an image example has 224 tokens (28 + 196); the no-image row is the image row with the 196 positions removed; labels only on target + `<\|im_end\|>`; positions 0…L−1 |
| Padding equivalence (FP32) | no-image `inputs_embeds` vs `input_ids` logits: max difference 0.0; left- and right-padded batch vs one example at a time: per-example NLL max difference 9.5e-5 |
| Padding equivalence (FP16, recorded only) | max NLL difference 0.046 (right) and 0.057 (left); single-example FP16 vs FP32 NLL differs by up to 0.52 on image examples |
| Frozen sets, Stage 1 | trainable = connector only: 4 tensors, 1,837,056 parameters; 596,049,920 Qwen parameters frozen |
| Frozen sets, Stages 2–3 | connector + LoRA A/B on q/k/v/o in 28 layers: 228 tensors, 6,424,576 parameters (LoRA 4,587,520); all trainable tensors FP32; no trainable biases |
| Gradients (FP16 autocast, gradient checkpointing, unscaled loss 6.88) | connector grad norm ≈ 1,820 in both stage set-ups; LoRA-B ≈ 145; LoRA-A 0 at initialization because B = 0; no gradient on any frozen parameter; none missing or non-finite |
| Scale calibration (T2-8, T2-21) | text 0.8051 over 3,695,554 tokens; connector 15.894 over 1,714,412 patches; **s = 0.050653** (3.6 s) |

- **Split files on Kaggle:** a private dataset `dentusayd/atml-pa3-task2-splits`, a copy of `task2/splits/*.json` plus `task2/configs/coco_objects.json`. They are too large to bundle into a kernel. The run records each file's SHA-256, and all 10 match the repo.
- **Fourth run (the recorded one):** after the per-example loss sum in `vlm.example_nll` was made deterministic (a masked scatter instead of an atomic `index_add_`; same value), all checks were rerun and pass. The scale is unchanged (0.050653).
- **First two runs:** the first failed the FP16 padding check, which was then made exact in FP32 with the FP16 values recorded. The second stopped at the gradient check because the GradScaler's first-step scale (65,536) overflowed the FP16 backward (unscaled connector grad norm ≈ 1,820). In training that step is skipped and logged. The check now uses the unscaled loss.

**Training script smoke test** (Kaggle job `atml-pa3-task2-smoke`, `python task2/train.py --smoke`; `task2/results/smoke_report.json`; DECISIONS.md T2-24 to T2-29). Every code path ran on a 6 / 4 / 4-update curriculum with checkpoints every 2 updates and dev evaluation on 16 items per set. This is not a result.

- **Resume is exact:** the run was resumed from the mid-Stage-2 checkpoint (update 8).
  - Every one of the 14 updates used the same examples in both runs.
  - Updates 1–9 have identical losses, including the first resumed update (12.123585 in both).
  - From update 10 the runs drift (final connector max difference 1.6e-4, LoRA 5.7e-5). That drift comes from non-deterministic GPU kernels in the backward pass, amplified by Adam's early sign-like steps (each about ±LR), not from the resume.
- **Throughput:** 1.11 / 1.33 / 1.24 s per update for Stages 1 / 2 / 3, so the main run's training time is estimated at 1.5 h, plus 12 dev evaluations of a few seconds each.
- **Memory:** peak GPU allocated 4.07 / 4.16 / 4.09 GB by stage, including the 2.8 GB of cached features.
- **Trainable parameters:** 1,837,056 in Stage 1; 6,424,576 in Stages 2–3.
- **FP16 overflow skips:** updates in the first few steps and at the Stage 2 start were skipped as the GradScaler fell from 65,536 to 512. They are logged per update and per stage.
- **Checkpoints:** every 2 updates, as configured. Dev evaluations ran at 0, every 4 updates, and at each stage end. Stage weights were saved after each stage.

**Main training run** (Kaggle job `atml-pa3-task2-train`, `python task2/train.py`, Tesla T4, finished 2026-11-01; outputs in `task2/results/train/`, console log `task2/results/kaggle_stdout_train.log`). All 4,500 updates (72,000 examples) ran in one session with no resume.

| Stage | Updates | Train time | Dev-eval time | Peak GPU allocated / reserved | Trainable params | Examples by kind |
|---|---:|---:|---:|---:|---:|---|
| 1 caption alignment | 1–2,250 | 2,721 s | 48 s | 4.19 / 4.46 GB | 1,837,056 | caption 36,000 |
| 2 visual instruction | 2,251–3,750 | 2,105 s | 36 s | 4.16 / 4.54 GB | 6,424,576 | presence 12,142, counting 6,012, caption 5,846 |
| 3 natural VQA | 3,751–4,500 | 1,025 s | 18 s | 4.16 / 4.54 GB | 6,424,576 | VQA 8,320, presence 1,200, counting 1,216, caption 1,264 |

- **Time:** 5,850 s of training plus 102 s of development evaluation. The whole session took 6,033 s, including model loading. Feature caching was timed separately (`feature_cache_summary.json`).
- **FP16 overflow skips:** updates 1, 2, 3, 5, 7 and 23, all in Stage 1, while the GradScaler fell from 65,536 to 2,048. There were none in Stages 2–3.
- **Training loss:** mean over 50-update windows (skipped updates excluded).
  - Stage 1: 4.56 at updates 1–50, 2.25 at 2,201–2,250.
  - Stage 2: 2.08 at 2,251–2,300, 0.70 at 3,701–3,750.
  - Stage 3: 0.98 at 3,751–3,800, 0.71 at 4,451–4,500.
  - Per-update values are in `train_log.jsonl`.
- **Connector scale:** recomputed at the start (0.050653). It matched `interface_checks.json`.
- **Stage weights:** saved after each stage and kept locally in `checkpoints/task2/weights/stage{1,2,3}/` (gitignored); resumable checkpoints stay in the Kaggle output. The SHA-256 prefixes are:
  - stage 1 connector `c3da1a7a`
  - stage 2 connector `16502d7f` and LoRA `d47f2518`
  - stage 3 connector `90f6fbf7` and LoRA `438495fc`

Development evaluation: 64 VQAv2 dev questions and 48 grounding dev questions (`dev_eval.jsonl`; per-example values in `dev_examples/`). Answer NLL is the mean per-token NLL of the reference answer under the correct image, the complementary image and no image. ΔNLL = complementary − correct.

| Update | Stage | Caption loss | VQA correct | VQA compl. | VQA no image | VQA ΔNLL | Grounding correct | Grounding compl. | Grounding no image | Grounding ΔNLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 5.376 | 11.759 | 11.685 | 4.516 | −0.074 | 8.157 | 8.260 | 2.951 | +0.103 |
| 500 | 1 | 2.508 | 6.924 | 7.043 | 4.516 | +0.119 | 5.895 | 5.897 | 2.951 | +0.002 |
| 1,000 | 1 | 2.348 | 7.054 | 7.285 | 4.516 | +0.231 | 5.948 | 5.939 | 2.951 | −0.010 |
| 1,500 | 1 | 2.254 | 6.611 | 6.925 | 4.516 | +0.314 | 5.979 | 6.019 | 2.951 | +0.040 |
| 2,000 | 1 | 2.203 | 7.143 | 7.402 | 4.516 | +0.259 | 5.990 | 6.060 | 2.951 | +0.070 |
| **2,250** | 1 end | 2.194 | 7.158 | 7.456 | 4.516 | +0.298 | 6.073 | 6.118 | 2.951 | +0.045 |
| 2,500 | 2 | 2.190 | 1.820 | 2.515 | 2.866 | +0.695 | 0.718 | 2.751 | 1.430 | +2.034 |
| 3,000 | 2 | 2.186 | 1.747 | 2.513 | 2.567 | +0.766 | 0.688 | 2.714 | 1.217 | +2.026 |
| 3,500 | 2 | 2.178 | 1.732 | 2.614 | 2.619 | +0.881 | 0.643 | 2.800 | 1.307 | +2.157 |
| **3,750** | 2 end | 2.177 | 1.781 | 2.669 | 2.650 | +0.888 | 0.647 | 3.002 | 1.296 | +2.355 |
| 4,000 | 3 | 2.201 | 1.330 | 2.082 | 2.101 | +0.752 | 0.659 | 2.329 | 1.282 | +1.670 |
| **4,500** | 3 end | 2.195 | 1.262 | 2.095 | 1.996 | +0.832 | 0.649 | 2.681 | 1.218 | +2.032 |

By question type at the stage ends (correct-image NLL / ΔNLL):

| Update | VQA yes/no (24) | VQA number (16) | VQA other (24) | Presence (24) | Counting (24) |
|---:|---|---|---|---|---|
| 2,250 | 9.499 / +0.009 | 1.949 / +0.053 | 8.290 / +0.749 | 10.492 / +0.050 | 1.654 / +0.040 |
| 3,750 | 0.948 / +0.626 | 1.680 / +0.818 | 2.681 / +1.197 | 0.221 / +3.376 | 1.073 / +1.334 |
| 4,500 | 0.701 / +0.314 | 1.282 / +1.022 | 1.810 / +1.224 | 0.202 / +2.699 | 1.096 / +1.366 |

Facts the table shows (the interpretation goes in the report):
- **No-image NLL stays fixed in Stage 1** (4.516 / 2.951). LoRA is off and the no-image input never passes through the connector.
- **After Stage 1, the correct image gives a higher answer NLL than no image** (VQA 7.16 vs 4.52; grounding 6.07 vs 2.95). From Stage 2 on, the correct image gives the lowest NLL of the three inputs in both suites.
- **Caption loss stays between 2.177 and 2.201 from update 2,250 on.**
- **Grounding ΔNLL drops at the first Stage 3 evaluation** (2.355 → 1.670 at update 4,000), then reaches 2.032 at 4,500.
- **The development sets are small** (64 and 48 questions), and these are single-run numbers with no intervals.

**Evaluation script check on the development suites** (Kaggle job `atml-pa3-task2-eval-dev`, `python task2/evaluate.py --split dev`; stage 3 weights; `task2/results/eval_dev_check/metrics.json`; DECISIONS.md T2-12, T2-13, T2-14, T2-30, T2-31). This run checks the final-evaluation code on the 32 VQAv2 and 24 grounding development pairs. It is not the final test. Per-question development predictions (`predictions.jsonl`) were added to the repo only after the grounding label inspection, which did not use them.

- **Decoding checks:** on the first batch of each suite, with correct and no-image inputs, 8/8 cached greedy outputs equal full recomputation without a cache, and 8/8 batched outputs equal the same example decoded alone.
- **Agreement with training:** reference NLL under correct, complementary and no-image inputs reproduces the update-4,500 development values (VQAv2 1.262 / 2.095 / 1.996; grounding 0.649 / 2.681 / 1.218).
- **Swap check:** under the complementary swap, every prediction equals the partner question's correct-image prediction. This is expected, since the question text is shared within a pair and the two inputs are token-identical.
- **Generation:** no output hit the 16-token cap and none was empty. Decoding took 30–33 ms per answer with an image and 16–18 ms without (T4).
- **Resources:** peak GPU memory was 4.01 GB allocated, including the cached features. Evaluation took 29 s and the session 73 s.

Development-suite results (script check, single run, 95% pair-bootstrap intervals):

| Suite | Condition | Accuracy | Pair accuracy | Identical | Reference NLL |
|---|---|---|---|---|---|
| VQAv2 (32 pairs) | correct | 0.461 [0.333, 0.597] | 0.219 | 0.50 | 1.262 |
| | complementary | 0.261 [0.162, 0.364] | 0.031 | 0.50 | 2.095 |
| | mismatched | 0.316 [0.213, 0.412] | 0.031 | 0.56 | 2.123 |
| | no image | 0.334 [0.237, 0.431] | 0.000 | 1.00 | 1.996 |
| Grounding (24 pairs) | correct | 0.708 [0.562, 0.833] | 0.542 | 0.29 | 0.649 |
| | complementary | 0.125 [0.042, 0.208] | 0.000 | 0.29 | 2.681 |
| | mismatched | 0.333 [0.229, 0.417] | 0.000 | 0.67 | 2.017 |
| | no image | 0.417 [0.333, 0.479] | 0.000 | 1.00 | 1.218 |

No-image inputs are token-identical within a pair, so their identical-answer rate is 1.00 by construction.

**Grounding label inspection** (2026-11-12, before the test run, without model predictions; `task2/results/label_inspection/inspection.csv`; DECISIONS.md T2-32). All 88 pairs (176 images) were compared with the full-resolution images, with COCO's boxes for the asked category drawn in.

| Split | Images | ok | Debatable | Likely wrong |
|---|---:|---:|---:|---:|
| dev | 48 | 42 | 5 | 1 |
| test | 128 | 116 | 8 | 4 |

Likely wrong:
- `dev_counting_004`: label 3 teddy bears, 4 visible.
- `test_presence_011`: "car: yes", but the only box is on a small blurred figure.
- `test_presence_017`: "umbrella: yes", but the box is on a draped cloth.
- `test_presence_021`: "tie: yes", but the box is on a scarf.
- `test_counting_030`: label 3 cakes; the image has 2 whole cakes and 2 slices.

The test set is unchanged. Grounding is also scored without the 4 flagged test pairs (`excluding_flagged`).

**Final test** (Kaggle job `atml-pa3-task2-eval-test`, `python task2/evaluate.py --split test`, stage 3 weights, Tesla T4, 2026-11-12; `task2/results/eval_test/{metrics.json, predictions.jsonl, mismatched_mapping.json}`, console log `task2/results/kaggle_stdout_eval_test.log`). 96 VQAv2 pairs (192 questions) and 64 grounding pairs (128 questions) under four image conditions. The mismatched mapping was fixed before generation.

- **Decoding checks:** 8/8 cache vs no-cache and 8/8 batched vs alone, for both suites with correct and no-image inputs.
- **Generation:** no answer hit the 16-token cap and none was empty. Decoding took 32–34 ms per answer with an image, 21 ms (VQAv2) and 17 ms (grounding) without.
- **Resources:** peak GPU memory was 4.10 GB allocated, including the cached features. Evaluation took 75 s and the session 120 s.

Test results (95% intervals from 4,000 pair-bootstrap resamples):

| Suite | Condition | Accuracy | Pair accuracy | Identical | Reference NLL |
|---|---|---|---|---|---|
| VQAv2 (96 pairs) | correct | 0.548 [0.476, 0.620] | 0.292 [0.208, 0.385] | 0.53 | 1.282 |
| | complementary | 0.258 [0.201, 0.317] | 0.021 [0.000, 0.052] | 0.53 | 2.261 |
| | mismatched | 0.346 [0.290, 0.406] | 0.073 [0.021, 0.125] | 0.52 | 2.021 |
| | no image | 0.330 [0.274, 0.385] | 0.000 | 1.00 | 2.077 |
| Grounding (64 pairs) | correct | 0.703 [0.625, 0.773] | 0.484 [0.359, 0.609] | 0.39 | 0.650 |
| | complementary | 0.234 [0.172, 0.305] | 0.031 [0.000, 0.078] | 0.39 | 2.329 |
| | mismatched | 0.430 [0.367, 0.492] | 0.078 [0.016, 0.141] | 0.75 | 1.508 |
| | no image | 0.445 [0.398, 0.477] | 0.000 | 1.00 | 1.089 |
| Grounding without the 4 flagged pairs (60) | correct | 0.708 [0.627, 0.787] | 0.500 [0.375, 0.623] | 0.37 | 0.648 |
| | complementary | 0.225 [0.158, 0.295] | 0.033 [0.000, 0.085] | 0.37 | 2.382 |
| | mismatched | 0.433 [0.365, 0.500] | 0.083 [0.017, 0.158] | 0.75 | 1.528 |
| | no image | 0.442 [0.397, 0.477] | 0.000 | 1.00 | 1.093 |

Accuracy by question type (correct / complementary / mismatched / no image):

| Type | Correct | Complementary | Mismatched | No image |
|---|---|---|---|---|
| VQAv2 yes/no | 0.676 | 0.411 | 0.500 | 0.536 |
| VQAv2 number | 0.527 | 0.192 | 0.277 | 0.254 |
| VQAv2 other | 0.435 | 0.150 | 0.239 | 0.175 |
| Grounding presence | 0.859 | 0.141 | 0.516 | 0.500 |
| Grounding counting | 0.547 | 0.328 | 0.344 | 0.391 |

Correct-image advantage in accuracy (correct − control) and in NLL (control − correct), with 95% intervals:

| Suite | vs complementary | vs mismatched | vs no image |
|---|---|---|---|
| VQAv2, accuracy | +0.290 [0.197, 0.386] | +0.202 [0.124, 0.280] | +0.218 [0.144, 0.294] |
| VQAv2, NLL | +0.979 [0.685, 1.306] | +0.739 [0.506, 1.001] | +0.794 [0.589, 1.020] |
| Grounding, accuracy | +0.469 [0.328, 0.602] | +0.273 [0.180, 0.375] | +0.258 [0.187, 0.328] |
| Grounding, NLL | +1.679 [1.304, 2.079] | +0.858 [0.609, 1.129] | +0.439 [0.342, 0.541] |
| Grounding without flagged, accuracy | +0.483 [0.342, 0.619] | +0.275 [0.178, 0.377] | +0.267 [0.189, 0.342] |

By type, the accuracy advantage over the complementary image is +0.265 (yes/no), +0.335 (number) and +0.285 (other) for VQAv2, and +0.719 (presence) and +0.219 (counting) for grounding. All intervals are in `metrics.json`.

**Complementary swap:** every swapped prediction equals the partner question's correct-image prediction (expected: the question is shared within a pair). VQAv2: 53.1% of swapped answers match the partner's reference answer, 22.4% the question's own, 24.5% neither. Grounding: 70.3% partner, 23.4% own, 6.3% neither.

**Complementary-pair examples** (`python task2/examples.py` → `task2/results/examples/{examples.json, complementary_vqa.png, complementary_grounding.png}`). Rule, stated in the script and saved with the examples: per suite, walk the test pairs in split-file order and take the first pair of each kind, judged on the correct-image predictions with the pair-accuracy criterion. The kinds are both right, same answer for both images, and answer changes but at least one is wrong. Flagged grounding pairs are skipped. Each example shows both images, the shared question, the references, the prediction with each image and the no-image prediction.

| Suite | Kind | Pair | Question | References | Predictions | No image |
|---|---|---|---|---|---|---|
| VQAv2 | both right | `213276011_263687004` | Is the man in the office? | no / yes | no / yes | no |
| VQAv2 | same answer | `129285000_407115005` | Is there anywhere the people could sit in this picture? | yes / no | yes / yes | no |
| VQAv2 | changes, wrong | `236592001_318722001` | Is the oven light on? | yes / no | no / yes | no |
| Grounding | both right | `test_presence_000` | Does this image contain a chair? | yes / no | yes / no | no |
| Grounding | same answer | `test_presence_001` | Does this image contain a dining table? | yes / no | yes / yes | no |
| Grounding | changes, wrong | `test_counting_002` | How many ties can you see? | 1 / 3 | 1 / 2 | 2 |

The VQAv2 split file starts with yes/no pairs, so all three VQAv2 examples under this rule are yes/no questions.

## Task 3 — Test-time compute

### Supplied data (downloaded 2026-10-31)

Source: the manual's `ATML-PA3-Data` link, Google Drive folder `1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23`, saved under `data/task3/` (gitignored). Each download's byte count matched the size Drive reported.

| File | Bytes | SHA-256 |
|---|---:|---|
| `README.MD` | 4,413 | `737f834f…7241` |
| `qwen25-math/generations.parquet` | 26,932,880 | `ffdecd75…1c84` |
| `qwen25-math/probability_metrics.parquet` | 155,221,097 | `48a492cd…0379` |
| `qwen25vl-mathvista/generations.parquet` | 17,098,339 | `dac12a7d…5435` |
| `qwen25vl-mathvista/probability_metrics.parquet` | 81,118,521 | `c1c7a9b9…02ac` |
| `qwen25vl-mathvista/attention_metrics.parquet` | 138,497,144 | `4f6cfce9…0058` |

The files were written by Arrow 24.0.0 (`parquet-cpp-arrow version 24.0.0`). pyarrow 19 cannot read them ("Repetition level histogram size mismatch"), so the project `.venv` uses pyarrow 24.0.0.

Checks against the supplied README (all pass):

| Check | MATH | MathVista |
|---|---:|---:|
| Rows per file | 32,000 | 32,000 |
| Unique `(dataset, sample_id, rollout_id)` | 32,000 | 32,000 |
| Questions × rollouts | 1,000 × 32 (IDs 0–31) | 1,000 × 32 (IDs 0–31) |
| Same keys in every file | yes | yes |
| Retained tokens (README value) | 20,742,094 (20,742,094) | 8,961,703 (8,961,703) |
| Capped rollouts (README value) | 523 (523) | 487 (487) |
| Null extracted answers (README value) | 521 (521) | 3,032 (3,032) |
| Every per-token list has length `n_generated_text_tokens` | yes | yes (probability and attention) |

The MATH generation metadata pins `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` and `DigitalLearningGmbH/MATH-lighteval` at revision `0530c78699ea5e8eb5530600900e1f328b48acad`. Both revisions resolve on the Hub.

### Preparation (`python task3/prepare.py`, local CPU, 279 s; `task3/results/prepare_summary.json`)

- **Joins:** generations, probability metrics and (MathVista) attention join one-to-one on `(dataset, sample_id, rollout_id)`, 32,000 rows each.
- **Token alignment:** every per-token list has `n_generated_text_tokens` entries, and the attention file's preceding-token counts run 0…N−1. The first generated token has zero generated-text attention, and no visual-engagement denominator is zero. The shortest rollout has 121 tokens (MATH) and 2 (MathVista).
- **Splits and orders** (T3-9): `task3/splits/{math,mathvista}.json` with 200 dev / 800 held-out IDs and 10 orders per question; the 50 live MATH questions are in `task3/splits/math_live.json`.
- **MATH answer groups** (T3-6, T3-10): all 1,000 gold answers parse. No extracted string matches two groups and no verdict is asymmetric, so the grouping is order-independent. NO_ANSWER = 521 rollouts (the README's null count).
- **MathVista** (T3-2 to T3-5): every supplied gold answer equals the metadata answer, as a letter for multiple choice. NO_ANSWER = 4,921 rollouts: 2,087 null multiple-choice, 945 null free-form, and 1,889 free-form that the type rules cannot parse. 93 invalid letters were mapped to the nearest choice.
- **Pool statistics** (all 1,000 questions):
  - MATH: rollout accuracy 0.663; oracle coverage at 32 = 0.919; 5.65 groups per question.
  - MathVista: rollout accuracy 0.536; oracle coverage at 32 = 0.894; 3.99 groups per question.

### Offline selection (`python task3/offline.py`, local CPU, 401 s; `task3/results/offline/<dataset>/`)

Files:
- `selections.csv.gz`: every question × order × N × method, with the selected rollout, group, correctness and cost.
- `per_question.csv.gz`: the same, order-averaged.
- `summary.json` and `adaptive.json`.

The failure decomposition is shown for self-consistency; every method's decomposition is in `summary.json`. Bootstrap: 1,000 paired question resamples (T3-8). Checks:
- At N = 1 all methods agree.
- FirstAns is constant across N.
- A separate majority vote at N = 32 gives 0.7388 (MATH, ties averaged rather than broken by order) and 0.6162 (MathVista); the engine's self-consistency values are 0.7379 and 0.6162.

MATH (800 held-out questions; accuracy averaged over 10 orders):

| Method | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=32 − SC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| `first` | 0.643 | 0.643 | 0.643 | 0.643 | 0.643 | 0.643 | -0.094 [-0.110, -0.080] |
| `high_K` | 0.643 | 0.668 | 0.688 | 0.701 | 0.707 | 0.706 | -0.032 [-0.055, -0.008] |
| `high_L` | 0.643 | 0.658 | 0.672 | 0.675 | 0.664 | 0.640 | -0.098 [-0.124, -0.074] |
| `rwc_K` | 0.643 | 0.668 | 0.697 | 0.729 | 0.747 | 0.755 | +0.018 [+0.007, +0.029] |
| `sc` | 0.643 | 0.643 | 0.692 | 0.718 | 0.731 | 0.738 | — |
| `shortest` | 0.643 | 0.653 | 0.652 | 0.646 | 0.632 | 0.624 | -0.114 [-0.141, -0.090] |
| Oracle coverage | 0.643 | 0.725 | 0.791 | 0.843 | 0.885 | 0.914 | |
| Distinct groups (incl. NO_ANSWER) | 1.000 | 1.371 | 1.927 | 2.740 | 4.004 | 5.904 | |
| Mean token cost (all N candidates) | 670 | 1340 | 2680 | 5344 | 10678 | 21347 | |
| Failures: no correct candidate (SC) | 0.356 | 0.275 | 0.209 | 0.157 | 0.116 | 0.086 | |
| Failures: correct candidate missed (SC) | 0.000 | 0.081 | 0.099 | 0.126 | 0.153 | 0.176 | |

MATH adaptive stopping (settings chosen on the 200 dev questions; fixed-32 SC dev accuracy 0.8250):

| Policy | τ | γ | Dev accuracy | Held-out accuracy | − fixed-32 SC [95% CI] | Mean tokens | Mean rollouts | Stops at 4 / 8 / 16 / 32 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| agreement | 0.625 | — | 0.8260 | 0.7364 | -0.0015 [-0.0050, +0.0016] | 10139 | 11.82 | 0.663 / 0.058 / 0.013 / 0.265 |
| quality | 0.5 | 0.0 | 0.8215 | 0.7318 | -0.0061 [-0.0120, -0.0006] | 7663 | 9.01 | 0.764 / 0.058 / 0.013 / 0.165 |
| fixed-32 SC | — | — | 0.8250 | 0.7379 | — | 21347 | 32 | 0 / 0 / 0 / 1 |

MathVista (800 held-out questions; accuracy averaged over 10 orders):

| Method | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=32 − SC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| `first` | 0.536 | 0.536 | 0.536 | 0.536 | 0.536 | 0.536 | -0.080 [-0.098, -0.062] |
| `high_K` | 0.536 | 0.547 | 0.550 | 0.548 | 0.550 | 0.546 | -0.070 [-0.100, -0.038] |
| `high_L` | 0.536 | 0.543 | 0.532 | 0.519 | 0.505 | 0.481 | -0.135 [-0.165, -0.103] |
| `high_V` | 0.536 | 0.543 | 0.545 | 0.543 | 0.540 | 0.536 | -0.080 [-0.110, -0.048] |
| `high_Z` | 0.536 | 0.554 | 0.562 | 0.563 | 0.567 | 0.578 | -0.039 [-0.066, -0.008] |
| `high_Zt` | 0.536 | 0.554 | 0.558 | 0.557 | 0.557 | 0.562 | -0.054 [-0.081, -0.022] |
| `rwc_K` | 0.536 | 0.547 | 0.568 | 0.600 | 0.613 | 0.625 | +0.009 [-0.004, +0.023] |
| `rwc_V` | 0.536 | 0.543 | 0.565 | 0.601 | 0.612 | 0.629 | +0.013 [-0.003, +0.029] |
| `rwc_Z` | 0.536 | 0.554 | 0.577 | 0.607 | 0.618 | 0.631 | +0.015 [+0.001, +0.030] |
| `rwc_Zt` | 0.536 | 0.554 | 0.575 | 0.608 | 0.618 | 0.626 | +0.010 [-0.004, +0.025] |
| `sc` | 0.536 | 0.536 | 0.577 | 0.598 | 0.608 | 0.616 | — |
| `shortest` | 0.536 | 0.550 | 0.551 | 0.546 | 0.543 | 0.538 | -0.078 [-0.109, -0.048] |
| Oracle coverage | 0.536 | 0.660 | 0.759 | 0.827 | 0.871 | 0.895 | |
| Distinct groups (incl. NO_ANSWER) | 1.000 | 1.417 | 1.919 | 2.506 | 3.179 | 3.936 | |
| Mean token cost (all N candidates) | 278 | 551 | 1112 | 2230 | 4462 | 8898 | |
| Failures: no correct candidate (SC) | 0.464 | 0.340 | 0.241 | 0.173 | 0.129 | 0.105 | |
| Failures: correct candidate missed (SC) | 0.000 | 0.124 | 0.181 | 0.229 | 0.263 | 0.279 | |

MathVista adaptive stopping (settings chosen on the 200 dev questions; fixed-32 SC dev accuracy 0.6425):

| Policy | τ | γ | Dev accuracy | Held-out accuracy | − fixed-32 SC [95% CI] | Mean tokens | Mean rollouts | Stops at 4 / 8 / 16 / 32 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| agreement | 0.5 | — | 0.6360 | 0.6125 | -0.0037 [-0.0140, +0.0076] | 3290 | 9.89 | 0.714 / 0.075 / 0.021 / 0.191 |
| quality | 0.5 | 0.0 | 0.6395 | 0.6155 | -0.0007 [-0.0096, +0.0085] | 3613 | 10.93 | 0.654 / 0.096 / 0.029 / 0.221 |
| fixed-32 SC | — | — | 0.6425 | 0.6162 | — | 8898 | 32 | 0 / 0 / 0 / 1 |

### Signal analysis at N = 32 (`python task3/signals.py`; `task3/results/signals/`)

On the 800 held-out questions; `short` = negative length (the length-only control), so its ρ(length) is −1 by construction. Within-question correlations are averaged across questions; correctness correlations use the mixed questions only (MATH 469, MathVista 634).

| Score | MATH: ρ(length) | MATH: ρ(correct) | MATH: top − bottom quartile correctness | MathVista: ρ(length) | MathVista: ρ(correct) | MathVista: top − bottom quartile |
|---|---:|---:|---:|---:|---:|---:|
| `L` | +0.225 | +0.134 | +0.078 | +0.329 | +0.029 | +0.016 |
| `K` | -0.161 | +0.198 | +0.120 | +0.349 | +0.048 | +0.035 |
| `short` | -1.000 | +0.076 | +0.046 | -1.000 | +0.050 | +0.041 |
| `V` | — | — | — | -0.686 | +0.041 | +0.038 |
| `Z` | — | — | — | -0.275 | +0.089 | +0.074 |
| `Zt` | — | — | — | -0.322 | +0.080 | +0.072 |

- **MathVista confidence–vision correlation** (Pearson of c_t and v_t within each rollout; 25,572 rollouts defined, 28 too short or constant): the mean of question means is −0.211 (median −0.212). 10.6% of questions are positive; the 5–95% range of question means is −0.464 to +0.056.
- **Tokenwise fusion:** Zt − Z equals Cov_t(c, v) to within 1.8e-8. The covariance is negative in 85.2% of rollouts (median −0.0018).
- **Illustrative trajectories:** `mathvista_trajectories.json` holds four questions picked by the fixed rule in `task3/signals.py` (the first held-out question in each category: SC and top-VisConf both right, both wrong, and each one alone right). For each, the highest- and lowest-VisConf rollouts with their c_t and v_t series.

### Early stops and examples (`python task3/examples.py`; `task3/results/examples/<dataset>.json`)

Held-out questions × 10 orders (8,000 runs per policy), frozen settings. *Recovered* = a wrong early stop where fixed-32 SC is correct; *lost* = a correct early stop where fixed-32 SC is wrong.

| Dataset | Policy | Early-stop share | Early-stop accuracy | Accuracy when run to 32 | Wrong early stops (share of runs) | Recovered by fixed-32 | Lost by fixed-32 |
|---|---|---:|---:|---:|---|---:|---:|
| MATH | agreement | 0.735 | 0.888 | 0.317 | 660 (0.083) | 51 | 39 |
| MATH | quality | 0.835 | 0.828 | 0.246 | 1152 (0.144) | 131 | 82 |
| MathVista | agreement | 0.809 | 0.732 | 0.106 | 1736 (0.217) | 311 | 281 |
| MathVista | quality | 0.779 | 0.751 | 0.137 | 1548 (0.194) | 251 | 245 |

The quality-aware policy, binned by the full pool's fraction of correct rollouts (retrospective only). Each cell: questions / mean rollouts / early-stop share / accuracy.

| Pool fraction correct | MATH | MathVista |
|---|---|---|
| 0 | 69 / 18.5 / 0.50 / 0.000 | 84 / 23.9 / 0.30 / 0.000 |
| (0, .25] | 142 / 19.4 / 0.47 / 0.150 | 156 / 18.0 / 0.52 / 0.064 |
| (.25, .5] | 78 / 12.4 / 0.74 / 0.724 | 135 / 13.0 / 0.72 / 0.500 |
| (.5, .75] | 74 / 5.7 / 0.98 / 0.962 | 133 / 7.0 / 0.95 / 0.925 |
| (.75, 1) | 175 / 4.2 / 1.00 / 0.997 | 210 / 4.3 / 1.00 / 1.000 |
| 1 | 262 / 4.0 / 1.00 / 1.000 | 82 / 4.0 / 1.00 / 1.000 |

The examples were picked by the rule stated in `task3/examples.py` and saved with it: the first two held-out questions (permuted order, rollout order 0) of each kind, for selection disagreement (SC vs the dataset's RWC selector at N = 32) and for premature stopping (quality-aware). Revision examples come from the live run.

### Live generation (Kaggle `atml-pa3-task3-live`, Tesla T4, finished 2026-11-07; `task3/live/`, scored by `python task3/live_score.py` → `task3/results/live/`)

`Qwen/Qwen2.5-3B-Instruct` at revision `aa8e7253…`, FP16, batch size 1, 50 held-out MATH questions, 250 calls (5 per question), every call independently seeded (T3-12). Model loading took 39.8 s and is excluded. Peak GPU memory was 6.17 GB allocated and 6.91 GB reserved. Raw stdout: `task3/results/kaggle_stdout_live.log`.

| Strategy | Cap per response | Accuracy | Mean input tokens | Mean output tokens | Mean seconds | Responses capped | Responses with a boxed answer | Final answer extracted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | 3,072 | 0.62 | 108.4 | 761.3 | 41.0 | 2 / 50 (0.04) | 0.96 | 0.96 |
| Parallel (3, self-consistency) | 1,024 each | 0.54 | 325.1 | 1,859.5 | 99.4 | 25 / 150 (0.167) | 0.833 | 0.78 |
| Refinement (initial + 1 revision) | 1,024 + 2,048 | 0.62 | 889.6 | 1,218.7 | 65.6 | 10 / 100 (0.10) | 0.90 | 1.00 |

- Refinement is charged the shared initial response (parallel response 1) plus the revision call. The revision call alone averages 781.2 input and 583.1 output tokens.
- Capped responses by slot: single 2, parallel responses 1 / 2 / 3: 10 / 8 / 7, revision 0. Every capped response lacks a boxed answer, and every response without a boxed answer is capped.
- Accuracy of the shared initial answer: 0.56.
- **Paired bootstrap differences** (1,000 question resamples, seed 6304): parallel − single −0.08 [−0.20, +0.06]; refinement − single 0.00 [−0.12, +0.12]; refinement − parallel +0.08 [0.00, +0.18].

**Revision transitions** (relative to the initial answer):

| Initial response | Revision changed the answer | Initial → revised | Questions |
|---|---|---|---:|
| boxed answer | no | correct → correct | 28 |
| boxed answer | no | wrong → wrong | 12 |
| hit the 1,024 cap (no answer) | yes | wrong → correct | 3 |
| hit the 1,024 cap (no answer) | yes | wrong → wrong | 7 |

Totals: wrong → correct 3, correct → wrong 0, correct → correct 28, wrong → wrong 19. The revision kept the answer in all 40 questions whose initial response gave one. All 10 changes came from initial responses that hit the cap.

**Revision examples** (`python task3/live_examples.py` → `task3/results/examples/math_live_revision.json`). Rule, stated in the script and saved with the examples: walk the 50 questions in live order and take the first two of each transition kind. Correct → wrong has no example, because no question made that transition. Picked: wrong → correct `test:3504`, `test:2732`; wrong → wrong `test:2688`, `test:2460`; correct → correct `test:943`, `test:4335`.

### Logging requirements

For the live-inference strategies on the 50 questions, log:
- Actual input and output tokens.
- Runtime, excluding model loading.
- Peak GPU memory.
- Truncation and extraction rates.
- Revision transitions.

Selection and stopping rules must not use gold answers or unrevealed candidates.
