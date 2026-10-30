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
| Annotation subset: first + last plain questions, 3 formats × 7 conditions = 42 | 168 responses | responses collected in `task1/results/qualitative/annotation_subset.md`; the annotation itself is the student's own work | responses ✅, annotation ☐ |

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

## Task 2 — VLM (to log during training, when built)

The manual asks for these, and they must be logged while training runs, not recomputed afterwards:
- Training loss, development caption loss, answer NLL and image-swap penalties, with stage boundaries marked.
- Training and generation time, peak GPU memory, and trainable parameters by stage. Feature-caching cost must be timed separately.

## Task 3 — Test-time compute (when ATML-PA3-Data is available)

For the live-inference strategies on the 50 questions, log:
- Actual input and output tokens.
- Runtime, excluding model loading.
- Peak GPU memory.
- Truncation and extraction rates.
- Revision transitions.

Selection and stopping rules must not use gold answers or unrevealed candidates.
