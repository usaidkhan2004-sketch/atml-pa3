# Implementation decisions

The manual fixes most settings. This file records every point it leaves open and the choice made, so the report can state deviations and interpretations.

## Task 1 — LLM fine-tuning

| # | Point left open by the manual | Choice | Rationale |
|---|---|---|---|
| 1 | Source of `first_plot_questions.yaml` | `emergent-misalignment/emergent-misalignment`, `evaluation/first_plot_questions.yaml` (main branch, SHA-256 `215abde5…387173`), saved to `task1/configs/` | Only public file of that name; 8 questions × plain/JSON/template = 24, as the manual describes |
| 3 | Recovery checkpoints vs. Kaggle's 20 GB output cap (~7 GB each for Full FT) | Rolling recovery checkpoint: each of updates 312/625/937/1,250 is saved, and the previous one is removed once the next is written. Model weights at 625 (Full FT) and 1,250 (all runs) are kept permanently | Keeps the checkpoints' stated purpose ("for recovery only") within the platform limit |
| 4 | GSM8K overlap "text normalization" | Unicode NFKC → lowercase → collapse whitespace → strip; drop if normalized `query` **or** `original_question` equals any normalized GSM8K test question | Most literal exact-match reading |
| 5 | "Formatted length" | Token count of the full Qwen3 chat-formatted pair (user + assistant + `<|im_end|>`), `enable_thinking=False` | Matches the training sequence actually fed to the model |
| 6 | Empty `<think>\n\n</think>\n\n` block inserted by the template | Masked as prompt; loss on response text + `<|im_end|>` | Manual: loss "only on assistant-response tokens and the end-of-turn token" |
| 7 | Cosine end value | Decays to 0 after 38 linear warmup updates | Task 2 states an explicit floor; Task 1 does not |
| 8 | L2-SP application | λΩ added once per optimizer update: its gradient λ(θ−θ₀) is added to the unscaled SFT gradient before clipping (same gradient as L_SFT + λΩ, kept out of FP16 loss scaling). θ₀ is a frozen FP32 copy in pinned **CPU** memory, moved to the GPU parameter by parameter each update; θ₀ on the GPU ran out of memory on the T4 (`task1/results/benchmark_t4.json`). AdamW weight decay 0.01 still applied | Penalty belongs to the per-update objective; manual separates the two references and asks for reference-storage and transfer overhead |
| 9 | LoRA "random A" distribution | Kaiming-uniform (a=√5), PEFT's default; one custom LoRA module shared by LoRA and OPLoRA so LoRA-16 and OPLoRA start from identical A, B; dropout on adapter input | Manual requires "same … initial unprojected factors" |
| 10 | GPU | One Kaggle T4 (16 GB) for every run | Comparable time/memory; FP16 tensor cores |
| 11 | Probe-set sampling | `numpy.random.default_rng(6304).choice(..., replace=False)`, 32 GSM8K test + 32 HellaSwag validation indices, saved to `task1/splits/` before any model run | Seeded, reproducible, fixed before results |
| 12 | Warmup shape | `transformers.get_cosine_schedule_with_warmup(38, 1250)`: linear from 0 over updates 1–38, peak at 39, cosine to 0 at 1,250; the scheduler advances on every update, including any skipped for FP16 overflow (skips are logged) | Standard implementation of "cosine decay; 38 warmup updates" |
| 13 | Memory-only implementation details | lm_head and cross-entropy run only on supervised positions, in checkpointed chunks of 1,024; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Change peak memory, not the loss or gradients |
| 14 | lm-evaluation-harness version | `lm_eval==0.4.13` (latest stable release); stock task YAMLs `gsm8k_cot` (its default 8 shots overridden to `num_fewshot=0`), `hellaswag`, `boolq`, `arc_easy`; HF backend (`HFLM`) | "Fixed task definitions": one pinned version for all seven conditions |
| 15 | How the GSM8K settings reach lm-eval | `apply_chat_template=True`, `enable_thinking=False` (HFLM argument → prompt ends with the empty think block, as in training), `gen_kwargs={"max_gen_toks": 1024}` (HFLM's default is 256); stock `do_sample: false` and stop sequences `Q:`, `</s>`, `<|im_end|>` | Manual's GSM8K settings; nothing else changed in the stock task |
| 16 | "Truncation flag" | A GSM8K generation is truncated when it reaches the 1,024-token cap without an end-of-sequence token or a stop sequence. A subclass of HFLM records the stop reason (`stop_sequence` / `eos` / `length`) from the raw generated tokens; "no extraction" (filter output `[invalid]`) is recorded separately | Separates truncation from extraction failure from wrong answers, as the manual asks |
| 17 | Evaluation batch sizes | 32 for GSM8K generation, 8 for likelihood tasks, the same for every condition | Batch composition can shift FP16 numerics slightly; fixed across conditions so comparisons are like for like |
| 18 | Benchmark dataset revisions | lm-eval loads each dataset's default branch; the commit at evaluation time is recorded per condition in `summary.json` | lm-eval task YAMLs do not pin revisions |
| 19 | Merging adapters | Fresh FP32 backbone copy, `W = W0 + (α/r)·B·A` for all 196 target projections, then one cast to FP16. OPLoRA uses `adapter_projected.safetensors`. Full FT weights (FP32 masters) are cast to FP16 | Manual: merge into fresh copies, projected factors for OPLoRA, evaluate in FP16 |
| 20 | Merged-vs-unmerged "small batch" | First 4 GSM8K (evaluation chat format) + first 4 HellaSwag probe-set examples. Unmerged = FP16 backbone + FP32 adapters under FP16 autocast (the training forward). Records max/mean absolute logit difference and top-1 agreement. For OPLoRA it also checks the exported projected factors against ones recomputed from raw factors and freshly computed bases, and records max \|U_kᵀB̃\| and \|ÃV_k\| | Fixed, results-independent inputs; covers math and non-math text |
| 21 | Two GPUs | Kaggle's 2×T4: GSM8K on one GPU and the likelihood tasks on the other, in separate processes | Halves wall time; does not change any result |
| 22 | Diagnostics: "benchmark prompt formats" and "last non-padding prompt token" | Math probes use the exact GSM8K prompt strings lm-eval produced in the baseline evaluation (chat template, empty think block); non-math probes use lm-eval's own HellaSwag `process_docs` context, checked against the evaluation's prompt hash (`task1/splits/probe_prompts.json`). Each prompt runs alone (batch of one, so no padding), and z is the `down_proj` input at its last token | The probe inputs are the ones the benchmarks actually scored |
| 23 | Diagnostics: starting weight and bases | W0 = the base checkpoint loaded in FP32, where Full FT starts (its FP16 round-trip difference is recorded per layer). U_k is computed as OPLoRA computes it: SVD of the FP16 backbone weight cast to FP32, on the GPU, k = 128. ΔW: W_trained − W0 for Full FT variants; (α/r)·B·A for adapters, using `adapter_projected.safetensors` for OPLoRA. TF32 is disabled | Manual: "using the same starting-weight bases", FP32, projected factors for OPLoRA |
| 24 | How R(z) is summarized | Per-prompt R(z) for every checkpoint × layer is saved. The summary gives the mean, median, std, min and max over the 32 math and 32 non-math prompts, for each layer and pooled over the three layers | The manual asks for a comparison "on math and non-math prompts" without naming a statistic, so the raw values are kept |

Note on #20: the merge check's four HellaSwag inputs are `activity_label: ctx` from the raw fields, not lm-eval's preprocessed context. That batch only compares two forms of the same model and is not a benchmark input. The diagnostics (#22) use the stock context.

### Pinned revisions (from `task1/results/data_prep_summary.json`)

| Repo | Revision |
|---|---|
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `meta-math/MetaMathQA` | `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18` |
| `openai/gsm8k` (main) | `740312add88f781978c0658806c59bc2815b9866` |
| `Rowan/hellaswag` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` |

Data prep result: source rows 0–40,001 scanned; 2 excluded (rows 26,704 and 30,481, 2,123 and 2,240 tokens); 0 empty; 0 GSM8K-test overlaps. The training sequence ends at `<|im_end|>`; the template's trailing newline is dropped (#5). Joint vs. split tokenization agreed on all 400 checked examples.

## Task 3

| # | Point | Status |
|---|---|---|
| 2 | Location of `ATML-PA3-Data` | **Unresolved**: not publicly findable; must come from LMS / course staff |
