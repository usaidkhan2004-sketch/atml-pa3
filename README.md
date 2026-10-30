# ATML PA3 — Fine-Tuning, Vision-Language Models, and Test-Time Compute

EE-5102 / CS-6304, Fall 2026. Seed `6304` is used throughout.

## Layout

```
shared/   common utilities (seeding, I/O, plotting helpers)
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

## Reproducing

_To be filled in per task._

## Attribution

_List materially reused external code here._
