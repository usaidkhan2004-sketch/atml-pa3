"""Task 1 figures from the saved results (no GPU, no model access).

  task1/results/plots/sft_loss.png      per-update SFT loss of the five runs (manual: combined SFT-loss plot)
  task1/results/plots/benchmarks.png    primary score of each benchmark for the seven conditions
  task1/results/plots/diagnostics.png   rho_k, eta and mean R(z) (math vs non-math) per down_proj layer

Every number comes from task1/results/{<run>/train_log.jsonl, eval/<condition>/summary.json,
diagnostics/diagnostics.json}. Usage: python make_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "task1/results"
OUT = RES / "plots"

RUNS = ["fullft", "l2sp", "lora4", "lora16", "oplora"]
RUN_LABEL = {"fullft": "Full FT", "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16",
             "oplora": "OPLoRA-16"}
COND = ["baseline", "fullft_0625", "fullft_1250", "l2sp_1250", "lora4_1250", "lora16_1250", "oplora_1250"]
COND_LABEL = {"baseline": "Baseline", "fullft_0625": "Full FT @625", "fullft_1250": "Full FT",
              "l2sp_1250": "Full FT + L2-SP", "lora4_1250": "LoRA-4", "lora16_1250": "LoRA-16",
              "oplora_1250": "OPLoRA-16"}
# Color follows the run in every figure; the Full FT midpoint is a lighter step of Full FT's hue.
RUN_COLOR = {"fullft": "#2a78d6", "l2sp": "#eb6834", "lora4": "#1baf7a", "lora16": "#eda100", "oplora": "#e87ba4"}
COND_COLOR = {"baseline": "#8a8a85", "fullft_0625": "#86b6ef", "fullft_1250": RUN_COLOR["fullft"],
              "l2sp_1250": RUN_COLOR["l2sp"], "lora4_1250": RUN_COLOR["lora4"],
              "lora16_1250": RUN_COLOR["lora16"], "oplora_1250": RUN_COLOR["oplora"]}
BENCH = [("gsm8k_cot", "exact_match,flexible-extract", "GSM8K (flexible-extract EM)"),
         ("hellaswag", "acc_norm,none", "HellaSwag (normalized acc.)"),
         ("boolq", "acc,none", "BoolQ (acc.)"),
         ("arc_easy", "acc_norm,none", "ARC-Easy (normalized acc.)")]
# L2-SP and OPLoRA lie almost on top of Full FT and LoRA-16, so they are also dashed.
RUN_STYLE = {"fullft": "-", "l2sp": "--", "lora4": "-", "lora16": "-", "oplora": "--"}
SMOOTH = 25  # updates in the centered moving average drawn over the raw per-update loss
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e6e5e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})


def moving_average(y, w):
    y = np.asarray(y, float)
    k = np.ones(w) / w
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")  # shorter window at the ends, no zero padding
    return num / den


def sft_loss():
    logs = {r: [json.loads(l) for l in open(RES / r / "train_log.jsonl")] for r in RUNS}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, start in zip(axes, [1, 100]):
        for r in RUNS:
            u = np.array([x["update"] for x in logs[r]])
            y = np.array([x["sft_loss"] for x in logs[r]])
            m = u >= start
            ax.plot(u[m], y[m], color=RUN_COLOR[r], lw=0.6, alpha=0.25)
            ax.plot(u[m], moving_average(y, SMOOTH)[m], color=RUN_COLOR[r], lw=2, ls=RUN_STYLE[r],
                    label=RUN_LABEL[r])
        ax.set_xlabel("Optimizer update")
    axes[0].set_ylabel("SFT loss (mean CE per supervised token)")
    axes[0].set_title("All updates", loc="left", color=INK)
    axes[1].set_title(f"Updates 100–1,250 (same data, rescaled y)", loc="left", color=INK)
    axes[1].legend(loc="upper right", ncol=1)
    fig.suptitle(f"SFT loss per update; faint = raw, bold = {SMOOTH}-update centered moving average. "
                 "L2-SP shown without its penalty term.", x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sft_loss.png")
    plt.close(fig)


def benchmarks():
    S = {c: json.loads((RES / "eval" / c / "summary.json").read_text())["scores"] for c in COND}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=False)
    x = np.arange(len(COND))
    for ax, (task, metric, title) in zip(axes, BENCH):
        v = np.array([100 * S[c][task][metric] for c in COND])
        se_key = metric.replace(",", "_stderr,")
        se = np.array([100 * S[c][task][se_key] for c in COND])
        ax.bar(x, v, width=0.8, color=[COND_COLOR[c] for c in COND], edgecolor="white", linewidth=2)
        ax.errorbar(x, v, yerr=se, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.axhline(v[0], color=MUTED, lw=0.8, ls="--")
        lo = min(v - se) - 4
        ax.set_ylim(max(0, lo), max(v + se) + 3)
        ax.set_xticks(x, [COND_LABEL[c] for c in COND], rotation=55, ha="right")
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score (%)")
    fig.suptitle("Primary score per benchmark (± 1 standard error); dashed line = baseline. "
                 "Y-axes do not start at zero.", x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "benchmarks.png")
    plt.close(fig)


def diagnostics():
    d = json.loads((RES / "diagnostics/diagnostics.json").read_text())
    ckpts, layers = d["checkpoints"], d["layers"]
    W = {(r["condition"], r["layer"]): r for r in d["weight"]}
    A = {(r["condition"], r["layer"], r["group"]): r for r in d["activation_summary"]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    width = 0.8 / len(ckpts)
    x = np.arange(len(layers))
    for j, c in enumerate(ckpts):
        off = x - 0.4 + width * (j + 0.5)
        axes[0].bar(off, [W[c, l]["rho_k"] for l in layers], width, color=COND_COLOR[c],
                    edgecolor="white", linewidth=1, label=COND_LABEL[c])
        axes[1].bar(off, [W[c, l]["eta"] for l in layers], width, color=COND_COLOR[c],
                    edgecolor="white", linewidth=1)
        axes[2].scatter(off, [A[c, l, "math"]["R_mean"] for l in layers], s=36, color=COND_COLOR[c],
                        edgecolor="white", linewidth=1, zorder=3)
        axes[2].scatter(off, [A[c, l, "nonmath"]["R_mean"] for l in layers], s=36, facecolor="white",
                        edgecolor=COND_COLOR[c], linewidth=1.5, zorder=3)
    k, m = d["protect_k"], d["starting_weights"]["0"]["shape"][0]
    axes[0].axhline(k / m, color=MUTED, lw=0.8, ls="--")
    axes[0].annotate(f"k/m = {k}/{m}", (len(layers) - 0.55, k / m), textcoords="offset points",
                     xytext=(0, 3), ha="right", fontsize=7.5, color=MUTED)
    titles = [f"Subspace interference ρ_k (k = {k})", "Relative update magnitude η",
              "Mean R(z): filled = math, hollow = non-math"]
    for ax, t in zip(axes, titles):
        ax.set_xticks(x, [f"Layer {l}" for l in layers])
        ax.set_title(t, loc="left", color=INK, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].annotate("OPLoRA: ρ_k = 0", (0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
                     fontsize=7.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=8)
    fig.suptitle("down_proj diagnostics, FP32. ρ_k = ||U_k U_kᵀ ΔW||²_F / ||ΔW||²_F; η = ||ΔW||_F / ||W0||_F; "
                 "R(z) = ||ΔW z|| / (||W0 z|| + 1e-8) over 32 + 32 probe prompts.",
                 x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.savefig(OUT / "diagnostics.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sft_loss()
    benchmarks()
    diagnostics()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))} to {OUT}")


if __name__ == "__main__":
    main()
