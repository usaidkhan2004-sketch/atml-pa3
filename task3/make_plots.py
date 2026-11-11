"""Task 3 figures from the saved results (no model access).

  task3/results/plots/sampling_selection.png   accuracy of the main selectors and oracle coverage vs N
  task3/results/plots/accuracy_cost.png        accuracy vs mean generated-token cost: fixed-N
                                               self-consistency and RWC, adaptive policies, fixed-32

Every number comes from task3/results/offline/<dataset>/{summary.json, adaptive.json} (800 held-out
questions, averaged over the ten orders). Usage: python task3/make_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "task3/results"
OUT = RES / "plots"
BUDGETS = [1, 2, 4, 8, 16, 32]
DATASETS = [("math", "MATH (Qwen2.5-3B)"), ("mathvista", "MathVista (Qwen2.5-VL-3B)")]
# color follows the method in every panel (Task 1 palette hues, fixed order)
METHOD = {"first": ("FirstAns", "#8a8a85", "-"), "sc": ("Self-consistency", "#2a78d6", "-"),
          "high_K": ("Highest self-certainty", "#eb6834", "--"), "rwc_K": ("Self-certainty + RWC", "#eb6834", "-"),
          "high_Z": ("Highest VisConf", "#1baf7a", "--"), "rwc_Z": ("VisConf + RWC", "#1baf7a", "-")}
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e6e5e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})


def load(name):
    s = json.loads((RES / f"offline/{name}/summary.json").read_text())["by_budget"]
    a = json.loads((RES / f"offline/{name}/adaptive.json").read_text())
    return s, a


def sampling_selection():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (name, title) in zip(axes, DATASETS):
        s, _ = load(name)
        ax.plot(BUDGETS, [s[str(n)]["sc"]["oracle_coverage"] for n in BUDGETS], color=INK, lw=1.5, ls=":",
                marker="o", ms=4, label="Oracle coverage")
        for m, (label, color, ls) in METHOD.items():
            if m not in s["32"]:
                continue
            ax.plot(BUDGETS, [s[str(n)][m]["accuracy"] for n in BUDGETS], color=color, lw=2, ls=ls, marker="o",
                    ms=4, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(n) for n in BUDGETS])
        ax.set_xlabel("Candidate budget N (rollouts)")
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_ylabel("Accuracy (800 held-out questions)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Accuracy and oracle coverage vs candidate budget, averaged over 10 rollout orders. "
                 "The gap between oracle coverage and a selector is a selection failure.",
                 x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(OUT / "sampling_selection.png")
    plt.close(fig)


def accuracy_cost():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (name, title) in zip(axes, DATASETS):
        s, a = load(name)
        rwc = "rwc_K" if name == "math" else "rwc_Z"
        for m in ("sc", rwc):
            label, color, ls = METHOD[m]
            x = [s[str(n)][m]["mean_cost_tokens"] for n in BUDGETS]
            y = [s[str(n)][m]["accuracy"] for n in BUDGETS]
            ax.plot(x, y, color=color, lw=2, ls=ls, marker="o", ms=4, label=f"{label}, N = 1…32")
        for fam, marker, label in (("agreement", "D", "Adaptive: agreement"), ("quality", "s", "Adaptive: quality-aware")):
            e, sel = a["eval"][fam], a["selected"][fam]
            ax.plot(e["mean_cost_tokens"], e["accuracy"], marker=marker, ms=8, color="#e87ba4" if fam == "quality" else "#eda100",
                    markeredgecolor="white", markeredgewidth=1.5, ls="none",
                    label=f"{label} (τ={sel['tau']}{'' if fam == 'agreement' else f', γ={sel['gamma']}'})")
        ax.set_xscale("log")
        ax.set_xlabel("Mean generated tokens per question (log scale)")
        ax.set_title(title, loc="left", color=INK)
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("Accuracy (800 held-out questions)")
    fig.suptitle("Accuracy vs generated-token cost (offline proxy: retained tokens of consumed rollouts). "
                 "Adaptive settings were chosen on the 200 development questions.",
                 x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "accuracy_cost.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sampling_selection()
    accuracy_cost()
