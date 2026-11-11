"""Task 2 figures from the saved results (no GPU, no model access).

  task2/results/plots/learning_curves.png   training loss, development caption loss, answer NLL under
                                            correct / complementary / no-image inputs, and the
                                            image-swap penalty, with stage boundaries marked
                                            (manual, Required Evidence: "Learning curves")

Every number comes from task2/results/train/{train_log.jsonl, dev_eval.jsonl}.
Usage: python task2/make_plots.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "task2/results"
OUT = RES / "plots"

STAGE_ENDS = [2250, 3750, 4500]
SUITES = [("vqa", "VQAv2"), ("grounding", "Grounding")]
# color follows the image input in every panel (Task 1 palette hues)
COND = [("nll_correct", "Correct image", "#2a78d6", "-"), ("nll_complementary", "Complementary image", "#eb6834", "-"),
        ("nll_no_image", "No image", "#8a8a85", "--")]
SUITE_COLOR = {"vqa": "#2a78d6", "grounding": "#1baf7a"}
SMOOTH = 50
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
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")


def stage_lines(ax, label=False):
    for e in STAGE_ENDS[:-1]:
        ax.axvline(e, color=MUTED, lw=0.8, ls=":")
    if label:
        lo = 0
        for k, e in enumerate(STAGE_ENDS):
            ax.text((lo + e) / 2, 0.98, f"S{k + 1}", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8, color=MUTED)
            lo = e


def learning_curves():
    train = [json.loads(l) for l in open(RES / "train/train_log.jsonl")]
    dev = [json.loads(l) for l in open(RES / "train/dev_eval.jsonl")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))

    ax = axes[0, 0]
    kept = [r for r in train if not r["skipped"] and math.isfinite(r["loss"])]
    u = np.array([r["update"] for r in kept])
    y = np.array([r["loss"] for r in kept])
    ax.plot(u, y, color="#2a78d6", lw=0.5, alpha=0.2)
    ax.plot(u, moving_average(y, SMOOTH), color="#2a78d6", lw=2)
    ax.set_title(f"Training loss ({SMOOTH}-update mean)", loc="left", color=INK, fontsize=9)
    ax.set_ylabel("Example-weighted CE")
    stage_lines(ax)

    ax = axes[0, 1]
    du = [r["update"] for r in dev]
    ax.plot(du, [r["caption_loss"] for r in dev], color="#2a78d6", lw=2, marker="o", ms=4)
    ax.set_title("Dev caption reference loss (96 images)", loc="left", color=INK, fontsize=9)
    ax.set_ylabel("Mean CE (caption + end token)")
    stage_lines(ax)

    ax = axes[0, 2]
    for s, name in SUITES:
        ax.plot(du, [r[s]["delta_nll"] for r in dev], color=SUITE_COLOR[s], lw=2, marker="o", ms=4, label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("Dev image-swap penalty ΔNLL", loc="left", color=INK, fontsize=9)
    ax.set_ylabel("ΔNLL (nats per token)")
    ax.legend(loc="center left")
    stage_lines(ax)

    for ax, (s, name) in zip(axes[1, :2], SUITES):
        for key, label, color, ls in COND:
            ax.plot(du, [r[s][key] for r in dev], color=color, lw=2, ls=ls, marker="o", ms=4, label=label)
        n = dev[0][s]["n"]
        ax.set_title(f"{name} dev answer NLL ({n} questions)", loc="left", color=INK, fontsize=9)
        ax.set_ylabel("Mean token NLL of the reference")
        stage_lines(ax)

    ax = axes[1, 2]
    final = dev[-1]
    types = [("vqa", "yes/no"), ("vqa", "number"), ("vqa", "other"), ("grounding", "presence"), ("grounding", "counting")]
    x = np.arange(len(types))
    for k, (key, label, color, _) in enumerate(COND):
        ax.bar(x + (k - 1) * 0.27, [final[s]["by_type"][t][key] for s, t in types], width=0.25, color=color,
               label=label, edgecolor="white", linewidth=1)
    ax.set_xticks(x, [f"{'VQA' if s == 'vqa' else 'Gr.'} {t}" for s, t in types], rotation=20)
    ax.set_title("Dev answer NLL by question type at update 4,500", loc="left", color=INK, fontsize=9)
    ax.set_ylabel("Mean token NLL of the reference")
    ax.set_ylim(0, max(final[s]["by_type"][t][k] for s, t in types for k, *_ in COND) * 1.3)
    ax.legend(loc="upper left", ncol=3, fontsize=7.5, handlelength=1.2, columnspacing=0.8)
    ax.grid(axis="x", visible=False)

    for ax in list(axes[0]) + list(axes[1, :2]):
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(0, 4600)
    for ax in list(axes[0]) + list(axes[1, :2]):
        stage_lines(ax, label=True)
    fig.suptitle("Task 2 learning curves. S1–S3: stages (dotted boundaries at updates 2,250 and 3,750); faint line: "
                 "per-update loss.\nΔNLL = NLL(partner image) − NLL(correct image). Dev evaluation at 0, every 500, "
                 "and 2,250 / 3,750 / 4,500.\nBottom row colors: blue correct image, orange complementary image, gray dashed no image.",
                 x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "learning_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    learning_curves()
