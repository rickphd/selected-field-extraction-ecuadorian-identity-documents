from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Manuscript"

models = ["MobileNet\nV3-Large", "ResNet50", "EfficientNet\nB0"]
x = np.arange(len(models))

train_f1 = np.array([0.7187, 0.7246, 0.7175])
val_f1 = np.array([0.9299, 0.9043, 0.9400])
train_iou = np.array([0.9167, 0.9252, 0.9144])
val_iou = np.array([0.9094, 0.9117, 0.9092])
epochs = np.array([39, 47, 50])
training_time = np.array([0.84, 1.82, 1.42])

colors = {
    "red": "#991b1b",
    "red_fill": "#f87171",
    "teal": "#0f766e",
    "teal_fill": "#5ecdc7",
    "dark": "#111827",
    "orange": "#f59e0b",
    "magenta": "#e11d48",
    "cyan": "#0891b2",
    "grid": "#e5e7eb",
    "text": "#111827",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": "#111827",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    }
)

fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.25), constrained_layout=True)
bar_w = 0.36
panel_labels = ["(a)", "(b)", "(c)"]

for ax, panel_label in zip(axes, panel_labels):
    ax.set_axisbelow(True)
    ax.grid(True, color=colors["grid"], linewidth=0.7, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, color=colors["text"])
    ax.text(
        0.5,
        -0.30,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=colors["text"],
    )

ax = axes[0]
b1 = ax.bar(
    x - bar_w / 2,
    train_f1,
    bar_w,
    label="Train F1",
    color=colors["red_fill"],
    edgecolor=colors["red"],
    linewidth=1.0,
    alpha=0.75,
)
b2 = ax.bar(
    x + bar_w / 2,
    val_f1,
    bar_w,
    label="Val F1",
    color=colors["teal_fill"],
    edgecolor=colors["teal"],
    linewidth=1.0,
    alpha=0.75,
)
ax.set_ylabel("F1 score")
ax.set_xticks(x, models)
ax.set_ylim(0.68, 0.965)
legend = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=True,
    edgecolor="black",
    handlelength=1.8,
)
legend.get_frame().set_linewidth(0.5)
for rect, value in zip(b2, val_f1):
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        value + 0.006,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=colors["text"],
    )

ax = axes[1]
b1 = ax.bar(
    x - bar_w / 2,
    train_iou,
    bar_w,
    label="Train IoU",
    color=colors["red_fill"],
    edgecolor=colors["red"],
    linewidth=1.0,
    alpha=0.75,
)
b2 = ax.bar(
    x + bar_w / 2,
    val_iou,
    bar_w,
    label="Val IoU",
    color=colors["teal_fill"],
    edgecolor=colors["teal"],
    linewidth=1.0,
    alpha=0.75,
)
ax.set_ylabel("Mean IoU")
ax.set_xticks(x, models)
ax.set_ylim(0.88, 0.94)
legend = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=True,
    edgecolor="black",
    handlelength=1.8,
)
legend.get_frame().set_linewidth(0.5)
for rect, value in zip(b2, val_iou):
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        value + 0.0016,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=colors["text"],
    )

ax = axes[2]
point_colors = [colors["magenta"], colors["cyan"], colors["orange"]]
labels = ["MobileNetV3\n39 ep.", "ResNet50\n47 ep.", "EfficientNet\n50 ep."]
ax.scatter(
    training_time,
    val_f1,
    s=82,
    c=point_colors,
    edgecolors="white",
    linewidths=1.4,
    zorder=5,
)
offsets = [(0.035, 0.003), (-0.06, 0.003), (0.035, 0.003)]
alignments = ["left", "right", "left"]
for tx, ty, label, (dx, dy), ha in zip(training_time, val_f1, labels, offsets, alignments):
    ax.text(
        tx + dx,
        ty + dy,
        label,
        ha=ha,
        va="center",
        fontsize=8.5,
        color=colors["text"],
        zorder=6,
    )
ax.set_xlabel("Training time (h)")
ax.set_ylabel("Val F1")
ax.set_xlim(0.65, 2.02)
ax.set_ylim(0.89, 0.955)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig5_seg_rot_comparison.{ext}", bbox_inches="tight")
