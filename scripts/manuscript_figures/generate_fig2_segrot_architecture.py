#!/usr/bin/env python3
"""Generate a publication-style architecture diagram for the SegRot module."""

from pathlib import Path
import os
import textwrap

OUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "Manuscript"
PDF_PATH = MANUSCRIPT_DIR / "fig2_segrot_architecture.pdf"
PNG_PATH = MANUSCRIPT_DIR / "fig2_segrot_architecture.png"


COLORS = {
    "input": ("#f8fafc", "#475569"),
    "encoder": ("#e8f1ff", "#2563eb"),
    "feature": ("#eef2ff", "#4f46e5"),
    "seg": ("#ecfdf5", "#059669"),
    "orient": ("#fff7ed", "#ea580c"),
    "output": ("#fefce8", "#ca8a04"),
    "norm": ("#f5f3ff", "#7c3aed"),
    "arrow": "#334155",
    "muted": "#64748b",
}


def wrap_label(text, width=24):
    lines = []
    for raw in text.split("\n"):
        lines.extend(textwrap.wrap(raw, width=width) or [""])
    return "\n".join(lines)


def add_box(ax, x, y, w, h, label, kind, fontsize=9.0, weight="normal", width=24):
    face, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.18,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        wrap_label(label, width),
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111827",
        linespacing=1.15,
        zorder=3,
    )
    return patch


def add_arrow(ax, start, end, label=None, rad=0.0, color=None, dashed=False, label_offset=0.06):
    color = color or COLORS["arrow"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.08,
        linestyle=(0, (4, 3)) if dashed else "solid",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + label_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=7.1,
            color=color,
            zorder=4,
        )


def add_header(ax):
    ax.text(
        0.42,
        4.42,
        "Detailed view of the multi-head U-Net block in Figure 1",
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color="#374151",
    )
    ax.text(
        0.42,
        4.16,
        "A shared encoder feeds two heads whose outputs are used by the subsequent geometric-normalization stage.",
        ha="left",
        va="top",
        fontsize=8.0,
        color=COLORS["muted"],
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(12.8, 4.7))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    add_header(ax)

    # Main shared trunk.
    add_box(ax, 0.42, 2.14, 1.25, 0.72, "Input image", "input", fontsize=8.8)
    add_box(
        ax,
        2.18,
        2.02,
        1.78,
        0.96,
        "Shared convolutional encoder\nMobileNetV3-Large",
        "encoder",
        fontsize=8.4,
        width=22,
    )
    add_box(ax, 4.48, 2.08, 1.55, 0.84, "Shared feature map", "feature", fontsize=8.3)

    # Task heads.
    add_box(
        ax,
        6.38,
        2.92,
        1.72,
        0.76,
        "Segmentation decoder\nupsampling path",
        "seg",
        fontsize=8.2,
        width=21,
    )
    add_box(
        ax,
        6.38,
        1.30,
        1.72,
        0.76,
        "Orientation classifier\nglobal pooling + linear head",
        "orient",
        fontsize=8.0,
        width=23,
    )

    # Module outputs.
    add_box(ax, 8.88, 2.92, 1.42, 0.76, "Binary document mask", "output", fontsize=7.9, width=19)
    add_box(
        ax,
        8.88,
        1.30,
        1.42,
        0.76,
        "Orientation class\n72 classes\n(5 deg bins)",
        "output",
        fontsize=7.2,
        width=19,
    )
    add_box(
        ax,
        10.76,
        2.08,
        1.44,
        0.84,
        "Inputs for geometric normalization",
        "norm",
        fontsize=7.2,
        width=17,
    )

    # Main arrows.
    add_arrow(ax, (1.67, 2.50), (2.18, 2.50))
    add_arrow(ax, (3.96, 2.50), (4.48, 2.50))
    add_arrow(ax, (6.03, 2.62), (6.38, 3.28), rad=0.08)
    add_arrow(ax, (6.03, 2.38), (6.38, 1.68), rad=-0.08)
    add_arrow(ax, (8.10, 3.30), (8.88, 3.30))
    add_arrow(ax, (8.10, 1.68), (8.88, 1.68))
    add_arrow(ax, (10.30, 3.12), (10.76, 2.70), rad=-0.05)
    add_arrow(ax, (10.30, 1.86), (10.76, 2.28), rad=0.05)

    # Evidence-safe note.
    ax.text(
        2.18,
        1.52,
        "Other evaluated backbones: ResNet50 and EfficientNet-B0",
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    ax.text(
        11.48,
        1.72,
        "next block in\nFigure 1",
        ha="center",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
        linespacing=1.12,
    )

    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    print(PDF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
