#!/usr/bin/env python3
"""Generate a publication-style OCR Transformer architecture figure."""

from pathlib import Path
import os
import textwrap

OUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "Manuscript"
PDF_PATH = MANUSCRIPT_DIR / "fig4_ocr_transformer_architecture.pdf"
PNG_PATH = MANUSCRIPT_DIR / "fig4_ocr_transformer_architecture.png"


COLORS = {
    "input": ("#f8fafc", "#475569"),
    "patch": ("#ecfdf5", "#059669"),
    "embed": ("#eef2ff", "#4f46e5"),
    "encoder": ("#e8f1ff", "#2563eb"),
    "decoder": ("#fff7ed", "#ea580c"),
    "logits": ("#f5f3ff", "#7c3aed"),
    "output": ("#fefce8", "#ca8a04"),
    "arrow": "#334155",
    "muted": "#64748b",
}


def wrap_label(text, width=22):
    lines = []
    for raw in text.split("\n"):
        lines.extend(textwrap.wrap(raw, width=width) or [""])
    return "\n".join(lines)


def add_box(ax, x, y, w, h, label, kind, fontsize=8.7, weight="normal", width=22):
    face, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.15,
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


def add_arrow(ax, start, end, label=None, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.05,
        color=COLORS["arrow"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.07, label, ha="center", va="bottom", fontsize=7.0, color=COLORS["muted"])


def draw_crop(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#f8fafc", edgecolor="#475569", linewidth=1.15, zorder=2))
    ax.text(
        x + w / 2,
        y + h / 2,
        "SAMPLE TEXT FIELD",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#111827",
        family="DejaVu Sans Mono",
        zorder=3,
    )
    ax.text(x + w / 2, y - 0.18, "input image", ha="center", va="top", fontsize=7.4, color=COLORS["muted"])


def draw_patch_grid(ax, x, y, w, h, rows=2, cols=8):
    add_box(ax, x, y, w, h, "", "patch")
    pad = 0.06
    gx, gy = x + pad, y + pad
    gw, gh = w - 2 * pad, h - 2 * pad
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (gx + c * gw / cols, gy + r * gh / rows),
                    gw / cols - 0.012,
                    gh / rows - 0.012,
                    facecolor="#ffffff",
                    edgecolor="#94a3b8",
                    linewidth=0.55,
                    zorder=3,
                )
            )
    ax.text(x + w / 2, y + h + 0.16, "image patches", ha="center", va="bottom", fontsize=8.2, color="#111827")


def draw_token_strip(ax, x, y, w, h, label, color="#4f46e5"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#ffffff", edgecolor=color, linewidth=0.95, zorder=3))
    n = 8
    pad = 0.045
    tw = (w - 2 * pad) / n
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (x + pad + i * tw, y + 0.08),
                tw * 0.72,
                h - 0.16,
                facecolor="#eef2ff",
                edgecolor=color,
                linewidth=0.55,
                zorder=4,
            )
        )
    ax.text(x + w / 2, y + h + 0.11, label, ha="center", va="bottom", fontsize=7.8, color="#111827")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(13.6, 4.65))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 4.65)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.42,
        4.40,
        "Transformer-based OCR module",
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color="#374151",
    )
    ax.text(
        0.42,
        4.15,
        "The cropped field image is split into patches, flattened into token embeddings, encoded, and decoded into text.",
        ha="left",
        va="top",
        fontsize=8.0,
        color=COLORS["muted"],
    )

    draw_crop(ax, 0.48, 2.15, 1.62, 0.58)
    draw_patch_grid(ax, 2.55, 1.96, 1.22, 0.90)
    add_box(ax, 4.20, 1.98, 1.14, 0.86, "Flatten + linear projection", "embed", fontsize=7.7, width=17)
    draw_token_strip(ax, 5.82, 2.11, 1.18, 0.58, "patch embeddings")
    add_box(ax, 5.82, 0.98, 1.18, 0.58, "position embedding", "embed", fontsize=7.5, width=17)
    add_box(ax, 7.48, 1.78, 1.42, 1.20, "Transformer encoder\nself-attention over visual tokens", "encoder", fontsize=7.9, width=23)
    add_box(ax, 9.48, 1.78, 1.34, 1.20, "Transformer decoder\ncharacter sequence context", "decoder", fontsize=7.8, width=22)
    add_box(ax, 9.48, 0.63, 1.34, 0.58, "start token + previous characters", "decoder", fontsize=7.3, width=22)
    add_box(ax, 11.38, 1.93, 0.94, 0.90, "character logits", "logits", fontsize=7.7, width=14)
    add_box(ax, 12.62, 1.93, 0.76, 0.90, "text output", "output", fontsize=7.7, width=12)

    add_arrow(ax, (2.10, 2.44), (2.55, 2.44))
    add_arrow(ax, (3.77, 2.41), (4.20, 2.41))
    add_arrow(ax, (5.34, 2.41), (5.82, 2.40))
    add_arrow(ax, (7.00, 2.40), (7.48, 2.50))
    add_arrow(ax, (6.41, 1.56), (7.48, 2.02), rad=-0.12)
    add_arrow(ax, (8.90, 2.38), (9.48, 2.38))
    add_arrow(ax, (10.15, 1.21), (10.15, 1.78))
    add_arrow(ax, (10.82, 2.38), (11.38, 2.38))
    add_arrow(ax, (12.32, 2.38), (12.62, 2.38))

    ax.text(
        13.00,
        1.58,
        "SAMPLE TEXT FIELD",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#111827",
        family="DejaVu Sans Mono",
    )
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    print(PDF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
