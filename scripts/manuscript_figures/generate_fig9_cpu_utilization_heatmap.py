from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Manuscript"

onnx_cpu = np.array(
    [
        [93.7, 93.7, 93.3, 93.2, 88.3, 88.2, 97.9, 88.6],
        [94.2, 94.1, 93.8, 93.7, 88.7, 88.7, 98.1, 89.1],
        [93.7, 93.8, 93.2, 93.1, 88.2, 88.4, 97.9, 88.6],
        [93.9, 93.8, 93.4, 93.3, 88.2, 88.2, 97.9, 88.3],
        [93.5, 93.4, 93.0, 92.8, 88.2, 87.9, 97.8, 88.3],
    ]
)

rknn_cpu = np.array(
    [
        [13.1, 5.2, 3.4, 2.7, 18.8, 12.8, 7.4, 5.8],
        [12.4, 5.9, 3.7, 2.8, 17.5, 12.9, 8.7, 5.8],
        [13.5, 5.1, 3.2, 2.7, 17.8, 11.8, 8.1, 6.7],
        [13.7, 6.2, 3.2, 2.9, 17.5, 14.1, 8.1, 6.4],
        [12.6, 5.6, 3.7, 2.9, 18.2, 12.1, 9.3, 7.0],
    ]
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": "#111827",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.35), constrained_layout=True)
data_panels = [onnx_cpu, rknn_cpu]
panel_labels = ["(a)", "(b)"]
cpu_cmap = LinearSegmentedColormap.from_list(
    "paper_cpu",
    ["#c4eee9", "#9fe2dc", "#65ccc4", "#2ea89f", "#0f766e"],
)

for ax, data, panel_label in zip(axes, data_panels, panel_labels):
    image = ax.imshow(data, cmap=cpu_cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xlabel("CPU core")
    ax.set_ylabel("Test folder")
    ax.set_xticks(np.arange(data.shape[1]), [f"CPU{i}" for i in range(data.shape[1])])
    ax.set_yticks(np.arange(data.shape[0]), [str(i) for i in range(1, data.shape[0] + 1)])
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            text_color = "#ffffff" if value >= 55 else "#111827"
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="bold",
                color=text_color,
            )
    ax.text(
        0.5,
        -0.28,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )

colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
colorbar.set_label("Average CPU utilization (%)")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig9_cpu_utilization_heatmap.{ext}", bbox_inches="tight")
