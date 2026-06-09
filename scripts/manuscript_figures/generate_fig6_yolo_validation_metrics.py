from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "supplementary_validation"
CSV_PATH = ROOT / "evidence" / "yolo_validation" / "results.csv"

METRICS = [
    ("metrics/precision(B)", "Precision", "#e11d48", "-"),
    ("metrics/recall(B)", "Recall", "#f59e0b", "--"),
    ("metrics/mAP50(B)", "mAP50", "#0f766e", "-."),
    ("metrics/mAP50-95(B)", "mAP50-95", "#7c3aed", ":"),
]

rows = []
with CSV_PATH.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        rows.append(row)

epochs = [int(row["epoch"]) for row in rows]

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

fig, ax = plt.subplots(figsize=(8.9, 3.95), constrained_layout=True)

for metric, label, color, linestyle in METRICS:
    values = [float(row[metric]) for row in rows]
    ax.plot(
        epochs,
        values,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )
    best_idx = max(range(len(values)), key=values.__getitem__)
    ax.scatter(
        epochs[best_idx],
        values[best_idx],
        s=34,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Metric value")
ax.set_xlim(0, 145)
ax.set_ylim(0.5, 1.01)
ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.9)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
legend = ax.legend(
    frameon=True,
    facecolor="white",
    edgecolor="#000000",
    framealpha=1.0,
    loc="center right",
    bbox_to_anchor=(1.0, 0.5),
    ncol=1,
)
legend.get_frame().set_linewidth(0.5)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supp_fig_s2_yolo_validation_metrics.{ext}", bbox_inches="tight")
