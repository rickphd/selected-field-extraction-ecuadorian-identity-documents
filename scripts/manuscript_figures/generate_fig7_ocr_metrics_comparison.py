from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts" / "manuscript_figures" / "generated"
CSV_PATH = ROOT / "evidence" / "supplementary_validation" / "ocr_tensorboard_scalars.csv"
OUT.mkdir(parents=True, exist_ok=True)

RUN_LABELS = {
    "gru_custom": "GRU-Custom",
    "gru_mobilenet": "GRU-MobileNet",
    "lstm_custom": "LSTM-Custom",
    "lstm_mobilenet": "LSTM-MobileNet",
    "transformer": "Transformer",
}

RUN_COLORS = {
    "gru_custom": "#e11d48",
    "gru_mobilenet": "#f59e0b",
    "lstm_custom": "#111827",
    "lstm_mobilenet": "#06b6d4",
    "transformer": "#7c3aed",
}

RUN_LINESTYLES = {
    "gru_custom": "-",
    "gru_mobilenet": "--",
    "lstm_custom": "-.",
    "lstm_mobilenet": ":",
    "transformer": "-",
}

PANEL_SPECS = [
    ("CER/train", "CER (%)", (0, 90)),
    ("CER/val", "CER (%)", (0, 87)),
    ("Loss/train", "Loss", (0.65, 2.95)),
    ("Loss/val", "Loss", (0.7, 2.9)),
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

series = defaultdict(lambda: defaultdict(list))
with CSV_PATH.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        run = row["run"]
        tag = row["tag"]
        if run not in RUN_LABELS:
            continue
        if tag not in {spec[0] for spec in PANEL_SPECS}:
            continue
        series[run][tag].append((int(row["step"]), float(row["value"])))

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

fig, axes = plt.subplots(2, 2, figsize=(10.7, 6.25), constrained_layout=True)

for ax, (tag, ylabel, ylim), panel_label in zip(axes.ravel(), PANEL_SPECS, PANEL_LABELS):
    for run, label in RUN_LABELS.items():
        points = sorted(series[run][tag])
        if not points:
            continue
        steps = [point[0] for point in points]
        values = [point[1] for point in points]
        ax.plot(
            steps,
            values,
            color=RUN_COLORS[run],
            linestyle=RUN_LINESTYLES[run],
            linewidth=1.7,
            label=label,
        )

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-2, 162)
    ax.set_ylim(*ylim)
    ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(frameon=True, loc="upper right")
    leg.get_frame().set_edgecolor("#000000")
    leg.get_frame().set_linewidth(0.5)
    ax.text(
        0.5,
        -0.26,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig7_ocr_metrics_comparison.{ext}", bbox_inches="tight")
