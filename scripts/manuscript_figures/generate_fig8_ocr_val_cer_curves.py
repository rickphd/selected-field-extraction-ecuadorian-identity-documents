from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "evidence" / "supplementary_validation" / "ocr_tensorboard_scalars.csv"
OUT = ROOT / "evidence" / "supplementary_validation"

SERIES = [
    ("transformer", "Transformer", "#0f766e", "-"),
    ("lstm_mobilenet", "LSTM-MobileNet", "#7c3aed", "--"),
    ("gru_mobilenet", "GRU-MobileNet", "#2563eb", "-."),
    ("gru_custom", "GRU-Custom", "#f59e0b", ":"),
    ("lstm_custom", "LSTM-Custom", "#e11d48", (0, (3, 1, 1, 1))),
]

rows_by_run_event = defaultdict(list)
with CSV_PATH.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row["tag"] != "CER/val":
            continue
        key = (row["run"], row["event_file"])
        rows_by_run_event[key].append(
            {
                "step": int(float(row["step"])),
                "value": float(row["value"]),
            }
        )

complete_series = {}
for run, _label, _color, _linestyle in SERIES:
    series_options = [
        (event_file, points)
        for (run_name, event_file), points in rows_by_run_event.items()
        if run_name == run
    ]
    if not series_options:
        raise RuntimeError(f"No CER/val series found for {run}")
    event_file, points = max(series_options, key=lambda item: len(item[1]))
    if len(points) < 2:
        raise RuntimeError(f"Only incomplete CER/val series found for {run}: {event_file}")
    complete_series[run] = sorted(points, key=lambda item: item["step"])

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

for run, label, color, linestyle in SERIES:
    points = complete_series[run]
    steps = [point["step"] for point in points]
    values = [point["value"] for point in points]
    ax.plot(
        steps,
        values,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )
    best_idx = min(range(len(values)), key=values.__getitem__)
    ax.scatter(
        steps[best_idx],
        values[best_idx],
        s=34,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

ax.set_xlabel("Checkpoint")
ax.set_ylabel("Validation CER (%)")
ax.set_xlim(0, 160)
ax.set_ylim(0, 85)
ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.9)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
legend = ax.legend(
    frameon=True,
    facecolor="white",
    edgecolor="#000000",
    framealpha=1.0,
    loc="upper right",
    ncol=1,
)
legend.get_frame().set_linewidth(0.5)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supp_fig_s1_ocr_val_cer_curves.{ext}", bbox_inches="tight")
