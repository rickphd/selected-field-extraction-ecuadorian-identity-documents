#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evidence" / "supplementary_validation"

REPLACEMENTS = {
    "events.out.tfevents.1762409345.b2588193a6fe.1429216.1": "run_01.tfevents",
    "events.out.tfevents.1762426187.b2588193a6fe.1429372.1": "run_01.tfevents",
    "events.out.tfevents.1762376848.b2588193a6fe.1429216.0": "run_01.tfevents",
    "events.out.tfevents.1762376851.b2588193a6fe.1429372.0": "run_01.tfevents",
    "events.out.tfevents.1762195396.Jeff.33112.0": "run_01.tfevents",
    "events.out.tfevents.1762376855.b2588193a6fe.1429624.0": "run_02.tfevents",
}


def normalize(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for original, replacement in REPLACEMENTS.items():
        text = text.replace(original, replacement)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for filename in ("ocr_tensorboard_scalars.csv", "ocr_tensorboard_summary.csv"):
        normalize(ROOT / filename)


if __name__ == "__main__":
    main()
