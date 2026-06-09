#!/usr/bin/env python3
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "metadata" / "training_tables"


def pseudonymize(value: str) -> str:
    raw = str(value)
    suffix = Path(raw).suffix or ".dat"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"sample_{digest}{suffix.lower()}"


def sanitize_table(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "File" in fieldnames:
        for row in rows:
            row["File"] = pseudonymize(row["File"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for csv_path in ROOT.glob("*.csv"):
        sanitize_table(csv_path)


if __name__ == "__main__":
    main()
