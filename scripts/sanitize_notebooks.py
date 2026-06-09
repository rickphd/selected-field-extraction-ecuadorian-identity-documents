#!/usr/bin/env python3
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "notebooks"

PRIVATE_PATTERNS = [
    (re.compile(r'[^"\n]*ImagenCedula[^"\n]*\.(jpg|png)', re.IGNORECASE), "sample_image.jpg"),
    (re.compile(r'[^"\n]*imagen_recibida_\d+\.(jpg|png)', re.IGNORECASE), "sample_image.jpg"),
    (re.compile(r"/home/[^\"'\n ]+"), "/path/to/private/data"),
    (re.compile(r"\.\./test/[^\"'\n ]+"), "../test/sample_image.png"),
    (re.compile(r"Test/[A-Za-z0-9_-]+\.(jpg|png)", re.IGNORECASE), "Test/sample_image.jpg"),
]


def sanitize_source(source):
    if isinstance(source, list):
        text = "".join(source)
        as_list = True
    else:
        text = source
        as_list = False

    for pattern, replacement in PRIVATE_PATTERNS:
        text = pattern.sub(replacement, text)

    if as_list:
        return text.splitlines(keepends=True)
    return text


def sanitize_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        if "source" in cell:
            new_source = sanitize_source(cell["source"])
            if new_source != cell["source"]:
                cell["source"] = new_source
                changed = True

        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

    metadata = notebook.setdefault("metadata", {})
    if "widgets" in metadata:
        metadata.pop("widgets", None)
        changed = True

    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    for notebook_path in ROOT.rglob("*.ipynb"):
        sanitize_notebook(notebook_path)


if __name__ == "__main__":
    main()
