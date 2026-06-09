# Selected-Field Extraction from Ecuadorian Identity Documents

This repository is a public companion to the manuscript on selected-field extraction from Ecuadorian identity documents. It brings together the implementation, supporting evidence, and release documentation needed to accompany the paper in a compact and publication-ready form.

## Manuscript Alignment

The repository accompanies the manuscript:

*Selected-Field Extraction from Ecuadorian Identity Documents with Oriented Detection and RKNN Deployment*

## Repository Goals

- preserve the code paths used for training, evaluation, export, and deployment preparation,
- preserve compact evidence that supports manuscript claims,
- document the release boundaries and retained materials,
- provide a repository that can be cited directly from the paper as the public code-and-evidence companion.

## Public Release Scope

This repository includes:

- training and utility code for segmentation-orientation, OCR, YOLO OBB detection, and model export,
- notebooks prepared for public inspection,
- compact CSV, PDF, and PNG evidence artifacts used during manuscript validation,
- selected training-log traces supporting the reported OCR development workflow,
- project documentation and release metadata.

This repository does not include:

- source image collections and derived crops,
- notebooks containing sample-level output traces,
- heavyweight model binaries (`.pth`, `.pt`, `.onnx`, `.onnx.data`, `.rknn`),
- Monte Carlo image folders and other bulky intermediate assets.

## Repository Layout

```text
src/
  detection_obb/            YOLO OBB training script and dataset config
  segmentation_rotation/    Segmentation + orientation training code
  ocr/                      OCR model definitions and training scripts
  model_export/             ONNX/RKNN export utilities and deployment helpers
  pipeline_runtime/         Runtime OCR/pipeline helper classes

notebooks/
  detection_obb/            YOLO OBB workflow notebooks
  segmentation_rotation/    Segmentation-orientation workflow notebooks
  ocr/                      OCR workflow notebooks
  model_export/             Export and runtime-characterization notebooks

evidence/
  supplementary_validation/ Validation tables and supplementary evidence artifacts
  yolo_validation/          YOLO validation tables and summary plots
  runtime/                  Runtime-distribution figures retained for documentation
  ocr_logs/                 Training-log traces retained for documentation

metadata/
  training_tables/          Release-ready CSV tables referenced by training code
  excluded_public_assets.csv

docs/
  source_provenance.md
  public_release_scope.md

scripts/
  manuscript_figures/       Figure-generation scripts retained for manuscript support
  sanitize_notebooks.py
  sanitize_metadata_tables.py
  normalize_ocr_evidence_names.py
```

## Reproducibility Strategy

The full project depended on non-public datasets and large local model artifacts. A direct, one-command full reproduction from scratch is therefore not possible from this repository alone. Instead, the release supports reproducibility at three levels:

1. Code reproducibility:
   the training, export, and supporting code paths are preserved.
2. Evidence reproducibility:
   compact metric tables, training-log traces, and validation summaries are preserved.
3. Release transparency:
   release boundaries and unavailable asset classes are documented explicitly.

## Environment

The most complete dependency snapshot preserved from the workspace is:

- [src/model_export/requirements.txt](src/model_export/requirements.txt)

That file is environment-heavy and includes notebook and CUDA packages. For a practical starting point, install the project dependencies in an isolated environment and adapt versions to your hardware:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/model_export/requirements.txt
```

If RKNN export is not needed, you can omit `rknn-toolkit2` and related Rockchip-specific steps.

## Additional Materials

Requests for additional materials related to the repository or manuscript should be directed to the corresponding author listed in the paper:

- Ricardo Flores-Moyano — `rflores@usfq.edu.ec`

## Recommended Reading Order

1. Read [docs/public_release_scope.md](docs/public_release_scope.md).
2. Read [docs/source_provenance.md](docs/source_provenance.md).
3. Inspect [metadata/excluded_public_assets.csv](metadata/excluded_public_assets.csv).
4. Start from the relevant module under `src/`.
5. Use the notebooks as supporting workflow references, not as the sole source of truth.

## Notes for Citation

This repository is intended to be cited in the manuscript as the public code-and-evidence companion repository. If a DOI-backed archival release is required, create a tagged GitHub release and archive it in Zenodo before submission.
