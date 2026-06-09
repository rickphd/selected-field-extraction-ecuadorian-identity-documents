# Repository Scope

## Scope Statement

This repository is an academic companion release aligned with the accompanying manuscript. It is intended to provide the implementation, compact evidence artifacts, and supporting documentation required for inspection and citation.

The release intentionally focuses on materials that are portable, inspectable, and directly useful for scholarly review.

## Release Preparation

- Jupyter notebook outputs were removed.
- Execution counters were reset.
- Notebook references were normalized where required for public distribution.
- Training-log files were retained under release-neutral filenames.

## Release Boundaries

Excluded assets are listed in:

- [metadata/excluded_public_assets.csv](../metadata/excluded_public_assets.csv)

The main exclusion categories are:

- source image collections and derived crops,
- notebooks containing sample-level output traces,
- heavyweight model binaries,
- bulky intermediate Monte Carlo folders,
- cache folders and generated auxiliary files.

## What this means for manuscript support

This repository supports the paper as a public code-and-evidence companion. It is designed to support inspection, citation, and methodological review of the reported workflow.

## Contact

Requests for additional repository or manuscript materials should be addressed to the corresponding author:

- Ricardo Flores-Moyano — `rflores@usfq.edu.ec`
