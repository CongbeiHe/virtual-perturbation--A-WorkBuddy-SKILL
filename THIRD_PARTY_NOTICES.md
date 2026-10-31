# Third-party notices

## Geneformer (vendored in `code/geneformer/`)

- Source: https://github.com/ctheodoris/Geneformer (official repository)
- License: MIT (retained in the vendored source headers/repository)
- Model weights (`V2-104M`) are NOT redistributed with this repository.
  Download via `tools/download_model.py` from the official HuggingFace
  repository `ctheodoris/Geneformer`; usage terms are defined on the
  HuggingFace model page and remain those of the original authors.

## pandas (deployment dependency, not vendored)

- Installed from PyPI (`pandas==2.3.3`), BSD 3-Clause License,
  (c) Pandas Development Team. https://pandas.pydata.org

## PyTorch / CUDA

- Installed from the official PyTorch wheel index; BSD-style license.
  CUDA toolkit components are subject to NVIDIA's license terms.

## Test dataset

- `GSE189175_NASHHCC.h5ad` is a processed matrix derived from public GEO
  accession GSE189175. Cite the original study when using it. It is
  distributed through GitHub Release assets (not the git repository).

## Other Python dependencies

All remaining dependencies (scanpy, anndata, transformers, datasets, igraph,
leidenalg, umap-learn, etc.) are installed from PyPI at the pinned versions in
`requirements.txt` and remain under their respective upstream licenses.
