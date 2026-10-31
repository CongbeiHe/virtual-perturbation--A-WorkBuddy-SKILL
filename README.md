# Geneformer Virtual Perturbation Pipeline

Standardized, reproducible **in-silico perturbation (ISP) pipeline** built on
Geneformer V2 (104M): virtual overexpression / knockout / activation / inhibition
of a target gene in any single-cell dataset, followed by cell-state displacement
analysis (UMAP shift, direction coherence with permutation baseline) and
downstream gene shift tables.

All methodology constants (fp32 forward, truncation semantics, UMAP parameters,
permutation seeds) are **hard-coded** in `geneformer_pert/config.py` so that
results are comparable across experiments and datasets.

## What you get

| Step | Script | Output |
|---|---|---|
| 1. Subset & map | `geneformer_pert/prepare.py` | `data/<set>_counts/<set>.h5ad` (int32 counts, Ensembl-indexed) |
| 2. Tokenize + ISP | `geneformer_pert/pipeline.py` | Official raw pickles + gene/cell shift CSVs |
| 3. Displacement | `geneformer_pert/displacement.py` | Per-cell shift, coherence vs permutation baseline, UMAP plots, `REPORT` inputs |

One command runs everything (idempotent — safe to re-run after interruption):

```bash
python -m geneformer_pert all \
  --src <path/to/your.h5ad> \
  --cell-filter "cell_type == 'Hepatocyte'" \
  --dataset-name myset \
  --target VNN1 \
  --mode overexpress
```

Modes: `overexpress` / `knockout` / `delete` / `activate` / `inhibit`.

## Quick start

See **[INSTALL.md](INSTALL.md)** for full deployment (GPU required).
Minimal path:

```powershell
# 1. create env (Windows; see INSTALL.md for Linux/Git Bash)
.\setup_env.ps1
# 2. download model weights (~400 MB from HuggingFace)
.\venv\Scripts\python.exe tools\download_model.py
# 3. run
.\venv\Scripts\python.exe -m geneformer_pert all --src <h5ad> --cell-filter "<query>" --dataset-name demo --target VNN1
```

## Interpretation rules (enforced in reports)

1. Cosine shift magnitude ranks *which genes are affected* — it is **not**
   a significance test.
2. Up/down labels require real-data anchoring; model-inferred direction alone
   is ~58-61% accurate (calibrated against real DE) and must be labeled as such.
3. Directional migration claims require coherence > permutation baseline p95.

Details and the full decision workflow: `skills/virtual-perturbation/SKILL.md`.

## Repository layout

```
geneformer_pert/     pipeline package (config / prepare / pipeline / displacement / cli)
code/geneformer/     official Geneformer source (installed editable)
tools/               utilities: VRAM guard, env checks, model download
scripts/             reference experiment implementations (fixed local paths, for study)
skills/              agent-facing workflow definition (SKILL.md + method notes)
```

## Requirements

- NVIDIA GPU (CUDA), 8 GB+ VRAM recommended; ~5-7 GPU-min per 1,000 cells per stage
- Python 3.10-3.13; model weights downloaded separately (400 MB, see INSTALL.md)
- Test dataset `GSE189175_NASHHCC.h5ad` (375 MB) is **not** in the repo —
  download from the GitHub Releases page of this project, or prepare your own h5ad
  with an integer-count layer.

## Third-party components

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Geneformer code (c) Christina Theodoris lab, MIT license, vendored unchanged
from the official repository.
