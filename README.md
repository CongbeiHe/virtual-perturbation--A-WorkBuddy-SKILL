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

Modes: `overexpress` / `knockout` / `delete`.
`activate` / `inhibit` are **not usable**: official Geneformer's
`validate_options()` raises on them ("currently under development") and
`perturber_utils.py` has no implementation. Passing them through this pipeline
silently falls back to **overexpress** — do not report results as activation or
inhibition.

## Overexpression: two caveats you must handle

1. **Restrict to cells that already express the target (target⁺).**
   `overexpress_tokens()` deletes the token in place and re-inserts it at
   position 1 when the gene is present (length preserved, rank p → 1), but
   **only inserts** when the gene is absent (length +1 → the whole sequence
   shifts). Cells that do not express the gene therefore take a purely
   mechanical shortcut and inflate 1−cos by ~8-11%. Check the target's median
   rank first: if p ≳ 500 the shifted span already covers the whole high-rank
   head, so the restriction only costs you sample size and buys nothing.

2. **A rank-matched null distribution is mandatory.**
   Restricting to target⁺ removes contamination but gives you no null. Run the
   same L1 chain on ≥20 rank-matched control genes over the same cells and
   compare: `z = (target − mean(controls)) / sd(controls)`, empirical
   `p = (k+1)/(n+1)`. With only 4 controls the floor is p = 0.20 — significance
   is unreachable by design; Kendiukhov et al. used 61 controls. When picking
   controls, validate against **both** `token_dictionary_gc104M.pkl` and
   `gene_name_id_dict_gc104M.pkl` (a gene can have an Ensembl ID but no token,
   which crashes the run).

Details: `skills/virtual-perturbation/SKILL.md` §10.

## Rank-matched null controls (scripts 40-47)

The two caveats above are implemented as a reusable workflow. All scripts take
`--workdir` (or env `GENEFORMER_PERT_WORKDIR`) and `--code-dir` (or env
`GENEFORMER_CODE_DIR`); nothing is hard-coded.

| # | Script | Purpose |
|---|---|---|
| 40 | `40_probe_target_detection.py` | Detection rate of the target per dataset — is a target⁺ subset even viable? |
| 41 | `41_target_rank_position.py` | **Run this first.** Target's median token rank + candidate control pool (`--top 300`) |
| 42 | `42_make_pos_subset.py` | Build the target⁺ (counts ≥ 1) subset |
| 43 | `43_make_null_subsample.py` | Fixed-size subsample so controls cost less GPU |
| 44 | `44_pick_null_controls.py` | Pick rank-matched controls (≥ 20 needed) |
| 45 | `45_run_controls_displacement.py` | Run controls, **displacement only** (skips ISP, ~half the time) |
| 46 | `46_null_comparison.py` | Mechanism test + null test (z, empirical p, rank-calibrated residual) |
| 47 | `47_pick_extra_controls.py` | Top controls up to 20 so empirical p can reach 0.05 |

Minimal flow:

```bash
export GENEFORMER_PERT_WORKDIR=/path/to/workdir
python scripts/41_target_rank_position.py --datasets myset --genes MYGENE --top 300
python scripts/42_make_pos_subset.py --target MYGENE --datasets myset
python scripts/47_pick_extra_controls.py --target MYGENE --datasets myset --n-new 16
python scripts/45_run_controls_displacement.py --datasets myset_mygene \
    --genes-file outputs/mygene20_genes.txt
python scripts/46_null_comparison.py --target MYGENE --datasets myset \
    --controls-file outputs/mygene20_genes.txt
```

Decision rule from step 41: **if the target's median rank p ≳ 500, restricting to
target⁺ only costs you sample size.** The shifted span `[1, p-1]` already covers
the whole high-rank head, so the mechanical signature survives the filter. Use a
length-preserving rank edit instead (SKILL.md §10.7).

Script numbering: SKILL.md §10 cites the original in-project numbers (18/19/20/
21/22/27/28) because those are the ones used in the reference experiment; they
map to the release numbers 40/41/43/44/46/47/45 respectively.

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
4. **Gene-specific** effects require a rank-matched null distribution
   (≥20 control genes). High coherence only means the shifts point the same
   way — it says nothing about whether the effect is specific to the target gene.

Details and the full decision workflow: `skills/virtual-perturbation/SKILL.md`.

## Repository layout

```
geneformer_pert/     pipeline package (config / prepare / pipeline / displacement / cli)
code/geneformer/     official Geneformer source (installed editable)
tools/               utilities: VRAM guard, env checks, model download
scripts/             reference implementations; 00-33 are the original experiments
                     (fixed local paths, for study), 40-47 are the portable
                     rank-matched null-control workflow (take --workdir)
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
