"""
nichenet_pbmc3k_example.py
==========================
End-to-end ligflow example using

* **NicheNet** ligand-target priors exported by ``build_nichenet_priors.R``
* **PBMC 3k** single-cell RNA-seq data from 10x Genomics (provided by scanpy)

Workflow
--------
1. Download / load PBMC 3k (scanpy built-in dataset, ~2 700 cells, 32 738 genes).
2. Standard preprocessing: filter, normalise, log-transform, PCA, UMAP.
3. Load NicheNet prior network and GRN produced by ``build_nichenet_priors.R``.
4. Choose a ligand of interest that is expressed in the dataset.
5. Run :func:`ligflow.run_ligand_flow`.
6. Save diagnostic plots.

Prerequisites
-------------
1. Run the R script first::

       Rscript examples/build_nichenet_priors.R

   This writes ``nichenet_prior.csv`` and ``nichenet_grn.csv`` to the current
   working directory.

2. Install Python dependencies::

       pip install ligflow scanpy

Run
---
::

    python examples/nichenet_pbmc3k_example.py

Output files
------------
* ``nichenet_pbmc3k_velocity.png``       – UMAP coloured by cell type + velocity arrows
* ``nichenet_pbmc3k_magnitude.png``      – UMAP coloured by perturbation effect magnitude
* ``nichenet_pbmc3k_top_genes.png``      – top predicted target genes
* ``nichenet_pbmc3k_diagnostics.png``    – six-panel workflow diagnostic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import

import scanpy as sc
import ligflow as lf

# ── Configuration ──────────────────────────────────────────────────────────────

# Ligand to perturb.  Must be present in PBMC 3k gene names AND in
# nichenet_prior.csv.  Common human ligands in PBMCs:
#   "CXCL12"  – stromal/CXCR4-axis
#   "CXCL10"  – interferon-induced, expressed by monocytes/DCs
#   "CCL5"    – expressed by T cells and NK cells (RANTES)
#   "TNF"     – expressed by monocytes and macrophages
#   "TGFB1"   – immunosuppressive cytokine
LIGAND = "CXCL12"

# Paths to CSV files produced by build_nichenet_priors.R
PRIOR_CSV = "nichenet_prior.csv"
GRN_CSV   = "nichenet_grn.csv"

# Embedding parameters
N_PCS       = 30
N_NEIGHBORS = 30

# ligflow parameters
N_ITER  = 3
DAMPING = 0.8

# ── 1. Check that NicheNet CSV files exist ─────────────────────────────────────

for csv_path in (PRIOR_CSV, GRN_CSV):
    if not Path(csv_path).exists():
        sys.exit(
            f"Error: '{csv_path}' not found.\n"
            "Please run the R script first:\n"
            "  Rscript examples/build_nichenet_priors.R\n"
            "This writes nichenet_prior.csv and nichenet_grn.csv to the current"
            " working directory."
        )

# ── 2. Load PBMC 3k ───────────────────────────────────────────────────────────

print("Loading PBMC 3k dataset ...")
# sc.datasets.pbmc3k() downloads the raw count matrix on first call and
# caches it in ~/.cache/scanpy.
adata = sc.datasets.pbmc3k()
print(f"  {adata}")                  # AnnData: 2 700 × 32 738

# ── 3. Preprocessing ──────────────────────────────────────────────────────────

print("\nPreprocessing ...")

# Basic quality-control filtering (minimal, dataset is already clean)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

print(f"  After QC: {adata.n_obs} cells, {adata.n_vars} genes")

# Identify and annotate mitochondrial genes before normalisation
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# Compute PCA+UMAP via ligflow's convenience wrapper (normalise → log1p → PCA → k-NN → UMAP)
print("  Computing PCA + UMAP embedding ...")
lf.compute_embedding(adata, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS, normalize=True)
print(f"  PCA  : {adata.obsm['X_pca'].shape}")
print(f"  UMAP : {adata.obsm['X_umap'].shape}")

# Cluster cells (Leiden) so we can colour the UMAP
print("  Clustering (Leiden) ...")
sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
print(f"  {adata.obs['leiden'].nunique()} clusters found")

# ── 4. Load NicheNet networks ─────────────────────────────────────────────────

print(f"\nLoading NicheNet prior from '{PRIOR_CSV}' ...")
prior = lf.load_prior(PRIOR_CSV)
print(f"  {len(prior):,} ligand-target edges; "
      f"{prior['source'].nunique():,} ligands; "
      f"{prior['target'].nunique():,} target genes")

print(f"Loading GRN from '{GRN_CSV}' ...")
grn = lf.load_grn(GRN_CSV)
print(f"  {len(grn):,} GRN edges")

# ── 5. Verify ligand is usable ────────────────────────────────────────────────

if LIGAND not in prior["source"].values:
    available = sorted(prior["source"].unique().tolist())
    sys.exit(
        f"Ligand '{LIGAND}' not found in the prior network.\n"
        f"Available ligands (first 20): {available[:20]}"
    )

if LIGAND not in adata.var_names:
    print(
        f"Warning: ligand '{LIGAND}' is not in the PBMC 3k gene list. "
        "The perturbation vector will still be built from prior network weights."
    )

# How many prior targets overlap with the dataset's genes?
prior_targets = set(prior.loc[prior["source"] == LIGAND, "target"])
overlap = prior_targets & set(adata.var_names)
print(f"\nLigand '{LIGAND}': {len(prior_targets)} prior targets, "
      f"{len(overlap)} overlap with PBMC 3k genes")

# ── 6. Run ligflow ────────────────────────────────────────────────────────────

print(f"\nRunning ligand flow for '{LIGAND}' ...")
result = lf.run_ligand_flow(
    adata,
    ligand=LIGAND,
    prior_network=prior,
    grn_network=grn,
    embedding_key="X_umap",
    n_neighbors=N_NEIGHBORS,
    n_iter=N_ITER,
    damping=DAMPING,
    copy=True,
)

# Print a short summary
shift = result.layers["ligand_shift"]
velocities = result.obsm["X_ligand_velocity"]
mean_shift = np.abs(shift).mean(axis=0)
top5_idx = np.argsort(mean_shift)[::-1][:5]
print("\nTop-5 predicted target genes by mean absolute shift:")
for idx in top5_idx:
    print(f"  {result.var_names[idx]}: {mean_shift[idx]:.4f}")
print(f"\nVelocity vector norm (mean): {np.linalg.norm(velocities, axis=1).mean():.4f}")

# ── 7. Save plots ─────────────────────────────────────────────────────────────

print("\nSaving plots ...")

lf.pl.velocity_embedding(
    result,
    basis="umap",
    color="leiden",
    show=False,
    save="nichenet_pbmc3k_velocity.png",
)
print("  → nichenet_pbmc3k_velocity.png")

lf.pl.ligand_effect_magnitude(
    result,
    basis="umap",
    show=False,
    save="nichenet_pbmc3k_magnitude.png",
)
print("  → nichenet_pbmc3k_magnitude.png")

lf.pl.top_target_genes(
    result,
    n_genes=20,
    show=False,
    save="nichenet_pbmc3k_top_genes.png",
)
print("  → nichenet_pbmc3k_top_genes.png")

lf.pl.workflow_diagnostics(
    result,
    ligand=LIGAND,
    prior_network=prior,
    grn_network=grn,
    basis="umap",
    n_neighbors=N_NEIGHBORS,
    n_iter=N_ITER,
    damping=DAMPING,
    show=False,
    save="nichenet_pbmc3k_diagnostics.png",
)
print("  → nichenet_pbmc3k_diagnostics.png")

print("\nDone.")
