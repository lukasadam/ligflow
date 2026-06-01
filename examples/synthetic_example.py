"""
synthetic_example.py
====================
Minimal end-to-end example of ligflow using fully synthetic data.

Run with:
    python examples/synthetic_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import ligflow as lf

# ── 1. Build a toy AnnData ────────────────────────────────────────────────────

rng = np.random.default_rng(42)

N_CELLS = 200
GENE_NAMES = [f"Gene{i}" for i in range(50)]
LIGAND = "Gene0"

X = rng.exponential(scale=2.0, size=(N_CELLS, len(GENE_NAMES))).astype(np.float32)
clusters = np.repeat(["TypeA", "TypeB", "TypeC", "TypeD"], N_CELLS // 4)

adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame(
        {"cell_type": pd.Categorical(clusters)},
        index=[f"Cell{i}" for i in range(N_CELLS)],
    ),
    var=pd.DataFrame(index=GENE_NAMES),
)

# Fake 2-D UMAP by embedding clusters as blobs
centres = {"TypeA": (-3, 3), "TypeB": (3, 3), "TypeC": (-3, -3), "TypeD": (3, -3)}
coords = np.vstack([
    rng.normal(loc=centres[c], scale=0.8, size=(N_CELLS // 4, 2))
    for c in ["TypeA", "TypeB", "TypeC", "TypeD"]
])
adata.obsm["X_umap"] = coords.astype(np.float32)

print(f"AnnData: {adata}")

# ── 2. Synthetic prior network ────────────────────────────────────────────────

prior_df = pd.DataFrame(
    {
        "source": LIGAND,
        "target": GENE_NAMES[1:16],          # Gene0 → Gene1 … Gene15
        "weight": rng.uniform(0.2, 1.0, 15),
        "interaction_type": "ligand_target",
    }
)

# ── 3. Synthetic GRN ─────────────────────────────────────────────────────────

n_edges = 100
grn_df = pd.DataFrame(
    {
        "source": rng.choice(GENE_NAMES[1:26], n_edges),
        "target": rng.choice(GENE_NAMES[1:26], n_edges),
        "weight": rng.uniform(0.05, 0.5, n_edges),
        "interaction_type": "TF_target",
    }
)

# ── 4. Run ligflow ────────────────────────────────────────────────────────────

print(f"\nRunning ligand flow for ligand: {LIGAND}")
result = lf.run_ligand_flow(
    adata,
    ligand=LIGAND,
    prior_network=prior_df,
    grn_network=grn_df,
    embedding_key="X_umap",
    n_neighbors=20,
    n_iter=3,
    damping=0.8,
    copy=True,
)

print("\nResults stored in AnnData:")
print(f"  layers:  {list(result.layers.keys())}")
print(f"  obsm:    {list(result.obsm.keys())}")
print(f"  uns:     {list(result.uns.keys())}")

shift = result.layers["ligand_shift"]
print(f"\nligand_shift shape: {shift.shape}")
print(f"Mean absolute shift per gene (top 5):")
mean_shift = np.abs(shift).mean(axis=0)
top5 = np.argsort(mean_shift)[::-1][:5]
for idx in top5:
    print(f"  {GENE_NAMES[idx]}: {mean_shift[idx]:.4f}")

velocities = result.obsm["X_ligand_velocity"]
print(f"\nVelocity vector norms (mean): {np.linalg.norm(velocities, axis=1).mean():.4f}")

# ── 5. (Optional) Plot ────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    lf.pl.velocity_embedding(
        result,
        basis="umap",
        color="cell_type",
        show=False,
        save="ligflow_velocity.png",
    )
    print("\nSaved velocity plot to ligflow_velocity.png")

    lf.pl.ligand_effect_magnitude(result, show=False, save="ligflow_magnitude.png")
    print("Saved magnitude plot to ligflow_magnitude.png")

    lf.pl.top_target_genes(result, n_genes=10, show=False, save="ligflow_top_genes.png")
    print("Saved top-genes plot to ligflow_top_genes.png")

    lf.pl.workflow_diagnostics(
        result,
        ligand=LIGAND,
        prior_network=prior_df,
        grn_network=grn_df,
        basis="umap",
        n_neighbors=20,
        n_iter=3,
        damping=0.8,
        show=False,
        save="ligflow_workflow_diagnostics.png",
    )
    print("Saved workflow diagnostics plot to ligflow_workflow_diagnostics.png")

except ImportError:
    print("\n(matplotlib not available; skipping plots)")

print("\nDone.")
