"""
Shared pytest fixtures for ligflow tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad
import scanpy as sc


# ── Synthetic AnnData ─────────────────────────────────────────────────────────

N_CELLS = 80
N_GENES = 40


@pytest.fixture(scope="session")
def gene_names() -> list[str]:
    return [f"Gene{i}" for i in range(N_GENES)]


@pytest.fixture(scope="session")
def adata(gene_names) -> ad.AnnData:
    """Small synthetic AnnData with random expression, PCA, and UMAP embedding."""
    rng = np.random.default_rng(42)
    X = rng.exponential(scale=1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs_names = [f"Cell{i}" for i in range(N_CELLS)]
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(np.repeat(["A", "B", "C", "D"], N_CELLS // 4))},
            index=obs_names,
        ),
        var=pd.DataFrame(index=gene_names),
    )
    # Compute PCA, k-NN graph, and UMAP via scanpy
    n_pcs = min(10, N_CELLS - 1, N_GENES - 1)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=n_pcs)
    sc.tl.umap(adata, n_components=2)
    return adata


# ── Synthetic prior network ───────────────────────────────────────────────────

LIGAND = "Gene0"  # treat Gene0 as a "ligand"


@pytest.fixture(scope="session")
def prior_network(gene_names) -> pd.DataFrame:
    """Small synthetic prior with Gene0 as source."""
    rng = np.random.default_rng(0)
    targets = gene_names[1:11]  # Gene1 … Gene10
    return pd.DataFrame(
        {
            "source": LIGAND,
            "target": targets,
            "weight": rng.uniform(0.1, 1.0, size=len(targets)),
            "interaction_type": "ligand_target",
        }
    )


@pytest.fixture(scope="session")
def grn_network(gene_names) -> pd.DataFrame:
    """Small synthetic GRN: random edges among Gene1…Gene20."""
    rng = np.random.default_rng(1)
    n_edges = 60
    all_genes = gene_names[1:21]
    sources = rng.choice(all_genes, size=n_edges)
    targets = rng.choice(all_genes, size=n_edges)
    return pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "weight": rng.uniform(0.1, 1.0, size=n_edges),
            "interaction_type": "TF_target",
        }
    )
