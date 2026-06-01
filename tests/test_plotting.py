"""Tests for ligflow.plotting."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import pandas as pd

from ligflow.imputation import knn_smooth_array
from ligflow.plotting import ligand_effect_magnitude, workflow_diagnostics


def test_ligand_effect_magnitude_smooths_over_knn_graph(adata):
    adata.obsm["X_ligand_velocity"] = np.column_stack(
        [np.arange(adata.n_obs, dtype=float), np.zeros(adata.n_obs, dtype=float)]
    )

    ax = ligand_effect_magnitude(adata, show=False)

    plotted = ax.collections[0].get_array()
    raw = np.linalg.norm(np.asarray(adata.obsm["X_ligand_velocity"]), axis=1)
    expected = knn_smooth_array(adata, raw[:, None])[:, 0]
    np.testing.assert_allclose(plotted, expected)


def test_ligand_effect_magnitude_can_disable_smoothing(adata):
    adata.obsm["X_ligand_velocity"] = np.column_stack(
        [np.arange(adata.n_obs, dtype=float), np.zeros(adata.n_obs, dtype=float)]
    )

    ax = ligand_effect_magnitude(adata, show=False, smooth=False)

    plotted = ax.collections[0].get_array()
    expected = np.linalg.norm(np.asarray(adata.obsm["X_ligand_velocity"]), axis=1)
    np.testing.assert_allclose(plotted, expected)


def test_workflow_diagnostics_returns_axes(adata, prior_network, grn_network):
    axes = workflow_diagnostics(
        adata=adata,
        ligand="Gene0",
        prior_network=prior_network,
        grn_network=grn_network,
        basis="umap",
        n_neighbors=10,
        n_iter=2,
        damping=0.8,
        show=False,
    )
    assert axes.shape == (2, 3)
    assert axes[1, 2].get_title() == "6) Velocity field"
    assert axes[1, 1].get_title() == "5) Effect size"


def test_workflow_diagnostics_handles_missing_ligand_targets(adata, prior_network, grn_network):
    prior_no_overlap = pd.DataFrame(
        {
            "source": ["Gene0", "Gene0"],
            "target": ["MissingGeneA", "MissingGeneB"],
            "weight": [0.4, 0.9],
            "interaction_type": ["ligand_target", "ligand_target"],
        }
    )

    axes = workflow_diagnostics(
        adata=adata,
        ligand="Gene0",
        prior_network=prior_no_overlap,
        grn_network=grn_network,
        basis="umap",
        n_neighbors=10,
        n_iter=2,
        damping=0.8,
        show=False,
    )
    assert axes[0, 0].get_title() == "1) Initial perturbation"
    assert any(text.get_text() == "No prior targets found" for text in axes[0, 0].texts)
    assert len(axes[0, 0].get_xticks()) == 0
    assert len(axes[0, 0].get_yticks()) == 0
