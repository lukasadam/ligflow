"""Tests for ligflow.plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from ligflow.plotting import workflow_diagnostics


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
