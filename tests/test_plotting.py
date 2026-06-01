"""Tests for ligflow.plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

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
    assert axes[1, 2].axison is False


def test_workflow_diagnostics_handles_missing_ligand_targets(adata, prior_network, grn_network):
    prior_no_overlap = prior_network.copy()
    prior_no_overlap["target"] = "MissingGene"

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
    assert axes[0, 1].get_title() == "2) Initial perturbation"
