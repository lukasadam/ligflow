"""Integration tests for ligflow.workflow.run_ligand_flow"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import ligflow as lf


def test_run_ligand_flow_inplace(adata, prior_network, grn_network):
    """run_ligand_flow should mutate adata in-place and return None."""
    import anndata as ad
    a = adata.copy()
    result = lf.run_ligand_flow(
        a,
        ligand="Gene0",
        prior_network=prior_network,
        grn_network=grn_network,
        n_neighbors=10,
        n_iter=2,
    )
    assert result is None
    assert "ligand_shift" in a.layers
    assert "X_ligand_velocity" in a.obsm
    assert "ligand_transition_probs" in a.uns
    assert "ligand_metadata" in a.uns


def test_run_ligand_flow_copy(adata, prior_network, grn_network):
    """copy=True should return a new AnnData and not touch the original."""
    original_keys = set(adata.layers.keys())
    result = lf.run_ligand_flow(
        adata,
        ligand="Gene0",
        prior_network=prior_network,
        grn_network=grn_network,
        n_neighbors=10,
        n_iter=2,
        copy=True,
    )
    assert result is not None
    assert "ligand_shift" in result.layers
    # Original should be unmodified
    assert set(adata.layers.keys()) == original_keys


def test_run_ligand_flow_ligand_shift_shape(adata, prior_network, grn_network):
    a = adata.copy()
    lf.run_ligand_flow(
        a,
        ligand="Gene0",
        prior_network=prior_network,
        grn_network=grn_network,
        n_neighbors=10,
        n_iter=2,
    )
    assert a.layers["ligand_shift"].shape == (a.n_obs, a.n_vars)


def test_run_ligand_flow_velocity_shape(adata, prior_network, grn_network):
    a = adata.copy()
    lf.run_ligand_flow(
        a,
        ligand="Gene0",
        prior_network=prior_network,
        grn_network=grn_network,
        n_neighbors=10,
        n_iter=2,
    )
    assert a.obsm["X_ligand_velocity"].shape == (a.n_obs, 2)


def test_run_ligand_flow_metadata(adata, prior_network, grn_network):
    a = adata.copy()
    lf.run_ligand_flow(
        a,
        ligand="Gene0",
        prior_network=prior_network,
        grn_network=grn_network,
        n_neighbors=10,
        n_iter=2,
        damping=0.7,
    )
    meta = a.uns["ligand_metadata"]
    assert meta["n_iter"] == 2
    assert meta["damping"] == 0.7
    assert "Gene0" in meta["ligand"]


def test_run_ligand_flow_bad_embedding(adata, prior_network, grn_network):
    with pytest.raises(KeyError, match="X_bad"):
        lf.run_ligand_flow(
            adata,
            ligand="Gene0",
            prior_network=prior_network,
            grn_network=grn_network,
            embedding_key="X_bad",
        )


def test_run_ligand_flow_bad_ligand(adata, prior_network, grn_network):
    with pytest.raises(ValueError, match="None of the supplied ligands"):
        lf.run_ligand_flow(
            adata,
            ligand="UNKNOWN_LIGAND",
            prior_network=prior_network,
            grn_network=grn_network,
        )
