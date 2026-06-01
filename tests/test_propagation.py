"""Tests for ligflow.propagation"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ligflow.propagation import build_initial_delta, propagate_signal


# ── build_initial_delta ───────────────────────────────────────────────────────


def test_build_initial_delta_shape(prior_network, gene_names):
    delta = build_initial_delta("Gene0", prior_network, gene_names)
    assert delta.shape == (len(gene_names),)


def test_build_initial_delta_nonzero(prior_network, gene_names):
    delta = build_initial_delta("Gene0", prior_network, gene_names)
    assert delta.sum() > 0


def test_build_initial_delta_unknown_ligand(prior_network, gene_names):
    """Unknown ligand → all-zero delta (no error)."""
    delta = build_initial_delta("UNKNOWN", prior_network, gene_names)
    assert delta.sum() == 0


def test_build_initial_delta_mean_aggregation(prior_network, gene_names):
    delta_sum = build_initial_delta("Gene0", prior_network, gene_names, aggregation="sum")
    delta_mean = build_initial_delta("Gene0", prior_network, gene_names, aggregation="mean")
    # sum >= mean (weights are positive)
    assert np.all(delta_sum >= delta_mean - 1e-9)


# ── propagate_signal ─────────────────────────────────────────────────────────


@pytest.fixture
def simple_inputs(gene_names):
    n_cells = 20
    n_genes = len(gene_names)
    rng = np.random.default_rng(7)
    expression = rng.random((n_cells, n_genes))
    delta = rng.random(n_genes)
    grn = sp.random(n_genes, n_genes, density=0.1, format="csr", random_state=7)
    return expression, delta, grn


def test_propagate_signal_shape(simple_inputs):
    expression, delta, grn = simple_inputs
    result = propagate_signal(expression, delta, grn, n_iter=3)
    assert result.shape == expression.shape


def test_propagate_signal_zero_iter(simple_inputs):
    """With n_iter=0 the result should equal the initial_delta broadcast."""
    expression, delta, grn = simple_inputs
    result = propagate_signal(expression, delta, grn, n_iter=0)
    # All rows should equal delta
    for row in result:
        np.testing.assert_allclose(row, delta)


def test_propagate_signal_dense_grn(simple_inputs):
    expression, delta, grn = simple_inputs
    grn_dense = grn.toarray()
    result = propagate_signal(expression, delta, grn_dense, n_iter=2)
    assert result.shape == expression.shape


def test_propagate_signal_invalid_delta(simple_inputs):
    expression, delta, grn = simple_inputs
    with pytest.raises(ValueError, match="initial_delta"):
        propagate_signal(expression, delta[:5], grn, n_iter=1)


def test_propagate_signal_invalid_grn(simple_inputs):
    expression, delta, grn = simple_inputs
    bad_grn = sp.eye(5, format="csr")
    with pytest.raises(ValueError, match="grn_matrix"):
        propagate_signal(expression, delta, bad_grn, n_iter=1)


def test_propagate_signal_invalid_damping(simple_inputs):
    expression, delta, grn = simple_inputs
    with pytest.raises(ValueError, match="damping"):
        propagate_signal(expression, delta, grn, damping=1.5)
