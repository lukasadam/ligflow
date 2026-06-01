"""Tests for ligflow.transitions"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ligflow.transitions import compute_transition_probabilities


def test_transition_probs_shape(adata):
    n = adata.n_obs
    n_genes = adata.n_vars
    rng = np.random.default_rng(3)
    expr = rng.random((n, n_genes))
    delta = rng.random((n, n_genes)) * 0.1

    T = compute_transition_probabilities(
        adata, expr, delta, n_neighbors=10, use_existing_neighbors=False
    )
    assert T.shape == (n, n)


def test_transition_probs_row_stochastic(adata):
    n = adata.n_obs
    n_genes = adata.n_vars
    rng = np.random.default_rng(4)
    expr = rng.random((n, n_genes))
    delta = rng.random((n, n_genes)) * 0.1

    T = compute_transition_probabilities(
        adata, expr, delta, n_neighbors=10, use_existing_neighbors=False
    )
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    # Each row should sum to ≈1
    np.testing.assert_allclose(row_sums, np.ones(n), atol=1e-8)


def test_transition_probs_nonnegative(adata):
    n = adata.n_obs
    n_genes = adata.n_vars
    rng = np.random.default_rng(5)
    expr = rng.random((n, n_genes))
    delta = rng.random((n, n_genes)) * 0.1

    T = compute_transition_probabilities(
        adata, expr, delta, n_neighbors=10, use_existing_neighbors=False
    )
    assert T.data.min() >= 0


def test_transition_probs_custom_sigma(adata):
    n = adata.n_obs
    n_genes = adata.n_vars
    rng = np.random.default_rng(6)
    expr = rng.random((n, n_genes))
    delta = np.zeros((n, n_genes))  # zero perturbation

    T = compute_transition_probabilities(
        adata,
        expr,
        delta,
        n_neighbors=5,
        kernel_sigma=1.0,
        use_existing_neighbors=False,
    )
    assert T.shape == (n, n)
