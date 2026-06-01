"""Tests for ligflow.vectorfield"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ligflow.vectorfield import transition_to_vectors


def test_vectorfield_shape(adata):
    n = adata.n_obs
    # Identity-like transition (self-loops) → zero vectors
    T = sp.eye(n, format="csr")
    vectors = transition_to_vectors(adata, T, embedding_key="X_umap")
    assert vectors.shape == (n, 2)


def test_vectorfield_identity_gives_zeros(adata):
    """A self-loop transition matrix should produce zero displacement."""
    n = adata.n_obs
    T = sp.eye(n, format="csr")
    vectors = transition_to_vectors(adata, T, embedding_key="X_umap")
    np.testing.assert_allclose(vectors, 0.0, atol=1e-6)


def test_vectorfield_missing_embedding(adata):
    n = adata.n_obs
    T = sp.eye(n, format="csr")
    with pytest.raises(KeyError, match="X_pca"):
        transition_to_vectors(adata, T, embedding_key="X_pca")


def test_vectorfield_non_zero(adata):
    """A uniform transition to neighbour 1 should give non-zero vectors."""
    rng = np.random.default_rng(9)
    n = adata.n_obs
    # Transition: each cell goes to the next cell with probability 1
    row = np.arange(n)
    col = (np.arange(n) + 1) % n
    data = np.ones(n)
    T = sp.csr_matrix((data, (row, col)), shape=(n, n))
    vectors = transition_to_vectors(adata, T, embedding_key="X_umap")
    assert np.linalg.norm(vectors) > 0


def test_vectorfield_effect_size_scales_lengths(adata):
    """effect_size should proportionally scale per-cell vector magnitudes."""
    n = adata.n_obs
    row = np.arange(n)
    col = (np.arange(n) + 1) % n
    data = np.ones(n)
    T = sp.csr_matrix((data, (row, col)), shape=(n, n))

    base = transition_to_vectors(adata, T, embedding_key="X_umap")
    effect = np.ones(n, dtype=float)
    effect[: n // 2] = 0.25
    scaled = transition_to_vectors(
        adata, T, embedding_key="X_umap", effect_size=effect
    )

    base_norm = np.linalg.norm(base, axis=1)
    scaled_norm = np.linalg.norm(scaled, axis=1)
    nonzero = base_norm > 0
    np.testing.assert_allclose(
        scaled_norm[nonzero], base_norm[nonzero] * effect[nonzero], rtol=1e-6
    )
