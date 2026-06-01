"""Tests for ligflow.imputation"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ligflow.imputation import knn_smooth


def test_knn_smooth_output_shape(adata):
    X_smooth = knn_smooth(adata, n_neighbors=5)
    assert X_smooth.shape == (adata.n_obs, adata.n_vars)


def test_knn_smooth_values_change(adata):
    X_smooth = knn_smooth(adata, n_neighbors=5)
    X_raw = np.array(adata.X)
    # Smoothed values should differ from raw (not identical)
    assert not np.allclose(X_smooth, X_raw)


def test_knn_smooth_non_negative(adata):
    """Smoothed values should stay non-negative for non-negative input."""
    X_smooth = knn_smooth(adata, n_neighbors=5)
    assert np.all(X_smooth >= 0)


def test_knn_smooth_with_layer(adata):
    import anndata as ad
    a = adata.copy()
    a.layers["counts"] = a.X.copy()
    X_smooth = knn_smooth(a, layer="counts", n_neighbors=5)
    assert X_smooth.shape == (a.n_obs, a.n_vars)


def test_knn_smooth_uses_existing_neighbors(adata):
    """Should not raise when connectivities already present."""
    import anndata as ad
    import scipy.sparse as sp
    a = adata.copy()
    n = a.n_obs
    # Add a trivial diagonal connectivities matrix
    a.obsp["connectivities"] = sp.eye(n, format="csr")
    X_smooth = knn_smooth(a, use_existing_neighbors=True)
    assert X_smooth.shape == (n, a.n_vars)
