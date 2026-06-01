"""
imputation.py
-------------
kNN-based expression smoothing / imputation.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def knn_smooth(
    adata: AnnData,
    layer: Optional[str] = None,
    n_neighbors: int = 30,
    use_existing_neighbors: bool = True,
) -> np.ndarray:
    """Smooth gene expression using k-nearest neighbours.

    For each cell the smoothed expression is the (unweighted) mean of itself
    and its *k* nearest neighbours.

    Parameters
    ----------
    adata:
        Annotated data matrix.  Cells on rows, genes on columns.
    layer:
        If provided, smooth ``adata.layers[layer]``; otherwise smooth
        ``adata.X``.
    n_neighbors:
        Number of neighbours to use when computing the kNN graph from scratch.
        Ignored when *use_existing_neighbors* is *True* and a graph is already
        stored in ``adata.obsp``.
    use_existing_neighbors:
        If *True* (default) and ``adata.obsp["connectivities"]`` already
        exists, reuse that graph instead of recomputing it.

    Returns
    -------
    numpy.ndarray
        Dense smoothed expression matrix of shape ``(n_cells, n_genes)``.
    """
    # ── 1. Fetch raw expression ─────────────────────────────────────────────
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if sp.issparse(X):
        X = np.asarray(X.todense())
    else:
        X = np.array(X)

    return knn_smooth_array(
        adata=adata,
        values=X,
        n_neighbors=n_neighbors,
        use_existing_neighbors=use_existing_neighbors,
    )


def knn_smooth_array(
    adata: AnnData,
    values: np.ndarray,
    n_neighbors: int = 30,
    use_existing_neighbors: bool = True,
) -> np.ndarray:
    """Smooth a cell-by-feature array over the kNN graph.

    Parameters
    ----------
    adata:
        Annotated data matrix providing cell neighbourhood information.
    values:
        Dense array of shape ``(n_cells, n_features)`` to smooth.
    n_neighbors:
        Number of neighbours to use when computing the kNN graph from scratch.
    use_existing_neighbors:
        If *True* and ``adata.obsp["connectivities"]`` exists, reuse it.

    Returns
    -------
    numpy.ndarray
        Smoothed array of shape ``(n_cells, n_features)``.
    """
    X = np.asarray(values)
    if X.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {X.shape}")
    if X.shape[0] != adata.n_obs:
        raise ValueError(
            "values must have one row per cell: "
            f"expected {adata.n_obs}, got {X.shape[0]}"
        )

    # ── 1. Build / retrieve kNN connectivity matrix ─────────────────────────
    conn = _get_connectivities(adata, n_neighbors, use_existing_neighbors)

    # ── 2. Smooth: mean over self + neighbours ───────────────────────────────
    n_cells = X.shape[0]
    conn_with_self = conn + sp.eye(n_cells, format="csr")
    row_sums = np.asarray(conn_with_self.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    norm_factor = sp.diags(1.0 / row_sums, format="csr")
    W = norm_factor @ conn_with_self

    X_smooth = W @ X
    return np.asarray(X_smooth)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_connectivities(
    adata: AnnData,
    n_neighbors: int,
    use_existing: bool,
) -> sp.csr_matrix:
    """Return a binary/weighted kNN connectivity matrix (n_cells × n_cells)."""
    if use_existing and "connectivities" in (adata.obsp or {}):
        conn = adata.obsp["connectivities"]
        if sp.issparse(conn):
            return conn.astype(np.float64)
        return sp.csr_matrix(conn, dtype=np.float64)

    # Fall back to sklearn NearestNeighbors on the PCA embedding if available,
    # otherwise on raw expression.
    if "X_pca" in adata.obsm:
        coords = adata.obsm["X_pca"]
    else:
        X = adata.X
        if sp.issparse(X):
            coords = np.asarray(X.todense())
        else:
            coords = np.array(X)

    from sklearn.neighbors import NearestNeighbors

    k = min(n_neighbors, adata.n_obs - 1)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    n = adata.n_obs
    row_idx = np.repeat(np.arange(n), k)
    col_idx = indices.ravel()
    data = np.ones(len(row_idx), dtype=np.float64)
    conn = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    # Make symmetric
    conn = (conn + conn.T)
    conn.data = np.ones_like(conn.data)
    return conn
