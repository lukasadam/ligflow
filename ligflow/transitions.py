"""
transitions.py
--------------
Compute local cell-cell transition probabilities based on predicted
post-perturbation expression states.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def compute_transition_probabilities(
    adata: AnnData,
    expression: np.ndarray,
    propagated_delta: np.ndarray,
    n_neighbors: int = 30,
    kernel_sigma: Optional[float] = None,
    use_existing_neighbors: bool = True,
) -> sp.csr_matrix:
    """Compute sparse cell-cell transition probabilities after perturbation.

    For each cell *i* the *predicted* post-perturbation state is

    .. math::

        \\hat{x}_i = x_i + \\delta_i

    where :math:`x_i` is its (smoothed) expression and :math:`\\delta_i` is
    the propagated perturbation.  Transition probability from cell *i* to
    neighbour *j* is proportional to a Gaussian kernel evaluated on the
    distance between :math:`\\hat{x}_i` and :math:`x_j`:

    .. math::

        p_{ij} \\propto \\exp\\!\\left(-\\frac{\\|\\hat{x}_i - x_j\\|^2}{2\\sigma^2}\\right)

    Parameters
    ----------
    adata:
        Annotated data matrix.
    expression:
        Smoothed expression matrix of shape ``(n_cells, n_genes)``.
    propagated_delta:
        Propagated perturbation matrix of shape ``(n_cells, n_genes)`` as
        returned by :func:`~ligflow.propagation.propagate_signal`.
    n_neighbors:
        Number of neighbours to consider (used only when no existing graph is
        available).
    kernel_sigma:
        Bandwidth of the Gaussian kernel.  Defaults to the median pairwise
        distance between predicted and neighbour states.
    use_existing_neighbors:
        If *True* (default) and ``adata.obsp["connectivities"]`` is present,
        use that neighbour graph.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-stochastic sparse matrix of shape ``(n_cells, n_cells)``.
    """
    n_cells = expression.shape[0]

    # ── 1. Get neighbour indices ─────────────────────────────────────────────
    neighbor_indices = _get_neighbor_indices(adata, n_neighbors, use_existing_neighbors)

    # ── 2. Predicted states ──────────────────────────────────────────────────
    predicted = expression + propagated_delta  # (n_cells, n_genes)

    # ── 3. Estimate sigma if not provided ───────────────────────────────────
    if kernel_sigma is None:
        kernel_sigma = _estimate_sigma(predicted, expression, neighbor_indices)

    sigma2 = kernel_sigma ** 2

    # ── 4. Build sparse transition matrix ───────────────────────────────────
    rows, cols, data = [], [], []
    for i in range(n_cells):
        nbrs = neighbor_indices[i]
        if len(nbrs) == 0:
            continue
        diffs = expression[nbrs] - predicted[i]  # (k, n_genes)
        sq_dists = np.sum(diffs ** 2, axis=1)     # (k,)
        weights = np.exp(-sq_dists / (2.0 * sigma2))
        total = weights.sum()
        if total == 0:
            weights = np.ones_like(weights) / len(nbrs)
        else:
            weights = weights / total
        rows.extend([i] * len(nbrs))
        cols.extend(nbrs.tolist())
        data.extend(weights.tolist())

    T = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    return T


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_neighbor_indices(
    adata: AnnData,
    n_neighbors: int,
    use_existing: bool,
) -> list[np.ndarray]:
    """Return a list of neighbour index arrays, one per cell."""
    if use_existing and "connectivities" in (adata.obsp or {}):
        conn = adata.obsp["connectivities"]
        if not sp.issparse(conn):
            conn = sp.csr_matrix(conn)
        return [conn[i].indices for i in range(adata.n_obs)]

    # Compute from PCA or raw expression
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
    _, indices = nn.kneighbors(coords)
    return [indices[i] for i in range(adata.n_obs)]


def _estimate_sigma(
    predicted: np.ndarray,
    expression: np.ndarray,
    neighbor_indices: list[np.ndarray],
    max_sample: int = 500,
) -> float:
    """Estimate kernel bandwidth as the median pairwise distance."""
    n_cells = predicted.shape[0]
    sample = np.random.choice(n_cells, size=min(max_sample, n_cells), replace=False)
    dists = []
    for i in sample:
        nbrs = neighbor_indices[i]
        if len(nbrs) == 0:
            continue
        diff = expression[nbrs] - predicted[i]
        sq = np.sum(diff ** 2, axis=1)
        dists.extend(np.sqrt(sq).tolist())
    if len(dists) == 0:
        return 1.0
    sigma = float(np.median(dists))
    return sigma if sigma > 0 else 1.0
