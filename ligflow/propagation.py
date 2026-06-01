"""
propagation.py
--------------
GRN-based signal propagation of ligand perturbations.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sp


def build_initial_delta(
    ligand: Union[str, list[str]],
    prior_network: "pd.DataFrame",  # noqa: F821
    gene_names: list[str],
    aggregation: str = "sum",
) -> np.ndarray:
    """Build the initial perturbation vector from a prior network.

    For each gene that is a *direct* target of the ligand(s) in the prior
    network, set the perturbation magnitude to the sum (or mean) of the
    corresponding edge weights.

    Parameters
    ----------
    ligand:
        One or more ligand names.
    prior_network:
        Prior network data-frame (``source``, ``target``, ``weight``).
    gene_names:
        Ordered list of gene names (typically ``adata.var_names``).
    aggregation:
        How to combine weights when a gene has multiple edges from the
        same ligand: ``"sum"`` (default) or ``"mean"``.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(n_genes,)`` with the initial perturbation
        magnitudes.
    """
    import pandas as pd

    if isinstance(ligand, str):
        ligand = [ligand]

    gene_index = {g: i for i, g in enumerate(gene_names)}
    delta = np.zeros(len(gene_names), dtype=np.float64)

    subset = prior_network[prior_network["source"].isin(set(ligand))]
    if subset.empty:
        return delta

    grouped = subset.groupby("target")["weight"]
    if aggregation == "mean":
        agg_weights = grouped.mean()
    else:
        agg_weights = grouped.sum()

    for gene, w in agg_weights.items():
        if gene in gene_index:
            delta[gene_index[gene]] = w

    return delta


def propagate_signal(
    expression: np.ndarray,
    initial_delta: np.ndarray,
    grn_matrix: Union[np.ndarray, sp.spmatrix],
    n_iter: int = 3,
    damping: float = 0.8,
) -> np.ndarray:
    """Propagate a ligand-induced perturbation through a gene regulatory network.

    The update rule is:

    .. math::

        \\delta_{t+1} = \\alpha \\cdot G \\, \\delta_t + \\delta_0

    where :math:`G` is the GRN adjacency matrix (column-normalised),
    :math:`\\delta_0` is the initial perturbation, and :math:`\\alpha` is the
    damping factor.

    Parameters
    ----------
    expression:
        Cell × gene expression matrix of shape ``(n_cells, n_genes)``.
        Used only to determine the number of cells; the perturbation is
        assumed to be the same across all cells (broadcast from
        *initial_delta*).
    initial_delta:
        1-D array of shape ``(n_genes,)`` with the initial perturbation.
    grn_matrix:
        Square matrix of shape ``(n_genes, n_genes)`` representing the GRN.
        ``grn_matrix[i, j]`` is the weight of the edge
        ``gene_j → gene_i``.  May be a dense or sparse matrix.
    n_iter:
        Number of propagation iterations (default 3).
    damping:
        Damping / decay factor :math:`\\alpha` (default 0.8).

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_cells, n_genes)`` with the propagated
        perturbation for each cell.  Each row is identical because the
        initial perturbation is uniform across cells.

    Raises
    ------
    ValueError
        If the dimensions of the inputs are inconsistent.
    """
    n_cells, n_genes = expression.shape
    if initial_delta.shape != (n_genes,):
        raise ValueError(
            f"initial_delta must have shape ({n_genes},), "
            f"got {initial_delta.shape}"
        )
    if grn_matrix.shape != (n_genes, n_genes):
        raise ValueError(
            f"grn_matrix must have shape ({n_genes}, {n_genes}), "
            f"got {grn_matrix.shape}"
        )
    if n_iter < 0:
        raise ValueError(f"n_iter must be non-negative, got {n_iter}")
    if not (0.0 <= damping <= 1.0):
        raise ValueError(f"damping must be in [0, 1], got {damping}")

    # Column-normalise the GRN matrix so that it does not explode
    G = _column_normalise(grn_matrix)

    delta = initial_delta.copy().astype(np.float64)
    for _ in range(n_iter):
        if sp.issparse(G):
            delta = damping * np.asarray(G @ delta).ravel() + initial_delta
        else:
            delta = damping * G @ delta + initial_delta

    # Broadcast to all cells
    propagated = np.tile(delta, (n_cells, 1))
    return propagated


# ── Internal helpers ──────────────────────────────────────────────────────────


def _column_normalise(
    matrix: Union[np.ndarray, sp.spmatrix],
) -> Union[np.ndarray, sp.spmatrix]:
    """Return a column-normalised copy of *matrix* (L1 norm per column)."""
    if sp.issparse(matrix):
        col_sums = np.asarray(np.abs(matrix).sum(axis=0)).ravel()
        col_sums[col_sums == 0] = 1.0
        norm = sp.diags(1.0 / col_sums, format="csr")
        return matrix @ norm
    else:
        col_sums = np.abs(matrix).sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        return matrix / col_sums[np.newaxis, :]
