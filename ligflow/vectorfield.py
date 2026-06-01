"""
vectorfield.py
--------------
Convert a cell-cell transition matrix into velocity vectors in embedding space.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def transition_to_vectors(
    adata: AnnData,
    transition_matrix: sp.csr_matrix,
    embedding_key: str = "X_umap",
) -> np.ndarray:
    """Compute velocity vectors from a transition probability matrix.

    For each cell *i* the vector is the probability-weighted average of the
    displacements to its transition targets:

    .. math::

        v_i = \\sum_j p_{ij} \\,(e_j - e_i)

    where :math:`e_i` is the embedding coordinate of cell *i* and
    :math:`p_{ij}` is the transition probability from *i* to *j*.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    transition_matrix:
        Row-stochastic sparse transition matrix of shape
        ``(n_cells, n_cells)``.
    embedding_key:
        Key in ``adata.obsm`` that holds the 2-D (or higher-dimensional)
        embedding coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_cells, n_dims)`` containing the velocity vectors
        in embedding space.

    Raises
    ------
    KeyError
        If *embedding_key* is not found in ``adata.obsm``.
    """
    if embedding_key not in adata.obsm:
        raise KeyError(
            f"Embedding key '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    embedding = np.array(adata.obsm[embedding_key])  # (n_cells, n_dims)
    n_cells, n_dims = embedding.shape

    if not sp.issparse(transition_matrix):
        transition_matrix = sp.csr_matrix(transition_matrix)

    # Weighted average of neighbour coordinates
    # T @ embedding  gives the weighted average target position for each cell.
    weighted_targets = np.array(transition_matrix @ embedding)  # (n_cells, n_dims)

    # Compute how many non-zero entries per row to handle cells with no targets
    row_sums = np.asarray(transition_matrix.sum(axis=1)).ravel()
    has_targets = row_sums > 0

    vectors = np.zeros((n_cells, n_dims), dtype=np.float64)
    vectors[has_targets] = weighted_targets[has_targets] - embedding[has_targets]

    return vectors
