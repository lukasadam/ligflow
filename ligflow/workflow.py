"""
workflow.py
-----------
High-level workflow that orchestrates the full ligflow pipeline.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .imputation import knn_smooth
from .propagation import build_initial_delta, propagate_signal
from .priors import network_to_adjacency, subset_by_ligand
from .transitions import compute_transition_probabilities
from .vectorfield import transition_to_vectors


def run_ligand_flow(
    adata: AnnData,
    ligand: Union[str, list[str]],
    prior_network: pd.DataFrame,
    grn_network: pd.DataFrame,
    expression_layer: Optional[str] = None,
    embedding_key: str = "X_umap",
    n_neighbors: int = 30,
    n_iter: int = 3,
    damping: float = 0.8,
    kernel_sigma: Optional[float] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """Run the full ligand-flow pipeline on an AnnData object.

    The function executes the following steps in order:

    1. Smooth / impute expression with kNN averaging.
    2. Build the initial perturbation vector from *prior_network*.
    3. Propagate the perturbation through *grn_network* for *n_iter* steps.
    4. Compute cell-cell transition probabilities via a Gaussian kernel.
    5. Convert the transition matrix into velocity vectors in embedding space.

    Results are stored in *adata* (or a copy if ``copy=True``):

    * ``adata.layers["ligand_shift"]``  – per-cell gene-expression shift.
    * ``adata.obsm["X_ligand_velocity"]`` – velocity vectors in embedding space.
    * ``adata.uns["ligand_transition_probs"]``  – sparse transition matrix
      serialised as a dict of arrays (``data``, ``indices``, ``indptr``,
      ``shape``).
    * ``adata.uns["ligand_metadata"]``  – run parameters.

    Parameters
    ----------
    adata:
        Annotated data matrix with cells on rows and genes on columns.
    ligand:
        Name(s) of the ligand(s) to perturb.
    prior_network:
        Prior ligand-target network as returned by
        :func:`~ligflow.priors.load_prior`.
    grn_network:
        Gene regulatory network in the same format (``source``, ``target``,
        ``weight``).
    expression_layer:
        Layer of *adata* to use as expression input.  ``None`` uses
        ``adata.X``.
    embedding_key:
        Key in ``adata.obsm`` for the 2-D embedding used for the vector field.
    n_neighbors:
        Number of nearest neighbours for kNN smoothing and transition
        probability estimation.
    n_iter:
        Number of GRN propagation iterations.
    damping:
        Damping factor for GRN propagation (0–1).
    kernel_sigma:
        Bandwidth of the Gaussian kernel for transition probabilities.
        Estimated automatically when *None*.
    copy:
        If *True* work on a copy of *adata* and return it; otherwise mutate
        *adata* in-place and return *None*.

    Returns
    -------
    AnnData or None
        Modified *AnnData* when ``copy=True``, otherwise *None*.

    Raises
    ------
    KeyError
        If *embedding_key* is not present in ``adata.obsm``.
    ValueError
        If the ligand is not found in *prior_network*.
    """
    # ── Input validation ─────────────────────────────────────────────────────
    if not isinstance(adata, AnnData):
        raise TypeError(f"adata must be an AnnData object, got {type(adata)}")
    if embedding_key not in adata.obsm:
        raise KeyError(
            f"Embedding key '{embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    if not (0.0 <= damping <= 1.0):
        raise ValueError(f"damping must be in [0, 1], got {damping}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be non-negative, got {n_iter}")

    if copy:
        adata = adata.copy()

    gene_names = list(adata.var_names)

    # ── Step 1: Smooth expression ────────────────────────────────────────────
    X_smooth = knn_smooth(
        adata,
        layer=expression_layer,
        n_neighbors=n_neighbors,
        use_existing_neighbors=True,
    )

    # ── Step 2: Build initial perturbation vector ────────────────────────────
    ligand_subnet = subset_by_ligand(prior_network, ligand)
    initial_delta = build_initial_delta(
        ligand=ligand,
        prior_network=ligand_subnet,
        gene_names=gene_names,
    )

    # ── Step 3: Build GRN matrix & propagate ────────────────────────────────
    grn_matrix = network_to_adjacency(grn_network, gene_names)
    propagated = propagate_signal(
        expression=X_smooth,
        initial_delta=initial_delta,
        grn_matrix=grn_matrix,
        n_iter=n_iter,
        damping=damping,
    )

    # ── Step 4: Transition probabilities ────────────────────────────────────
    T = compute_transition_probabilities(
        adata=adata,
        expression=X_smooth,
        propagated_delta=propagated,
        n_neighbors=n_neighbors,
        kernel_sigma=kernel_sigma,
        use_existing_neighbors=True,
    )

    # ── Step 5: Vector field ─────────────────────────────────────────────────
    vectors = transition_to_vectors(
        adata=adata,
        transition_matrix=T,
        embedding_key=embedding_key,
    )

    # ── Store results ────────────────────────────────────────────────────────
    adata.layers["ligand_shift"] = propagated
    adata.obsm["X_ligand_velocity"] = vectors

    # Serialise sparse transition matrix for storage in uns
    adata.uns["ligand_transition_probs"] = {
        "data": T.data,
        "indices": T.indices,
        "indptr": T.indptr,
        "shape": list(T.shape),
    }

    adata.uns["ligand_metadata"] = {
        "ligand": ligand if isinstance(ligand, list) else [ligand],
        "n_iter": n_iter,
        "damping": damping,
        "n_neighbors": n_neighbors,
        "kernel_sigma": kernel_sigma,
        "embedding_key": embedding_key,
        "expression_layer": expression_layer,
    }

    if copy:
        return adata
    return None


def load_grn(
    path: Union[str, "pathlib.Path"],  # noqa: F821
    sep: Optional[str] = None,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Load a gene regulatory network from a CSV/TSV file.

    This is a thin convenience wrapper around
    :func:`~ligflow.priors.load_prior` tailored for GRN files which may
    omit the ``interaction_type`` column.

    Parameters
    ----------
    path:
        Path to the file.
    sep:
        Column separator (inferred from extension when *None*).
    source_col, target_col, weight_col:
        Column names for source gene, target gene and edge weight.

    Returns
    -------
    pd.DataFrame
        Data-frame with columns ``[source, target, weight, interaction_type]``.
    """
    from .priors import load_prior

    return load_prior(
        path=path,
        sep=sep,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
    )
