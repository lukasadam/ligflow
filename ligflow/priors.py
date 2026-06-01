"""
priors.py
---------
Functions for loading and manipulating ligand-target prior networks.

The expected file format is a tab/comma-separated table with at least:
    source, target, weight, interaction_type
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_prior(
    path: Union[str, Path],
    sep: Optional[str] = None,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    interaction_type_col: str = "interaction_type",
) -> pd.DataFrame:
    """Load a ligand-target prior network from a CSV or TSV file.

    Parameters
    ----------
    path:
        Path to the CSV/TSV file.
    sep:
        Column separator.  Inferred from the file extension when *None*:
        ``.tsv`` / ``.txt`` → ``"\\t"``, everything else → ``","``.
    source_col:
        Column name for source nodes (ligands / upstream genes).
    target_col:
        Column name for target nodes (downstream genes).
    weight_col:
        Column name for edge weights.
    interaction_type_col:
        Column name for interaction type annotation.

    Returns
    -------
    pd.DataFrame
        Data-frame with columns ``[source, target, weight, interaction_type]``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prior network file not found: {path}")

    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","

    df = pd.read_csv(path, sep=sep)

    required = {source_col, target_col, weight_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing from prior network: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    rename = {}
    if source_col != "source":
        rename[source_col] = "source"
    if target_col != "target":
        rename[target_col] = "target"
    if weight_col != "weight":
        rename[weight_col] = "weight"
    if interaction_type_col in df.columns and interaction_type_col != "interaction_type":
        rename[interaction_type_col] = "interaction_type"

    if rename:
        df = df.rename(columns=rename)

    if "interaction_type" not in df.columns:
        df["interaction_type"] = "unknown"

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"])

    return df[["source", "target", "weight", "interaction_type"]].reset_index(drop=True)


def subset_by_ligand(
    prior: pd.DataFrame,
    ligand: Union[str, list[str]],
) -> pd.DataFrame:
    """Return the sub-network reachable from a given ligand (or list of ligands).

    Parameters
    ----------
    prior:
        Prior network as returned by :func:`load_prior`.
    ligand:
        One or more ligand names.  These are matched against the ``source``
        column.

    Returns
    -------
    pd.DataFrame
        Filtered prior network containing only rows whose ``source`` is in
        *ligand*.

    Raises
    ------
    ValueError
        If none of the supplied ligands are present in the network.
    """
    if isinstance(ligand, str):
        ligand = [ligand]

    ligand_set = set(ligand)
    subset = prior[prior["source"].isin(ligand_set)]

    if subset.empty:
        available = sorted(prior["source"].unique().tolist())
        raise ValueError(
            f"None of the supplied ligands {ligand} were found as source nodes "
            f"in the prior network.  Available sources (first 20): {available[:20]}"
        )

    return subset.reset_index(drop=True)


def network_to_adjacency(
    network: pd.DataFrame,
    gene_names: list[str],
    use_abs_weight: bool = True,
) -> sp.csr_matrix:
    """Convert a prior network data-frame to a sparse adjacency matrix.

    The matrix A is indexed by *gene_names* so that ``A[i, j]`` is the weight
    of the directed edge ``source=gene_names[j] → target=gene_names[i]``.

    Parameters
    ----------
    network:
        Prior network data-frame with columns ``source``, ``target``,
        ``weight``.
    gene_names:
        Ordered list of gene names (typically ``adata.var_names``).
    use_abs_weight:
        If *True* (default) use the absolute value of the weights.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix of shape ``(n_genes, n_genes)``.
    """
    gene_index = {g: i for i, g in enumerate(gene_names)}
    n = len(gene_names)

    rows, cols, data = [], [], []
    for _, row in network.iterrows():
        src, tgt, w = row["source"], row["target"], row["weight"]
        if src in gene_index and tgt in gene_index:
            j = gene_index[src]
            i = gene_index[tgt]
            rows.append(i)
            cols.append(j)
            data.append(abs(w) if use_abs_weight else w)

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return adj
