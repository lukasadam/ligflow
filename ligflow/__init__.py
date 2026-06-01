"""
ligflow – Ligand-induced cell-state transition simulation for single-cell data.

Public API
----------
Core functions::

    lf.load_prior(path)
    lf.load_grn(path)
    lf.run_ligand_flow(adata, ligand, prior_network, grn_network, ...)

Plotting (``lf.pl``)::

    lf.pl.velocity_embedding(adata, basis="umap")
    lf.pl.ligand_effect_magnitude(adata)
    lf.pl.top_target_genes(adata)
    lf.pl.workflow_diagnostics(adata, ligand, prior_network, grn_network)
"""

from __future__ import annotations

from . import plotting as pl
from .priors import load_prior, network_to_adjacency, subset_by_ligand
from .workflow import load_grn, run_ligand_flow

__all__ = [
    # high-level workflow
    "run_ligand_flow",
    # data loading
    "load_prior",
    "load_grn",
    # priors helpers
    "subset_by_ligand",
    "network_to_adjacency",
    # plotting sub-module
    "pl",
]
