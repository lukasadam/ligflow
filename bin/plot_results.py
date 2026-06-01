#!/usr/bin/env python
"""
plot_results.py
---------------
CLI wrapper for ligflow plotting functions.

Reads a result AnnData h5ad file (output of run_ligand_flow.py) and generates
diagnostic PNG plots.

Usage
-----
plot_results.py \\
    --input  result.h5ad \\
    --ligand CXCL12 \\
    --prior  nichenet_prior.csv \\
    --grn    nichenet_grn.csv \\
    [--basis umap] \\
    [--n-genes 20] \\
    [--prefix my_analysis]
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import scanpy as sc
import ligflow as lf


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate diagnostic plots from a ligflow result h5ad file."
    )
    p.add_argument("--input", required=True, help="Input result h5ad file.")
    p.add_argument("--ligand", required=True, help="Ligand name (or comma-separated list).")
    p.add_argument(
        "--prior",
        required=True,
        help="Prior ligand-target network (CSV/TSV) used in the analysis.",
    )
    p.add_argument(
        "--grn",
        required=True,
        help="Gene regulatory network (CSV/TSV) used in the analysis.",
    )
    p.add_argument(
        "--basis", default="umap", help="Embedding basis for plots (default: umap)."
    )
    p.add_argument(
        "--color",
        default=None,
        help="Column in adata.obs to colour velocity plot (e.g. cell_type).",
    )
    p.add_argument(
        "--n-genes",
        type=int,
        default=20,
        help="Number of top target genes to show (default: 20).",
    )
    p.add_argument(
        "--prefix",
        default="ligflow",
        help="Output filename prefix (default: ligflow).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    ligand = [l.strip() for l in args.ligand.split(",")]
    if len(ligand) == 1:
        ligand = ligand[0]

    print(f"[plot_results] Reading {args.input}", flush=True)
    adata = sc.read_h5ad(args.input)

    print(f"[plot_results] Loading prior network: {args.prior}", flush=True)
    prior_network = lf.load_prior(args.prior)

    print(f"[plot_results] Loading GRN: {args.grn}", flush=True)
    grn_network = lf.load_grn(args.grn)

    velocity_plot = f"{args.prefix}_velocity.png"
    magnitude_plot = f"{args.prefix}_magnitude.png"
    top_genes_plot = f"{args.prefix}_top_genes.png"
    diagnostics_plot = f"{args.prefix}_diagnostics.png"

    print(f"[plot_results] Saving velocity plot → {velocity_plot}", flush=True)
    lf.pl.velocity_embedding(
        adata,
        basis=args.basis,
        color=args.color,
        show=False,
        save=velocity_plot,
    )

    print(f"[plot_results] Saving magnitude plot → {magnitude_plot}", flush=True)
    lf.pl.ligand_effect_magnitude(adata, show=False, save=magnitude_plot)

    print(f"[plot_results] Saving top-genes plot → {top_genes_plot}", flush=True)
    lf.pl.top_target_genes(adata, n_genes=args.n_genes, show=False, save=top_genes_plot)

    print(f"[plot_results] Saving diagnostics plot → {diagnostics_plot}", flush=True)
    lf.pl.workflow_diagnostics(
        adata,
        ligand=ligand,
        prior_network=prior_network,
        grn_network=grn_network,
        basis=args.basis,
        show=False,
        save=diagnostics_plot,
    )

    print("[plot_results] Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
