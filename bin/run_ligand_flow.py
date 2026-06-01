#!/usr/bin/env python
"""
run_ligand_flow.py
------------------
CLI wrapper for ligflow.run_ligand_flow.

Reads an embedded AnnData h5ad file plus prior and GRN network CSVs, runs the
full ligand-flow pipeline, and writes the result to a new h5ad file.

Usage
-----
run_ligand_flow.py \\
    --input   cells_embedded.h5ad \\
    --prior   nichenet_prior.csv \\
    --grn     nichenet_grn.csv \\
    --ligand  CXCL12 \\
    --output  result.h5ad \\
    [--embedding-key X_umap] \\
    [--n-neighbors 30] \\
    [--n-iter 3] \\
    [--damping 0.8] \\
    [--expression-layer spliced]
"""

from __future__ import annotations

import argparse
import sys

import scanpy as sc
import ligflow as lf


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the ligand-flow pipeline on an embedded AnnData object."
    )
    p.add_argument("--input", required=True, help="Input h5ad file (must have embedding).")
    p.add_argument("--prior", required=True, help="Prior ligand-target network (CSV/TSV).")
    p.add_argument("--grn", required=True, help="Gene regulatory network (CSV/TSV).")
    p.add_argument(
        "--ligand",
        required=True,
        help="Ligand name(s). Multiple ligands separated by commas.",
    )
    p.add_argument("--output", required=True, help="Output h5ad file.")
    p.add_argument(
        "--embedding-key",
        default="X_umap",
        help="Key in adata.obsm for the embedding (default: X_umap).",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="k-NN neighbours for transition probabilities (default: 30).",
    )
    p.add_argument(
        "--n-iter", type=int, default=3, help="GRN propagation iterations (default: 3)."
    )
    p.add_argument(
        "--damping",
        type=float,
        default=0.8,
        help="GRN propagation damping factor 0–1 (default: 0.8).",
    )
    p.add_argument(
        "--expression-layer",
        default=None,
        help="AnnData layer to use as expression input (default: adata.X).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Parse ligand list
    ligand = [l.strip() for l in args.ligand.split(",")]
    if len(ligand) == 1:
        ligand = ligand[0]

    print(f"[run_ligand_flow] Reading {args.input}", flush=True)
    adata = sc.read_h5ad(args.input)
    print(f"[run_ligand_flow] AnnData: {adata}", flush=True)

    print(f"[run_ligand_flow] Loading prior network: {args.prior}", flush=True)
    prior_network = lf.load_prior(args.prior)

    print(f"[run_ligand_flow] Loading GRN: {args.grn}", flush=True)
    grn_network = lf.load_grn(args.grn)

    print(f"[run_ligand_flow] Running ligand flow for ligand(s): {ligand}", flush=True)
    lf.run_ligand_flow(
        adata,
        ligand=ligand,
        prior_network=prior_network,
        grn_network=grn_network,
        expression_layer=args.expression_layer,
        embedding_key=args.embedding_key,
        n_neighbors=args.n_neighbors,
        n_iter=args.n_iter,
        damping=args.damping,
        copy=False,
    )

    print(f"[run_ligand_flow] Writing {args.output}", flush=True)
    adata.write_h5ad(args.output)
    print("[run_ligand_flow] Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
