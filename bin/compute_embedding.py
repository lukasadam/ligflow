#!/usr/bin/env python
"""
compute_embedding.py
--------------------
CLI wrapper for ligflow.compute_embedding.

Reads an AnnData h5ad file, computes PCA + UMAP embeddings, and writes the
result to a new h5ad file.

Usage
-----
compute_embedding.py \\
    --input  cells.h5ad \\
    --output cells_embedded.h5ad \\
    [--n-pcs 30] \\
    [--n-neighbors 30] \\
    [--no-normalize] \\
    [--target-sum 10000] \\
    [--umap-components 2]
"""

from __future__ import annotations

import argparse
import sys

import scanpy as sc
import ligflow as lf


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute PCA + UMAP embeddings for an AnnData object."
    )
    p.add_argument("--input", required=True, help="Input h5ad file.")
    p.add_argument("--output", required=True, help="Output h5ad file.")
    p.add_argument("--n-pcs", type=int, default=30, help="Number of PCs (default: 30).")
    p.add_argument(
        "--n-neighbors", type=int, default=30, help="k-NN neighbours (default: 30)."
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip normalisation / log1p (use when input is already normalised).",
    )
    p.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Target total count for normalisation (default: 10000).",
    )
    p.add_argument(
        "--umap-components",
        type=int,
        default=2,
        help="Number of UMAP dimensions (default: 2).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[compute_embedding] Reading {args.input}", flush=True)
    adata = sc.read_h5ad(args.input)
    print(f"[compute_embedding] AnnData: {adata}", flush=True)

    lf.compute_embedding(
        adata,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        normalize=not args.no_normalize,
        target_sum=args.target_sum,
        umap_n_components=args.umap_components,
        copy=False,
    )

    print(f"[compute_embedding] Writing {args.output}", flush=True)
    adata.write_h5ad(args.output)
    print("[compute_embedding] Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
