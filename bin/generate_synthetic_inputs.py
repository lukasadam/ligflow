#!/usr/bin/env python
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd

pd.options.future.infer_string = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ligflow inputs.")
    parser.add_argument("--output-h5ad", required=True, help="Path to write the synthetic AnnData file.")
    parser.add_argument("--output-prior", required=True, help="Path to write the synthetic prior CSV.")
    parser.add_argument("--output-grn", required=True, help="Path to write the synthetic GRN CSV.")
    parser.add_argument("--ligand", default="Gene0", help="Ligand gene name to seed the synthetic networks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-cells", type=int, default=200, help="Number of synthetic cells.")
    parser.add_argument("--n-genes", type=int, default=50, help="Number of synthetic genes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    gene_names = [f"Gene{i}" for i in range(args.n_genes)]
    ligand = args.ligand
    if ligand not in gene_names:
        gene_names[0] = ligand

    expression = rng.exponential(scale=2.0, size=(args.n_cells, args.n_genes)).astype(np.float32)
    cluster_names = np.array(["TypeA", "TypeB", "TypeC", "TypeD"])
    clusters = np.resize(cluster_names, args.n_cells)

    adata = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(clusters)},
            index=pd.Index(np.asarray([f"Cell{i}" for i in range(args.n_cells)], dtype=object)),
        ),
        var=pd.DataFrame(index=pd.Index(np.asarray(gene_names, dtype=object))),
    )

    prior_targets = gene_names[1 : min(16, len(gene_names))]
    prior_df = pd.DataFrame(
        {
            "source": ligand,
            "target": prior_targets,
            "weight": rng.uniform(0.2, 1.0, len(prior_targets)),
            "interaction_type": "ligand_target",
        }
    )

    grn_genes = gene_names[1:] or [ligand]
    n_edges = max(100, len(grn_genes) * 2)
    grn_df = pd.DataFrame(
        {
            "source": rng.choice(grn_genes, n_edges),
            "target": rng.choice(grn_genes, n_edges),
            "weight": rng.uniform(0.05, 0.5, n_edges),
            "interaction_type": "TF_target",
        }
    )

    adata.write_h5ad(args.output_h5ad)
    prior_df.to_csv(args.output_prior, index=False)
    grn_df.to_csv(args.output_grn, index=False)


if __name__ == "__main__":
    main()