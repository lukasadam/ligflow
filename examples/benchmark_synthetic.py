"""
benchmark_synthetic.py
======================
Run simple synthetic benchmarks for ligflow and save a runtime plot.

Run with:
    python examples/benchmark_synthetic.py
"""

from __future__ import annotations

import time

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

import ligflow as lf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_synthetic_inputs(
    n_cells: int,
    n_genes: int,
    n_grn_edges: int,
    seed: int = 42,
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    ligand = "Gene0"

    X = rng.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)
    clusters = np.repeat(["TypeA", "TypeB", "TypeC", "TypeD"], n_cells // 4)
    if len(clusters) < n_cells:
        clusters = np.concatenate([clusters, np.repeat("TypeA", n_cells - len(clusters))])

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": pd.Categorical(clusters)}, index=[f"Cell{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=gene_names),
    )

    centres = {"TypeA": (-3, 3), "TypeB": (3, 3), "TypeC": (-3, -3), "TypeD": (3, -3)}
    coords = np.vstack(
        [
            rng.normal(loc=centres[c], scale=0.8, size=((clusters == c).sum(), 2))
            for c in ["TypeA", "TypeB", "TypeC", "TypeD"]
        ]
    )
    adata.obsm["X_umap"] = coords.astype(np.float32)

    prior_df = pd.DataFrame(
        {
            "source": ligand,
            "target": gene_names[1 : min(16, n_genes)],
            "weight": rng.uniform(0.2, 1.0, min(15, max(0, n_genes - 1))),
            "interaction_type": "ligand_target",
        }
    )

    grn_df = pd.DataFrame(
        {
            "source": rng.choice(gene_names[1:], n_grn_edges),
            "target": rng.choice(gene_names[1:], n_grn_edges),
            "weight": rng.uniform(0.05, 0.5, n_grn_edges),
            "interaction_type": "TF_target",
        }
    )

    return adata, prior_df, grn_df, ligand


def benchmark_config(
    n_cells: int,
    n_genes: int,
    repeats: int = 3,
    warmup: int = 1,
    seed: int = 42,
) -> dict[str, float]:
    times = []
    total_runs = warmup + repeats
    for rep in range(total_runs):
        adata, prior_df, grn_df, ligand = build_synthetic_inputs(
            n_cells=n_cells,
            n_genes=n_genes,
            n_grn_edges=max(100, n_genes * 3),
            seed=seed + rep,
        )
        t0 = time.perf_counter()
        lf.run_ligand_flow(
            adata,
            ligand=ligand,
            prior_network=prior_df,
            grn_network=grn_df,
            embedding_key="X_umap",
            n_neighbors=20,
            n_iter=3,
            damping=0.8,
            copy=False,
        )
        elapsed = time.perf_counter() - t0
        if rep >= warmup:
            times.append(elapsed)

    arr = np.asarray(times, dtype=float)
    return {
        "n_cells": float(n_cells),
        "n_genes": float(n_genes),
        "mean_seconds": float(arr.mean()),
        "std_seconds": float(arr.std(ddof=0)),
        "min_seconds": float(arr.min()),
    }


def main() -> None:
    configs = [
        (200, 50),
        (500, 100),
        (1000, 200),
        (2000, 400),
    ]

    print("Running ligflow synthetic benchmarks...")
    rows = []
    for n_cells, n_genes in configs:
        row = benchmark_config(n_cells=n_cells, n_genes=n_genes, repeats=3)
        rows.append(row)
        print(
            f"  n_cells={int(row['n_cells']):4d}, n_genes={int(row['n_genes']):4d} | "
            f"mean={row['mean_seconds']:.4f}s ± {row['std_seconds']:.4f}s (min {row['min_seconds']:.4f}s)"
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv("ligflow_benchmark_results.csv", index=False)
    print("\nSaved benchmark table to ligflow_benchmark_results.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        result_df["n_cells"].astype(int),
        result_df["mean_seconds"],
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="mean runtime",
    )
    ax.fill_between(
        result_df["n_cells"].astype(int),
        result_df["mean_seconds"] - result_df["std_seconds"],
        result_df["mean_seconds"] + result_df["std_seconds"],
        alpha=0.2,
        color="tab:blue",
        label="±1 std",
    )
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("ligflow runtime on synthetic data")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("ligflow_benchmark_runtime.png", dpi=150, bbox_inches="tight")
    print("Saved benchmark plot to ligflow_benchmark_runtime.png")


if __name__ == "__main__":
    main()
