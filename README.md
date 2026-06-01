# ligflow

**ligflow** is a Python package for simulating ligand-induced cell-state transitions in single-cell RNA-seq data.

## Conceptual overview

The method does *not* infer true temporal dynamics from time-series data.  Instead, it simulates the likely local cell-state shift that would result from exposing cells to a specific ligand, by:

1. **Computing an initial perturbation vector** from a ligand-target prior network (e.g. NicheNet).
2. **Propagating** the perturbation through a gene regulatory network for a user-defined number of iterations.
3. **Estimating transition probabilities** by comparing each cell's predicted post-perturbation state to the expression of its local neighbours (Gaussian kernel).
4. **Projecting** ligand shift vectors into embedding space via `scvelo.tl.velocity_embedding`.

This approach is conceptually similar to RNA velocity (scVelo) but driven by *prior knowledge* about ligand signalling rather than splicing kinetics.

## Installation

```bash
pip install .
```

Or in editable mode for development:

```bash
pip install -e ".[dev]"
```

## Minimal example

```python
import scanpy as sc
import ligflow as lf

adata = sc.read_h5ad("data.h5ad")

# Load prior networks (CSV/TSV with columns: source, target, weight, interaction_type)
prior = lf.load_prior("nichenet_prior.tsv")
grn   = lf.load_grn("grn.tsv")

# Run the full pipeline
lf.run_ligand_flow(
    adata,
    ligand="WNT3A",
    prior_network=prior,
    grn_network=grn,
    embedding_key="X_umap",   # must be present in adata.obsm
    n_neighbors=30,
    n_iter=3,
    damping=0.8,
)

# Visualise
lf.pl.velocity_embedding(adata, basis="umap", color="cell_type")
lf.pl.ligand_effect_magnitude(adata)
lf.pl.top_target_genes(adata, n_genes=20)
lf.pl.workflow_diagnostics(
    adata,
    ligand="WNT3A",
    prior_network=prior,
    grn_network=grn,
    basis="umap",
)

# Save plots as PNGs
lf.pl.velocity_embedding(adata, basis="umap", color="cell_type", show=False, save="velocity.png")
lf.pl.ligand_effect_magnitude(adata, show=False, save="magnitude.png")
lf.pl.top_target_genes(adata, n_genes=20, show=False, save="top_genes.png")
lf.pl.workflow_diagnostics(
    adata,
    ligand="WNT3A",
    prior_network=prior,
    grn_network=grn,
    basis="umap",
    show=False,
    save="workflow_diagnostics.png",
)
```

`lf.pl.workflow_diagnostics(...)` now visualises six stages:
initial perturbation, GRN propagation, propagated perturbation, transition confidence, effect-size distribution, and embedding velocity.

Results are stored in:

| Key | Description |
|-----|-------------|
| `adata.layers["ligand_shift"]` | Per-cell gene-expression shift (n_cells × n_genes) |
| `adata.obsm["X_ligand_velocity"]` | Velocity vectors in embedding space (n_cells × n_dims) |
| `adata.uns["ligand_transition_probs"]` | Sparse cell–cell transition matrix |
| `adata.uns["ligand_metadata"]` | Run parameters |

## Prior network format

Both the prior ligand-target network and the GRN file should be CSV or TSV files with (at minimum) the following columns:

```
source  target  weight  interaction_type
WNT3A   FZD1    0.85    ligand_receptor
FZD1    CTNNB1  0.70    receptor_TF
CTNNB1  MYC     0.60    TF_target
```

`interaction_type` is optional; if absent it defaults to `"unknown"`.

## Package structure

```
ligflow/
    __init__.py        # public API
    priors.py          # load & manipulate prior networks
    imputation.py      # kNN expression smoothing
    propagation.py     # GRN signal propagation
    transitions.py     # cell-cell transition probabilities
    vectorfield.py     # transition matrix → embedding vectors
    plotting.py        # visualisation (lf.pl.*)
    workflow.py        # high-level run_ligand_flow()
```

## Running tests

```bash
pytest
```

## Running the synthetic example

```bash
python examples/synthetic_example.py
```

## Benchmarking

Run a synthetic runtime benchmark (CSV + runtime plot):

```bash
python examples/benchmark_synthetic.py
```

This writes:

- `ligflow_benchmark_results.csv`
- `ligflow_benchmark_runtime.png`