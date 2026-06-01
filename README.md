# ligflow

**ligflow** is a Python package for simulating ligand-induced cell-state transitions in single-cell RNA-seq data.

## Conceptual overview

The method does *not* infer true temporal dynamics from time-series data.  Instead, it simulates the likely local cell-state shift that would result from exposing cells to a specific ligand, by:

1. **Computing an initial perturbation vector** from a ligand-target prior network (e.g. NicheNet).
2. **Propagating** the perturbation through a gene regulatory network for a user-defined number of iterations.
3. **Estimating transition probabilities** by comparing each cell's predicted post-perturbation state to the expression of its local neighbours (Gaussian kernel).
4. **Projecting** ligand shift vectors into embedding space via `scvelo.tl.velocity_embedding`.

This approach is conceptually similar to RNA velocity (scVelo) but driven by *prior knowledge* about ligand signalling rather than splicing kinetics.

## Nextflow pipeline (recommended)

The recommended way to run ligflow is via the bundled Nextflow pipeline, which
manages all dependencies through conda environments automatically.

### Prerequisites

- [Nextflow ≥ 23.04](https://www.nextflow.io/docs/latest/getstarted.html)
- [conda](https://docs.conda.io/en/latest/miniconda.html) (or
  [mamba](https://mamba.readthedocs.io/) for faster solves)

### Quick start

```bash
# Standard run with pre-built prior networks
nextflow run main.nf -profile conda \
    --input      data.h5ad \
    --ligand     CXCL12 \
    --prior      nichenet_prior.csv \
    --grn        nichenet_grn.csv \
    --outdir     results
```

Results are written to `results/`:

| Subdirectory | Contents |
|---|---|
| `results/embedding/` | h5ad with PCA + UMAP |
| `results/results/` | h5ad with ligand-flow outputs |
| `results/plots/` | PNG diagnostic plots |
| `results/pipeline_info/` | Nextflow execution report / trace |

### Build NicheNet priors automatically

If you do not have pre-built prior networks, the pipeline can build them from
the [NicheNet](https://github.com/saeyslab/nichenetr) R package (requires R
and Internet access to Zenodo):

```bash
nextflow run main.nf -profile conda \
    --input        data.h5ad \
    --ligand       CXCL12 \
    --build_priors true \
    --outdir       results
```

### Use mamba for faster dependency solves

```bash
nextflow run main.nf -profile mamba \
    --input data.h5ad --ligand CXCL12 \
    --prior nichenet_prior.csv --grn nichenet_grn.csv
```

### Full parameter reference

```
nextflow run main.nf --help
```

### Pipeline structure

```
main.nf                     # Main Nextflow workflow
nextflow.config             # Global config + profiles
conf/
  base.config               # Default resource limits
envs/
  ligflow.yml               # Python conda environment (all Python steps)
  nichenet.yml              # R conda environment (NicheNet prior building)
modules/local/
  build_priors.nf           # Process: build NicheNet priors (R)
  compute_embedding.nf      # Process: PCA + UMAP
  run_ligand_flow.nf        # Process: ligand-flow simulation
  plot_results.nf           # Process: diagnostic plots
bin/
  compute_embedding.py      # CLI script called by COMPUTE_EMBEDDING
  run_ligand_flow.py        # CLI script called by RUN_LIGAND_FLOW
  plot_results.py           # CLI script called by PLOT_RESULTS
```

---

## Python package (direct use)

You can also use ligflow directly as a Python package without Nextflow.

### Installation

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

# Compute PCA and UMAP embeddings (normalise → PCA → k-NN → UMAP)
lf.compute_embedding(adata, n_pcs=30, n_neighbors=30)

# Run the full pipeline
lf.run_ligand_flow(
    adata,
    ligand="WNT3A",
    prior_network=prior,
    grn_network=grn,
    embedding_key="X_umap",
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

After running `lf.compute_embedding` the following are also available:

| Key | Description |
|-----|-------------|
| `adata.obsm["X_pca"]` | PCA coordinates (n_cells × n_pcs) |
| `adata.obsm["X_umap"]` | UMAP coordinates (n_cells × 2) |
| `adata.obsp["connectivities"]` | k-NN connectivity matrix |

Results stored by `lf.run_ligand_flow`:

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

## NicheNet + PBMC 3k example

This example runs ligflow with the real NicheNet ligand-target priors on the
canonical PBMC 3k 10x dataset.

**Step 1 – export NicheNet priors to CSV (requires R + nichenetr):**

```bash
# install nichenetr once
Rscript -e "devtools::install_github('saeyslab/nichenetr')"

# export prior network and GRN as CSV
Rscript examples/build_nichenet_priors.R
```

This writes `nichenet_prior.csv`, `nichenet_grn.csv`, and `nichenet_lr.csv` to
the current working directory.

**Step 2 – run the Python example:**

```bash
python examples/nichenet_pbmc3k_example.py
```

Output plots:

- `nichenet_pbmc3k_velocity.png`    – UMAP + velocity arrows coloured by Leiden cluster
- `nichenet_pbmc3k_magnitude.png`   – per-cell perturbation effect magnitude
- `nichenet_pbmc3k_top_genes.png`   – top predicted target genes
- `nichenet_pbmc3k_diagnostics.png` – six-panel workflow diagnostic

The default ligand is `CXCL12`; change the `LIGAND` variable at the top of
the script to explore other ligands (e.g. `CCL5`, `CXCL10`, `TNF`, `TGFB1`).

## Benchmarking

Run a synthetic runtime benchmark (CSV + runtime plot):

```bash
python examples/benchmark_synthetic.py
```

This writes:

- `ligflow_benchmark_results.csv`
- `ligflow_benchmark_runtime.png`