#!/usr/bin/env nextflow
/*
 * ligflow – Nextflow pipeline
 * ===========================
 *
 * Orchestrates the full ligand-flow analysis:
 *
 *   1. (Optional) Build NicheNet priors from R.
 *   2. Compute PCA + UMAP embedding.
 *   3. Run the ligand-flow pipeline.
 *   4. Generate diagnostic plots.
 *
 * Usage
 * -----
 * Standard run (with pre-built networks):
 *
 *   nextflow run main.nf -profile conda \
 *       --input      data.h5ad \
 *       --ligand     CXCL12 \
 *       --prior      nichenet_prior.csv \
 *       --grn        nichenet_grn.csv \
 *       --outdir     results
 *
 * Build NicheNet priors automatically (requires R + nichenetr):
 *
 *   nextflow run main.nf -profile conda \
 *       --input        data.h5ad \
 *       --ligand       CXCL12 \
 *       --build_priors true \
 *       --outdir       results
 *
 * Synthetic test (no real data needed):
 *
 *   nextflow run main.nf -profile conda,test
 *
 * Full parameter list:  nextflow run main.nf --help
 */

nextflow.enable.dsl = 2

// ── Modules ───────────────────────────────────────────────────────────────────

include { BUILD_PRIORS       } from './modules/local/build_priors'
include { COMPUTE_EMBEDDING  } from './modules/local/compute_embedding'
include { RUN_LIGAND_FLOW    } from './modules/local/run_ligand_flow'
include { PLOT_RESULTS       } from './modules/local/plot_results'

// ── Help message ─────────────────────────────────────────────────────────────

def helpMessage() {
    log.info """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    l i g f l o w                            ║
    ║   Ligand-induced cell-state transition simulation pipeline   ║
    ╚══════════════════════════════════════════════════════════════╝

    Usage:
        nextflow run main.nf -profile conda [options]

    Required (unless --build_priors true or -profile test):
        --input          Path to input AnnData h5ad file.
        --ligand         Ligand name(s), comma-separated for multiple.
        --prior          Ligand-target prior network (CSV/TSV).
        --grn            Gene regulatory network (CSV/TSV).

    Output:
        --outdir         Results directory  [default: results]

    Embedding:
        --n_pcs          Number of principal components  [default: 30]
        --n_neighbors    k-NN neighbours  [default: 30]
        --normalize      Normalise counts before PCA  [default: true]
        --target_sum     Target count for normalisation  [default: 10000]
        --embedding_key  adata.obsm key for the embedding  [default: X_umap]

    Ligand-flow:
        --n_iter         GRN propagation iterations  [default: 3]
        --damping        GRN propagation damping 0–1  [default: 0.8]
        --expression_layer  AnnData layer for expression  [default: adata.X]

    Plotting:
        --n_genes        Top target genes to show  [default: 20]
        --color          adata.obs column for velocity-plot colouring

    NicheNet priors (optional R step):
        --build_priors   Build NicheNet priors via R  [default: false]

    Profiles:
        conda            Use conda environments (recommended)
        mamba            Use mamba for faster conda solves
        test             Run the built-in synthetic test (no input required)

    Example:
        nextflow run main.nf -profile conda \\
            --input data.h5ad --ligand CXCL12 \\
            --prior nichenet_prior.csv --grn nichenet_grn.csv
    """.stripIndent()
}

if (params.help) {
    helpMessage()
    exit 0
}

// ── Validation ────────────────────────────────────────────────────────────────

workflow.onComplete {
    log.info (workflow.success ? "\n[ligflow] Pipeline completed successfully." : "\n[ligflow] Pipeline failed.")
}

// ── Main workflow ─────────────────────────────────────────────────────────────

workflow {

    // ── 0. Validate required parameters ──────────────────────────────────────
    if (params.input == null) {
        error "Please provide --input (path to h5ad file)."
    }
    if (params.ligand == null) {
        error "Please provide --ligand (ligand name(s) to simulate)."
    }
    if (!params.build_priors && (params.prior == null || params.grn == null)) {
        error "Please provide --prior and --grn (network files), or use --build_priors true to build NicheNet priors automatically."
    }

    // ── 1. (Optional) Build NicheNet priors via R ─────────────────────────────
    if (params.build_priors) {
        BUILD_PRIORS()
        ch_prior = BUILD_PRIORS.out.prior
        ch_grn   = BUILD_PRIORS.out.grn
    } else {
        ch_prior = Channel.fromPath(params.prior, checkIfExists: true)
        ch_grn   = Channel.fromPath(params.grn,   checkIfExists: true)
    }

    // ── 2. Build input channel ────────────────────────────────────────────────
    // meta map carries sample id, ligand(s), and optional colour column.
    ch_input = Channel
        .fromPath(params.input, checkIfExists: true)
        .map { h5ad ->
            def meta = [
                id    : h5ad.baseName,
                ligand: params.ligand,
                color : params.color ?: null,
            ]
            [ meta, h5ad ]
        }

    // ── 3. Compute embedding ──────────────────────────────────────────────────
    COMPUTE_EMBEDDING(ch_input)

    // ── 4. Run ligand-flow ────────────────────────────────────────────────────
    RUN_LIGAND_FLOW(
        COMPUTE_EMBEDDING.out.embedded,
        ch_prior,
        ch_grn
    )

    // ── 5. Plot results ───────────────────────────────────────────────────────
    PLOT_RESULTS(
        RUN_LIGAND_FLOW.out.result,
        ch_prior,
        ch_grn
    )
}
