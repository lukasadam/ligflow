process PLOT_RESULTS {
    tag "${meta.id}:${meta.ligand}"

    conda "${moduleDir}/../../envs/ligflow.yml"

    publishDir "${params.outdir}/plots", mode: 'copy'

    input:
    tuple val(meta), path(h5ad)
    path prior
    path grn

    output:
    path "*.png", emit: plots

    script:
    def args     = task.ext.args  ?: ''
    def color    = meta.color     ? "--color ${meta.color}" : ''
    def n_genes  = params.n_genes ?: 20
    def basis    = params.embedding_key ? params.embedding_key.replaceAll(/^X_/, '') : 'umap'
    def prefix   = "${meta.id}_${meta.ligand}"
    """
    plot_results.py \\
        --input   ${h5ad} \\
        --ligand  "${meta.ligand}" \\
        --prior   ${prior} \\
        --grn     ${grn} \\
        --basis   ${basis} \\
        --n-genes ${n_genes} \\
        --prefix  ${prefix} \\
        ${color} \\
        ${args}
    """
}
