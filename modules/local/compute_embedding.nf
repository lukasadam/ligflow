process COMPUTE_EMBEDDING {
    tag "${meta.id}"

    conda "${moduleDir}/../../envs/ligflow.yml"

    publishDir "${params.outdir}/embedding", mode: 'copy'

    input:
    tuple val(meta), path(h5ad)

    output:
    tuple val(meta), path("${meta.id}_embedded.h5ad"), emit: embedded

    script:
    def args        = task.ext.args        ?: ''
    def n_pcs       = params.n_pcs         ?: 30
    def n_neighbors = params.n_neighbors   ?: 30
    def target_sum  = params.target_sum    ?: 10000
    def normalize   = params.normalize == false ? '--no-normalize' : ''
    """
    compute_embedding.py \\
        --input        ${h5ad} \\
        --output       ${meta.id}_embedded.h5ad \\
        --n-pcs        ${n_pcs} \\
        --n-neighbors  ${n_neighbors} \\
        --target-sum   ${target_sum} \\
        ${normalize} \\
        ${args}
    """
}
