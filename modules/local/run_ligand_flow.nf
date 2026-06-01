process RUN_LIGAND_FLOW {
    tag "${meta.id}:${meta.ligand}"

    conda "${moduleDir}/../../envs/ligflow.yml"

    publishDir "${params.outdir}/results", mode: 'copy'

    input:
    tuple val(meta), path(h5ad)
    path prior
    path grn

    output:
    tuple val(meta), path("${meta.id}_result.h5ad"), emit: result

    script:
    def args             = task.ext.args             ?: ''
    def n_neighbors      = params.n_neighbors        ?: 30
    def n_iter           = params.n_iter             ?: 3
    def damping          = params.damping            ?: 0.8
    def embedding_key    = params.embedding_key      ?: 'X_umap'
    def expression_layer = params.expression_layer   ? "--expression-layer ${params.expression_layer}" : ''
    """
    python ${projectDir}/bin/run_ligand_flow.py \\
        --input         ${h5ad} \\
        --prior         ${prior} \\
        --grn           ${grn} \\
        --ligand        "${meta.ligand}" \\
        --output        ${meta.id}_result.h5ad \\
        --embedding-key ${embedding_key} \\
        --n-neighbors   ${n_neighbors} \\
        --n-iter        ${n_iter} \\
        --damping       ${damping} \\
        ${expression_layer} \\
        ${args}
    """
}
