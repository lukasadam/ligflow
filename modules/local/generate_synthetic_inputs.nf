process GENERATE_SYNTHETIC_INPUTS {
    tag "synthetic_inputs"

    conda "${moduleDir}/../../envs/ligflow.yml"

    output:
    path "synthetic_input.h5ad", emit: input
    path "synthetic_prior.csv", emit: prior
    path "synthetic_grn.csv", emit: grn

    script:
    def ligand = params.ligand ?: 'Gene0'
    """
    python ${projectDir}/bin/generate_synthetic_inputs.py \\
        --output-h5ad synthetic_input.h5ad \\
        --output-prior synthetic_prior.csv \\
        --output-grn synthetic_grn.csv \\
        --ligand ${ligand}
    """
}