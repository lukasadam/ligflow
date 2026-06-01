process BUILD_PRIORS {
    tag "build_nichenet_priors"

    conda "${moduleDir}/../../envs/nichenet.yml"

    publishDir "${params.outdir}/priors", mode: 'copy'

    output:
    path "nichenet_prior.csv",  emit: prior
    path "nichenet_grn.csv",    emit: grn
    path "nichenet_lr.csv",     emit: lr

    script:
    """
    Rscript ${projectDir}/examples/build_nichenet_priors.R
    """
}
