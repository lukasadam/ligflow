"""Tests for ligflow.priors"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from ligflow.priors import load_prior, network_to_adjacency, subset_by_ligand


# ── load_prior ────────────────────────────────────────────────────────────────


def _write_tmp_csv(tmp_path, content, filename="net.csv"):
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


def test_load_prior_csv(tmp_path):
    csv = _write_tmp_csv(
        tmp_path,
        """\
        source,target,weight,interaction_type
        LIG1,TGT1,0.5,ligand_target
        LIG1,TGT2,0.3,ligand_target
        """,
    )
    df = load_prior(csv)
    assert list(df.columns) == ["source", "target", "weight", "interaction_type"]
    assert len(df) == 2
    assert df["source"].iloc[0] == "LIG1"


def test_load_prior_tsv(tmp_path):
    tsv = tmp_path / "net.tsv"
    tsv.write_text("source\ttarget\tweight\nA\tB\t1.0\nC\tD\t0.2\n")
    df = load_prior(tsv)
    assert len(df) == 2


def test_load_prior_missing_column(tmp_path):
    csv = _write_tmp_csv(
        tmp_path,
        """\
        src,tgt,w
        LIG1,TGT1,0.5
        """,
    )
    with pytest.raises(ValueError, match="Required columns missing"):
        load_prior(csv)


def test_load_prior_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_prior("/nonexistent/path/net.tsv")


def test_load_prior_adds_interaction_type_if_missing(tmp_path):
    csv = _write_tmp_csv(
        tmp_path,
        """\
        source,target,weight
        A,B,0.5
        """,
    )
    df = load_prior(csv)
    assert "interaction_type" in df.columns
    assert df["interaction_type"].iloc[0] == "unknown"


# ── subset_by_ligand ──────────────────────────────────────────────────────────


def test_subset_by_ligand(prior_network):
    sub = subset_by_ligand(prior_network, "Gene0")
    assert len(sub) > 0
    assert all(sub["source"] == "Gene0")


def test_subset_by_ligand_list(prior_network):
    sub = subset_by_ligand(prior_network, ["Gene0"])
    assert len(sub) > 0


def test_subset_by_ligand_not_found(prior_network):
    with pytest.raises(ValueError, match="None of the supplied ligands"):
        subset_by_ligand(prior_network, "NONEXISTENT_LIGAND")


# ── network_to_adjacency ──────────────────────────────────────────────────────


def test_network_to_adjacency_shape(prior_network, gene_names):
    adj = network_to_adjacency(prior_network, gene_names)
    n = len(gene_names)
    assert adj.shape == (n, n)


def test_network_to_adjacency_nonzero(prior_network, gene_names):
    adj = network_to_adjacency(prior_network, gene_names)
    assert adj.nnz > 0


def test_network_to_adjacency_unknown_genes(prior_network):
    """Genes not in the network should result in an all-zero adjacency."""
    small_genes = ["ABSENT1", "ABSENT2"]
    adj = network_to_adjacency(prior_network, small_genes)
    assert adj.nnz == 0
