#!/usr/bin/env Rscript
# build_nichenet_priors.R
# =======================
# Extract the NicheNet ligand-target prior network and gene regulatory network
# (GRN) directly from the nichenetr R package and write them as CSV files that
# ligflow can consume.
#
# Prerequisites
# -------------
# Install nichenetr from GitHub (requires devtools):
#   install.packages("devtools")
#   devtools::install_github("saeyslab/nichenetr")
#
# Usage
# -----
# Rscript examples/build_nichenet_priors.R
#
# Output files (written to the current working directory)
# -------------------------------------------------------
#   nichenet_prior.csv  – ligand-to-target regulatory-potential scores
#   nichenet_grn.csv    – gene regulatory network (TF → target)
#   nichenet_lr.csv     – ligand-receptor binding network (optional)
#
# Column format understood by ligflow
# ------------------------------------
#   source           character  upstream node (ligand or TF)
#   target           character  downstream node (target gene or receptor)
#   weight           numeric    regulatory potential / edge weight
#   interaction_type character  annotation label

# ── 0. Parameters ─────────────────────────────────────────────────────────────

# Set to "human" or "mouse" depending on your scRNA-seq data.
ORGANISM <- "human"

# Minimum regulatory-potential threshold for ligand-target edges.
# Raising this value shrinks the file but loses weak interactions.
# NicheNet scores are roughly in [0, 1]; the recommended threshold is 0.
# A value of ~0.001 gives a good signal-to-noise tradeoff for most analyses.
LIGAND_TARGET_THRESHOLD <- 0.001

# Output paths
OUT_PRIOR <- "nichenet_prior.csv"
OUT_GRN   <- "nichenet_grn.csv"
OUT_LR    <- "nichenet_lr.csv"

# ── 1. Load package ───────────────────────────────────────────────────────────

if (!requireNamespace("nichenetr", quietly = TRUE)) {
  stop(
    "nichenetr is not installed.\n",
    "Please run:\n",
    "  install.packages('devtools')\n",
    "  devtools::install_github('saeyslab/nichenetr')\n"
  )
}
library(nichenetr)

cat("nichenetr loaded.\n")

# ── 2. Load NicheNet data objects ─────────────────────────────────────────────
# nichenetr ships built-in RDS objects for both human and mouse.
# For the most up-to-date networks, download from Zenodo (see comments below).

if (ORGANISM == "human") {
  # Option A: built-in package data (may lag latest Zenodo release)
  ligand_target_matrix <- readRDS(url(
    "https://zenodo.org/record/7074291/files/ligand_target_matrix_nsga2r_final.rds"
  ))
  lr_network <- readRDS(url(
    "https://zenodo.org/record/7074291/files/lr_network_human_21122021.rds"
  ))
  weighted_networks <- readRDS(url(
    "https://zenodo.org/record/7074291/files/weighted_networks_nsga2r_final.rds"
  ))
} else if (ORGANISM == "mouse") {
  ligand_target_matrix <- readRDS(url(
    "https://zenodo.org/record/7074291/files/ligand_target_matrix_nsga2r_final_mouse.rds"
  ))
  lr_network <- readRDS(url(
    "https://zenodo.org/record/7074291/files/lr_network_mouse_21122021.rds"
  ))
  weighted_networks <- readRDS(url(
    "https://zenodo.org/record/7074291/files/weighted_networks_nsga2r_final_mouse.rds"
  ))
} else {
  stop("ORGANISM must be 'human' or 'mouse', got: ", ORGANISM)
}

cat(sprintf(
  "Loaded ligand_target_matrix: %d target genes × %d ligands\n",
  nrow(ligand_target_matrix), ncol(ligand_target_matrix)
))
cat(sprintf(
  "Loaded lr_network: %d rows\n",
  nrow(lr_network)
))
cat(sprintf(
  "Loaded weighted_networks$gr: %d rows\n",
  nrow(weighted_networks$gr)
))

# ── 3. Build prior network (ligand → target genes) ───────────────────────────
# ligand_target_matrix: rows = target genes, columns = ligands.
# We convert to long (edge-list) format, filter low-weight edges, and add
# the interaction_type column expected by ligflow.

cat("\nConverting ligand_target_matrix to long format ...\n")
lt_long <- as.data.frame(as.table(ligand_target_matrix), stringsAsFactors = FALSE)
colnames(lt_long) <- c("target", "source", "weight")

# Keep only edges above the threshold
lt_long <- lt_long[lt_long$weight >= LIGAND_TARGET_THRESHOLD, ]
lt_long$interaction_type <- "ligand_target"

# Reorder columns to match ligflow convention
prior_df <- lt_long[, c("source", "target", "weight", "interaction_type")]
rownames(prior_df) <- NULL

cat(sprintf(
  "  %d ligand-target edges retained (threshold = %.4f).\n",
  nrow(prior_df), LIGAND_TARGET_THRESHOLD
))

write.csv(prior_df, OUT_PRIOR, row.names = FALSE)
cat(sprintf("Saved prior network → %s\n", OUT_PRIOR))

# ── 4. Build GRN (TF / upstream gene → target gene) ──────────────────────────
# weighted_networks$gr has columns: from, to, weight
# We rename to source / target as expected by ligflow.

cat("\nBuilding GRN from weighted_networks$gr ...\n")
grn_raw <- as.data.frame(weighted_networks$gr, stringsAsFactors = FALSE)
grn_df <- data.frame(
  source           = grn_raw$from,
  target           = grn_raw$to,
  weight           = grn_raw$weight,
  interaction_type = "TF_target",
  stringsAsFactors = FALSE
)

cat(sprintf("  %d GRN edges.\n", nrow(grn_df)))
write.csv(grn_df, OUT_GRN, row.names = FALSE)
cat(sprintf("Saved GRN → %s\n", OUT_GRN))

# ── 5. (Optional) Build ligand-receptor network ───────────────────────────────
# lr_network has columns: from (ligand), to (receptor), source, database.
# We expose this as a separate CSV for reference / downstream filtering.

cat("\nBuilding ligand-receptor network from lr_network ...\n")
lr_raw <- as.data.frame(lr_network, stringsAsFactors = FALSE)
lr_df <- data.frame(
  source           = lr_raw$from,
  target           = lr_raw$to,
  weight           = if ("weight" %in% colnames(lr_raw)) lr_raw$weight else 1.0,
  interaction_type = "ligand_receptor",
  stringsAsFactors = FALSE
)
# Remove duplicates (lr_network may list the same pair from multiple databases)
lr_df <- unique(lr_df)

cat(sprintf("  %d ligand-receptor edges.\n", nrow(lr_df)))
write.csv(lr_df, OUT_LR, row.names = FALSE)
cat(sprintf("Saved LR network → %s\n", OUT_LR))

# ── 6. Summary ────────────────────────────────────────────────────────────────

cat("\nSummary\n")
cat(sprintf("  Organism            : %s\n", ORGANISM))
cat(sprintf("  Ligands in prior    : %d\n", length(unique(prior_df$source))))
cat(sprintf("  Target genes        : %d\n", length(unique(prior_df$target))))
cat(sprintf("  GRN genes           : %d\n", length(unique(c(grn_df$source, grn_df$target)))))
cat(sprintf("  LR pairs            : %d\n", nrow(lr_df)))
cat("\nDone. Use these CSVs with the ligflow Python package:\n")
cat(sprintf(
  '  lf.load_prior("%s") and lf.load_grn("%s")\n', OUT_PRIOR, OUT_GRN
))
