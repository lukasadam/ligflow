"""
plotting.py
-----------
Visualisation functions for ligflow results.
All functions live under ``ligflow.pl``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData


def velocity_embedding(
    adata: AnnData,
    basis: str = "umap",
    color: Optional[str] = None,
    velocity_key: str = "X_ligand_velocity",
    scale: float = 1.0,
    arrow_size: float = 1.0,
    arrow_color: str = "black",
    alpha: float = 0.8,
    figsize: tuple[float, float] = (7, 6),
    show: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,  # noqa: F821
    title: Optional[str] = None,
) -> Optional["matplotlib.axes.Axes"]:  # noqa: F821
    """Plot embedding with ligand-velocity arrows.

    Parameters
    ----------
    adata:
        Annotated data matrix containing ``adata.obsm[velocity_key]`` and
        ``adata.obsm["X_<basis>"]``.
    basis:
        Name of the embedding (e.g. ``"umap"``, ``"tsne"``).
    color:
        ``obs`` column used to colour cells (e.g. a cluster label).
    velocity_key:
        Key in ``adata.obsm`` that contains the velocity vectors.
    scale:
        Multiplicative scaling factor for arrow lengths.
    arrow_size:
        Relative size of the arrow heads.
    arrow_color:
        Colour of the arrows.
    alpha:
        Transparency of the arrows.
    figsize:
        Figure size ``(width, height)`` in inches.
    show:
        If *True* call ``plt.show()`` before returning.
    ax:
        Existing axes to draw into.  A new figure is created when *None*.
    title:
        Plot title.  Defaults to ``"Ligand velocity (<basis>)"``.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes object if *show* is *False*, otherwise *None*.

    Raises
    ------
    KeyError
        If the required keys are not present in *adata*.
    """
    import matplotlib.pyplot as plt

    emb_key = f"X_{basis}"
    if emb_key not in adata.obsm:
        raise KeyError(
            f"Embedding key '{emb_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    if n_iter < 0:
        raise ValueError(f"n_iter must be non-negative, got {n_iter}")
    if not (0.0 <= damping <= 1.0):
        raise ValueError(f"damping must be in [0, 1], got {damping}")
    if velocity_key not in adata.obsm:
        raise KeyError(
            f"Velocity key '{velocity_key}' not found in adata.obsm. "
            "Run lf.run_ligand_flow() first."
        )

    coords = np.array(adata.obsm[emb_key])
    vels = np.array(adata.obsm[velocity_key])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # ── Background scatter ───────────────────────────────────────────────────
    if color is not None and color in adata.obs.columns:
        groups = adata.obs[color].astype("category")
        cats = groups.cat.categories
        cmap = plt.get_cmap("tab20", len(cats))
        for k, cat in enumerate(cats):
            mask = groups == cat
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=4,
                color=cmap(k),
                label=str(cat),
                rasterized=True,
            )
        ax.legend(markerscale=3, loc="best", fontsize=7, framealpha=0.5)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=4, color="lightgrey", rasterized=True)

    # ── Arrows ───────────────────────────────────────────────────────────────
    x, y = coords[:, 0], coords[:, 1]
    dx = vels[:, 0] * scale
    dy = vels[:, 1] * scale
    ax.quiver(
        x, y, dx, dy,
        color=arrow_color,
        alpha=alpha,
        scale_units="xy",
        angles="xy",
        scale=1.0 / arrow_size,
        headwidth=4,
        headlength=5,
        width=0.002,
    )

    ax.set_xlabel(f"{basis.upper()} 1")
    ax.set_ylabel(f"{basis.upper()} 2")
    ax.set_title(title or f"Ligand velocity ({basis})")
    ax.set_aspect("equal", "box")

    if show:
        plt.tight_layout()
        plt.show()
        return None
    return ax


def ligand_effect_magnitude(
    adata: AnnData,
    basis: str = "umap",
    velocity_key: str = "X_ligand_velocity",
    figsize: tuple[float, float] = (7, 6),
    cmap: str = "viridis",
    show: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,  # noqa: F821
    title: Optional[str] = None,
) -> Optional["matplotlib.axes.Axes"]:  # noqa: F821
    """Plot ligand effect magnitude (vector length) per cell in embedding space.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    basis:
        Name of the embedding.
    velocity_key:
        Key in ``adata.obsm`` for velocity vectors.
    figsize:
        Figure size.
    cmap:
        Matplotlib colormap name.
    show:
        If *True* call ``plt.show()`` before returning.
    ax:
        Existing axes.
    title:
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes or None
    """
    import matplotlib.pyplot as plt

    emb_key = f"X_{basis}"
    if emb_key not in adata.obsm:
        raise KeyError(f"Embedding key '{emb_key}' not found in adata.obsm.")
    if velocity_key not in adata.obsm:
        raise KeyError(f"Velocity key '{velocity_key}' not found in adata.obsm.")

    coords = np.array(adata.obsm[emb_key])
    vels = np.array(adata.obsm[velocity_key])
    magnitudes = np.linalg.norm(vels, axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=magnitudes,
        s=4,
        cmap=cmap,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Effect magnitude")
    ax.set_xlabel(f"{basis.upper()} 1")
    ax.set_ylabel(f"{basis.upper()} 2")
    ax.set_title(title or f"Ligand effect magnitude ({basis})")
    ax.set_aspect("equal", "box")

    if show:
        plt.tight_layout()
        plt.show()
        return None
    return ax


def top_target_genes(
    adata: AnnData,
    n_genes: int = 20,
    layer: str = "ligand_shift",
    figsize: tuple[float, float] = (8, 5),
    color: str = "steelblue",
    show: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,  # noqa: F821
    title: Optional[str] = None,
) -> Optional["matplotlib.axes.Axes"]:  # noqa: F821
    """Bar plot of the top predicted target genes ranked by mean perturbation.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    n_genes:
        Number of top genes to display.
    layer:
        Layer in ``adata.layers`` containing the perturbation shift.
    figsize:
        Figure size.
    color:
        Bar colour.
    show:
        If *True* call ``plt.show()`` before returning.
    ax:
        Existing axes.
    title:
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes or None
    """
    import matplotlib.pyplot as plt
    import scipy.sparse as sp

    if layer not in adata.layers:
        raise KeyError(
            f"Layer '{layer}' not found in adata.layers. "
            "Run lf.run_ligand_flow() first."
        )

    shift = adata.layers[layer]
    if sp.issparse(shift):
        shift = np.asarray(shift.todense())

    mean_shift = np.abs(shift).mean(axis=0)
    top_idx = np.argsort(mean_shift)[::-1][:n_genes]
    top_genes = np.array(adata.var_names)[top_idx]
    top_vals = mean_shift[top_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.barh(top_genes[::-1], top_vals[::-1], color=color)
    ax.set_xlabel("Mean |ligand shift|")
    ax.set_title(title or f"Top {n_genes} predicted target genes")
    ax.invert_yaxis()

    if show:
        plt.tight_layout()
        plt.show()
        return None
    return ax


def workflow_diagnostics(
    adata: AnnData,
    ligand: str | list[str],
    prior_network: pd.DataFrame,
    grn_network: pd.DataFrame,
    expression_layer: Optional[str] = None,
    basis: str = "umap",
    n_neighbors: int = 30,
    n_iter: int = 3,
    damping: float = 0.8,
    kernel_sigma: Optional[float] = None,
    top_n_genes: int = 12,
    figsize: tuple[float, float] = (16, 9),
    show: bool = True,
) -> Optional[np.ndarray]:
    """Plot a five-step visual diagnostic overview of the ligflow workflow.

    The figure contains panels for:
    1) kNN expression smoothing,
    2) initial perturbation from ligand-target priors,
    3) GRN propagation over iterations,
    4) transition probability confidence,
    5) velocity vector field on embedding.
    """
    import matplotlib.pyplot as plt
    import scipy.sparse as sp

    from .imputation import knn_smooth
    from .priors import network_to_adjacency, subset_by_ligand
    from .propagation import _column_normalise, build_initial_delta
    from .transitions import compute_transition_probabilities
    from .vectorfield import transition_to_vectors

    emb_key = f"X_{basis}"
    if emb_key not in adata.obsm:
        raise KeyError(
            f"Embedding key '{emb_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    if expression_layer is None:
        X_raw = adata.X
    else:
        X_raw = adata.layers[expression_layer]
    if sp.issparse(X_raw):
        X_raw = np.asarray(X_raw.todense())
    else:
        X_raw = np.asarray(X_raw)

    # Step 1: kNN smoothing
    X_smooth = knn_smooth(
        adata=adata,
        layer=expression_layer,
        n_neighbors=n_neighbors,
        use_existing_neighbors=True,
    )

    # Step 2: initial perturbation from ligand-target prior
    gene_names = list(adata.var_names)
    ligand_list = [ligand] if isinstance(ligand, str) else list(ligand)
    ligand_subnet = subset_by_ligand(prior_network, ligand_list)
    initial_delta = build_initial_delta(
        ligand=ligand_list,
        prior_network=ligand_subnet,
        gene_names=gene_names,
    )

    # Step 3: propagation through GRN
    grn_matrix = network_to_adjacency(grn_network, gene_names)
    G = _column_normalise(grn_matrix)
    delta = initial_delta.copy().astype(np.float64)
    propagation_norms = [float(np.linalg.norm(delta))]
    for _ in range(n_iter):
        if sp.issparse(G):
            delta = damping * np.asarray(G @ delta).ravel() + initial_delta
        else:
            delta = damping * (G @ delta) + initial_delta
        propagation_norms.append(float(np.linalg.norm(delta)))
    propagated = np.tile(delta, (X_smooth.shape[0], 1))

    # Step 4: transition matrix from Gaussian kernel
    T = compute_transition_probabilities(
        adata=adata,
        expression=X_smooth,
        propagated_delta=propagated,
        n_neighbors=n_neighbors,
        kernel_sigma=kernel_sigma,
        use_existing_neighbors=True,
    )

    # Step 5: velocity vectors in embedding
    vectors = transition_to_vectors(adata=adata, transition_matrix=T, embedding_key=emb_key)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax_smooth, ax_delta, ax_prop = axes[0]
    ax_trans, ax_vel, ax_unused = axes[1]

    # 1) Smoothing panel
    raw_mean = X_raw.mean(axis=1)
    smooth_mean = X_smooth.mean(axis=1)
    ax_smooth.scatter(raw_mean, smooth_mean, s=10, alpha=0.75, color="tab:blue")
    lo = float(min(raw_mean.min(), smooth_mean.min()))
    hi = float(max(raw_mean.max(), smooth_mean.max()))
    ax_smooth.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    ax_smooth.set_xlabel("Per-cell mean (raw)")
    ax_smooth.set_ylabel("Per-cell mean (smoothed)")
    ax_smooth.set_title("1) kNN smoothing")

    # 2) Initial perturbation panel
    nonzero_idx = np.where(np.abs(initial_delta) > 0)[0]
    if nonzero_idx.size == 0:
        ax_delta.text(0.5, 0.5, "No prior targets found", ha="center", va="center")
        ax_delta.set_xticks([])
        ax_delta.set_yticks([])
    else:
        ranked_idx = nonzero_idx[np.argsort(np.abs(initial_delta[nonzero_idx]))[::-1]]
        top_idx = ranked_idx[:top_n_genes]
        genes = [gene_names[i] for i in top_idx]
        vals = initial_delta[top_idx]
        ax_delta.barh(genes, vals, color="tab:orange")
        ax_delta.set_xlabel("Initial perturbation weight")
        ax_delta.invert_yaxis()
    ax_delta.set_title("2) Initial perturbation")

    # 3) Propagation panel
    ax_prop.plot(range(len(propagation_norms)), propagation_norms, marker="o", color="tab:green")
    ax_prop.set_xlabel("Iteration")
    ax_prop.set_ylabel(r"$||\delta||_2$")
    ax_prop.set_title("3) GRN propagation")

    # 4) Transition probabilities panel
    row_max = np.asarray(T.max(axis=1).toarray()).ravel()
    ax_trans.hist(row_max, bins=25, color="tab:purple", alpha=0.85)
    ax_trans.set_xlabel("Max transition probability per cell")
    ax_trans.set_ylabel("Cell count")
    ax_trans.set_title("4) Transition probabilities")

    # 5) Velocity field panel
    coords = np.asarray(adata.obsm[emb_key])
    ax_vel.scatter(coords[:, 0], coords[:, 1], s=5, color="lightgrey", rasterized=True)
    ax_vel.quiver(
        coords[:, 0],
        coords[:, 1],
        vectors[:, 0],
        vectors[:, 1],
        color="black",
        alpha=0.8,
        scale_units="xy",
        angles="xy",
        scale=1.0,
        headwidth=4,
        headlength=5,
        width=0.002,
    )
    ax_vel.set_xlabel(f"{basis.upper()} 1")
    ax_vel.set_ylabel(f"{basis.upper()} 2")
    ax_vel.set_title("5) Velocity field")
    ax_vel.set_aspect("equal", "box")

    ax_unused.axis("off")
    fig.tight_layout()

    if show:
        plt.show()
        return None
    return axes
