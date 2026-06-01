"""
plotting.py
-----------
Visualisation functions for ligflow results.
All functions live under ``ligflow.pl``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
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
