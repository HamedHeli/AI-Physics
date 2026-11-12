"""Drawing helpers: toy molecules, annotated heatmaps, denoising strips."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .physics import H, L, P

COLORS = {H: "#B4541F", P: "#2F5F8A", L: "#3C8D5A"}
NAMES = {H: "H", P: "P", L: "L"}


def token_labels(types) -> list[str]:
    t = np.asarray(types)
    return [f"{i}{NAMES[int(k)]}" for i, k in enumerate(t)]


def draw_molecule(ax, x, types, title: str | None = None, alpha: float = 1.0, labels: bool = True,
                  ghost=None, lim: float = 3.8, size: float = 260):
    """Chain as bonded beads (H orange, P blue), ligand as a green diamond.

    ``ghost`` (optional, same shape as ``x``) is drawn faintly underneath, e.g. the true structure.
    """
    x = np.asarray(x)
    t = np.asarray(types)
    if ghost is not None:
        g = np.asarray(ghost)
        ax.plot(g[:-1, 0], g[:-1, 1], "-", color="#BBBBBB", lw=6, zorder=0, solid_capstyle="round")
        ax.scatter(g[-1, 0], g[-1, 1], marker="D", s=size * 0.8, color="#CCCCCC", zorder=0)
    ax.plot(x[:-1, 0], x[:-1, 1], "-", color="#555555", lw=2, alpha=alpha, zorder=1)
    for i, (p, k) in enumerate(zip(x, t)):
        k = int(k)
        ax.scatter(p[0], p[1], s=size, marker="D" if k == L else "o", color=COLORS[k],
                   edgecolor="white", lw=1.5, alpha=alpha, zorder=2)
        if labels:
            ax.text(p[0], p[1], str(i), color="white", ha="center", va="center", fontsize=8,
                    fontweight="bold", zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def heatmap(ax, m, row_labels=None, col_labels=None, title: str | None = None,
            annotate: bool = True, fmt: str = "{:.2f}", cmap: str = "RdBu_r", vmax: float | None = None,
            fontsize: int = 7):
    """Symmetric-colour heatmap with optional values written in the cells."""
    m = np.asarray(m)
    if m.ndim == 1:
        m = m[None]
    v = vmax if vmax is not None else (np.abs(m).max() + 1e-9)
    ax.imshow(m, cmap=cmap, vmin=-v if cmap == "RdBu_r" else 0, vmax=v, aspect="auto")
    if annotate and m.size <= 400:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=fontsize,
                        color="white" if abs(m[i, j]) > 0.6 * v else "black")
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
    else:
        ax.set_yticks([])
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8, rotation=90 if len(col_labels) > 12 else 0)
    else:
        ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=10)


def pair_channels(z, types, title: str = "", vmax: float | None = None):
    """One heatmap per pair channel: z[:, :, c] is an N x N matrix."""
    z = np.asarray(z)
    c = z.shape[-1]
    cols = min(c, 4)
    rows = int(np.ceil(c / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    labs = token_labels(types)
    v = vmax if vmax is not None else np.abs(z).max()
    for k, ax in enumerate(np.ravel(axes)):
        if k < c:
            heatmap(ax, z[:, :, k], labs, labs, f"channel {k}", annotate=False, vmax=v)
        else:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def denoising_strip(path, dens, sigmas, types, truth=None, frames=(0, 8, 14, 18, 24, 39)):
    """Top: the current sample x at chosen steps. Bottom: the denoiser's guess x^ at that step."""
    fig, axes = plt.subplots(2, len(frames), figsize=(2.6 * len(frames), 5.4))
    for col, f in enumerate(frames):
        lim = max(3.8, float(np.abs(np.asarray(path[f])).max()) + 0.5) if f < 6 else 3.8
        draw_molecule(axes[0, col], path[f], types, f"step {f + 1}: x (σ={float(sigmas[f + 1]):.2g})",
                      labels=False, lim=lim, size=120)
        draw_molecule(axes[1, col], dens[f], types, "denoiser guess x̂", labels=False, size=120)
    axes[0, 0].set_ylabel("sample")
    axes[1, 0].set_ylabel("guess")
    fig.tight_layout()
    return fig
