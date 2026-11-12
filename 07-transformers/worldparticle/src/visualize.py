"""Plotting helpers for the WorldParticle notebook: particles in the box, kernels, super tokens."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

TOKEN_COLORS = plt.get_cmap("tab20").colors


def matrix(ax, M, row_labels=None, col_labels=None, title="", vmax=None, fmt=None, cmap="RdBu_r",
           annotate=True, fontsize=6):
    """Heatmap of a small real matrix with every value written in its cell (unless ``annotate=False``)."""
    M = np.asarray(M)
    vmax = vmax if vmax is not None else max(float(np.abs(M).max()), 1e-6)
    fmt = fmt or ("{:.3f}" if vmax < 0.1 else "{:.2f}")
    vmin = 0 if cmap in ("magma", "viridis", "Greys", "Blues") else -vmax
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    if annotate:
        for (i, j), v in np.ndenumerate(M):
            dark = abs(v) > 0.6 * vmax if vmin < 0 else v > 0.6 * vmax
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else "black")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels if row_labels is not None else range(M.shape[0]), fontsize=6)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels if col_labels is not None else range(M.shape[1]), fontsize=6)
    ax.set_title(title, fontsize=9)


def box(ax, title="", lim=(-0.02, 1.02)):
    """The unit box, equal axes, no ticks."""
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="k", lw=1.2))
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def particles(ax, x, edges=None, c="tab:blue", s=14, title="", alpha=1.0, edge_alpha=0.35, **kw):
    """Particles (and springs, when ``edges`` is given) in the box."""
    x = np.asarray(x)
    if edges is not None:
        seg = x[np.asarray(edges)]
        for a, b in seg:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.5", lw=0.6, alpha=edge_alpha, zorder=1)
    sc = ax.scatter(x[:, 0], x[:, 1], c=c, s=s, alpha=alpha, zorder=2, lw=0, **kw)
    box(ax, title)
    return sc


def compare_frames(X_true, X_pred, frames, edges=None, title_prefix="", row_titles=("ground truth", "WorldParticle")):
    """Two rows of snapshots: the solver's frames on top, the model's rollout below."""
    fig, axes = plt.subplots(2, len(frames), figsize=(2.3 * len(frames), 4.8))
    for j, t in enumerate(frames):
        particles(axes[0, j], X_true[t], edges, c="0.25", title=f"{title_prefix}frame {t}")
        particles(axes[1, j], X_pred[t], edges, c="tab:orange")
    for r, name in enumerate(row_titles):
        axes[r, 0].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    return fig


def kernel_lattice(ax, W_slice, radius, title="", cmap="RdBu_r"):
    """One input/output channel pair of a kernel as an image over the displacement square ``[-R, R]^2``.

    ``W_slice`` is ``(K, K)``; the kernel between lattice nodes is bilinear, which
    ``imshow(interpolation="bilinear")`` reproduces.
    """
    v = float(np.abs(W_slice).max()) or 1.0
    im = ax.imshow(np.asarray(W_slice).T, origin="lower", extent=(-radius, radius, -radius, radius), cmap=cmap,
                   vmin=-v, vmax=v, interpolation="bilinear")
    t = np.linspace(-1, 1, 200)
    ax.plot(radius * np.cos(np.pi * t), radius * np.sin(np.pi * t), "k:", lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def token_colors(members):
    """Colour of every particle: the index of the super token it belongs to. ``members (n_tokens, N)``."""
    owner = np.asarray(members).argmax(0)
    return [TOKEN_COLORS[i % len(TOKEN_COLORS)] for i in owner], owner


def merge_levels(axes, x, layers, edges=None, s=22):
    """One panel per encoder layer: particles coloured by the (super) token they have been merged into.

    Each token's anchor position is drawn as a cross, sized by its multiplicity.
    """
    x = np.asarray(x)
    particles(axes[0], x, edges, c=[TOKEN_COLORS[i % 20] for i in range(len(x))], s=s,
              title=f"input: {len(x)} tokens")
    for ax, L in zip(axes[1:], layers):
        cols, _ = token_colors(L["members"])
        particles(ax, x, edges, c=cols, s=s, title=f"after merge: {L['pos'].shape[0]} tokens")
        pos, mult = np.asarray(L["pos"]), np.asarray(L["mult"])
        ax.scatter(pos[:, 0], pos[:, 1], marker="x", c="k", s=8 + 4 * mult, lw=1.2, zorder=3)


def zoom_to(ax, x, pad=0.08):
    """Zoom an axis onto a set of particles."""
    x = np.asarray(x)
    lo, hi = x.min(0) - pad, x.max(0) + pad
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)


def cross_attention(ax, x, z_pos, weights, i, edges=None, title=""):
    """Where particle ``i`` looks: super-token anchors sized and coloured by its attention weights."""
    x, z_pos, w = np.asarray(x), np.asarray(z_pos), np.asarray(weights)
    particles(ax, x, edges, c="0.8", s=10)
    ax.scatter(z_pos[:, 0], z_pos[:, 1], s=30 + 900 * w, c=w, cmap="viridis", vmin=0, vmax=max(w.max(), 1e-6),
               ec="k", lw=0.5, zorder=3)
    ax.scatter(*x[i], marker="*", s=160, c="red", ec="k", lw=0.5, zorder=4)
    ax.set_title(title, fontsize=9)


def error_curves(ax, curves, ylabel="position MSE", title=""):
    """Error per frame on a log axis. ``curves``: ``{label: (values, style dict)}``."""
    for label, (vals, style) in curves.items():
        ax.semilogy(np.arange(len(vals)), np.maximum(vals, 1e-14), label=label, **style)
    ax.set_xlabel("frame"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)


def pipeline_diagram(n, cfg):
    """The corrector as a row of boxes whose heights are the number of rows (particles or tokens)."""
    fig, ax = plt.subplots(figsize=(14, 3.6))
    stages = [("X~, V~\n+ attributes", n, "#d9d9d9"), ("tokenizer\n3 kernels", n, "#9ecae1")]
    m = n
    for l in range(cfg.enc_layers):
        m = (m + 1) // 2
        stages.append((f"encoder {l + 1}\nattn + merge", m, "#fdae6b"))
    stages.append(("super tokens", m, "#e6550d"))
    for l in range(cfg.dec_layers):
        stages.append((f"decoder {l + 1}\ncross + self", n, "#a1d99b"))
    stages.append(("head\n(dx, dv)", n, "#31a354"))
    x0, w, gap, hmax = 0.0, 1.0, 0.45, 3.0
    centres = []
    for name, rows, col in stages:
        h = hmax * rows / n
        ax.add_patch(FancyBboxPatch((x0, -h / 2), w, h, boxstyle="round,pad=0.02", fc=col, ec="k", lw=0.8))
        ax.text(x0 + w / 2, hmax / 2 + 0.25, name, ha="center", va="bottom", fontsize=8)
        ax.text(x0 + w / 2, 0, f"{rows}", ha="center", va="center", fontsize=9, fontweight="bold")
        centres.append(x0 + w / 2)
        x0 += w + gap
    for a, b in zip(centres[:-1], centres[1:]):
        ax.annotate("", (b - w / 2 - 0.02, 0), (a + w / 2 + 0.02, 0), arrowprops=dict(arrowstyle="->", lw=0.8))
    # the skip connection: particle tokens bypass the encoder
    i_tok, i_dec = 1, 2 + cfg.enc_layers + 1
    ax.annotate("", (centres[i_dec], -hmax / 2 - 0.05), (centres[i_tok], -hmax / 2 - 0.05),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="tab:green", connectionstyle="arc3,rad=0.25"))
    ax.text((centres[i_tok] + centres[i_dec]) / 2, -hmax / 2 - 1.25, "particle tokens bypass the encoder (queries)",
            ha="center", fontsize=8, color="tab:green")
    ax.set_xlim(-0.2, x0); ax.set_ylim(-hmax / 2 - 1.5, hmax / 2 + 1.0)
    ax.axis("off")
    ax.set_title("box height = number of rows (particles or tokens)", fontsize=9)
    return fig
