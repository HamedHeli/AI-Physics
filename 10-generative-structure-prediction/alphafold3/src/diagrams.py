"""Structure diagrams of the toy AlphaFold 3: every layer, every block, both algorithms.

Four figures, all drawn from the sizes in :class:`AF3Config` so they stay in sync with
``model.py``:

* :func:`network_overview`: the whole pipeline with tensor shapes.
* :func:`layer_panels`: every small MLP / Linear as neurons and weights, in the same
  style as the GNS notebook (one panel per learned map).
* :func:`block_panels`: Pairformer block, triangle multiplication, attention with
  pair bias, and the diffusion module, as tensor-flow diagrams.
* :func:`algorithm_panels`: one training step and the sampler (AF3 Alg. 18) as
  pseudocode next to the code that implements them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

from .model import N_REL, N_TYPES, AF3Config

BLUE, ORANGE, GREEN, PURPLE, GREY, INK = "#2F5F8A", "#B4541F", "#3C8D5A", "#8172B3", "#6F7580", "#1B2433"
FILL = {"s": "#DCE5EE", "z": "#F2DFD3", "x": "#DDEBDD", "op": "#FFFFFF", "loop": "#F3F1EA"}


# --------------------------------------------------------------------------- neuron drawings

def mlp_diagram(ax, sizes, in_labels=None, out_labels=None, title="", hidden_label="ReLU"):
    """Neuron-and-weight drawing of a small MLP (one column of circles per layer).

    Same drawing as ``05-graph-neural-networks/gnn/src/visualize.py``; a two-entry
    ``sizes`` such as ``[3, 16]`` draws a single Linear layer.
    """
    xs = np.linspace(0, 1, len(sizes))
    ys = [np.linspace(0.95, 0.05, n) if n > 1 else np.array([0.5]) for n in sizes]
    for k in range(len(sizes) - 1):
        for y0 in ys[k]:
            for y1 in ys[k + 1]:
                ax.plot([xs[k], xs[k + 1]], [y0, y1], color="0.6", lw=0.3, alpha=0.5, zorder=0)
    colors = ["#4C72B0"] + ["#8172B3"] * (len(sizes) - 2) + ["#C44E52"]
    for k, (x, y) in enumerate(zip(xs, ys)):
        s = 60 if len(y) <= 10 else (30 if len(y) <= 20 else 14)
        ax.scatter(np.full_like(y, x), y, s=s, color=colors[k], zorder=2, edgecolor="white", lw=0.5)
    for labels, k, dx, ha in ((in_labels, 0, -0.03, "right"), (out_labels, -1, 0.03, "left")):
        if labels is not None:
            fs = 8 if len(labels) <= 10 else 6.5
            for y, lab in zip(ys[k], labels):
                ax.text(xs[k] + dx, y, lab, ha=ha, va="center", fontsize=fs)
    names = ["input"] + [f"hidden\n{n} units, {hidden_label}" for n in sizes[1:-1]] + ["output\n(linear)"]
    for x, n, name in zip(xs, sizes, names):
        ax.text(x, -0.06, f"{name}\n[{n}]" if "hidden" not in name else name, ha="center", va="top", fontsize=8)
    for k in range(len(sizes) - 1):
        ax.text((xs[k] + xs[k + 1]) / 2, 1.0, f"W{k + 1}: {sizes[k]}×{sizes[k + 1]}", ha="center",
                va="bottom", fontsize=8, color="0.3")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.08)
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def _group_labels(groups):
    """One label per group (at its first neuron), blanks for the rest."""
    out = []
    for name, n in groups:
        out += [f"{name} [{n}]"] + [""] * (n - 1)
    return out


def layer_panels(cfg: AF3Config = AF3Config()):
    """Every learned map of the toy, one panel each (2 rows: trunk, then diffusion module)."""
    rel = [f"rel {o:+d}" for o in range(-4, 5)] + ["ligand"]
    types = ["H", "P", "L"]
    s_lab = [f"s{i}" for i in range(cfg.c_s)]
    z_lab = [f"z{i}" for i in range(cfg.c_z)]
    a_lab = [f"a{i}" for i in range(cfg.c_a)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    mlp_diagram(axes[0, 0], [N_TYPES, cfg.c_s], types, s_lab,
                "single embedder\none-hot type of token i → s_i")
    mlp_diagram(axes[0, 1], [2 * N_TYPES + N_REL, cfg.c_z],
                [t + "_i" for t in types] + [t + "_j" for t in types] + rel, z_lab,
                "pair embedder\n[type_i, type_j, relpos_ij] → z_ij")
    mlp_diagram(axes[0, 2], [cfg.c_z, 2 * cfg.c_z, cfg.c_z], z_lab, z_lab,
                "pair transition (in every Pairformer block)\nLayerNorm(z_ij) → update of z_ij")
    mlp_diagram(axes[0, 3], [cfg.c_s, 2 * cfg.c_s, cfg.c_s], s_lab, s_lab,
                "single transition (in every Pairformer block)\nLayerNorm(s_i) → update of s_i")
    mlp_diagram(axes[1, 0], [cfg.c_z, cfg.heads], z_lab, [f"head {h}" for h in range(cfg.heads)],
                "pair bias (in every attention)\nz_ij → one logit bias per head")
    n_in = cfg.c_s + N_TYPES + cfg.n_fourier + 2
    mlp_diagram(axes[1, 1], [n_in, cfg.c_a],
                _group_labels([("s_i", cfg.c_s), ("type", N_TYPES), ("Fourier", cfg.n_fourier),
                               ("c_in·x̃", 2)]), None,
                "diffusion input (sum of 3 Linears)\n[s_i, type_i, Fourier(c_noise), c_in·x̃_i] → a_i")
    mlp_diagram(axes[1, 2], [cfg.c_a, 2 * cfg.c_a, cfg.c_a], None, None,
                "diffusion transition (in every diffusion block)\nLayerNorm(a_i) → update of a_i")
    mlp_diagram(axes[1, 3], [cfg.c_a, 2], None, ["r_x", "r_y"],
                "output head\nLayerNorm(a_i) → r_i, then x̂ = c_skip·x̃ + c_out·r")
    fig.text(0.01, 0.97, "Trunk (computed once per sequence)", fontsize=13, weight="bold", color=BLUE)
    fig.text(0.01, 0.49, "Diffusion module (called once per noise level)", fontsize=13, weight="bold",
             color=ORANGE)
    fig.text(0.99, 0.005, f"Toy sizes: C_s={cfg.c_s}, C_z={cfg.c_z}, C_a={cfg.c_a}, heads={cfg.heads}. "
             "AF3: 384, 128, 768, 16. Each map is applied to every token (or every pair) with the same weights.",
             ha="right", fontsize=9, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=4)
    return fig


# --------------------------------------------------------------------------- box drawings

def _box(ax, x, y, w, h, text, fill=FILL["op"], edge=INK, fs=9, weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fill, ec=edge, lw=1.3, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, color=INK)


def _arrow(ax, p, q, color=GREY, text=None, fs=8, side="right", rad=0.0, ls="-"):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, ls=ls,
                                          connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        dx = 0.012 if side == "right" else -0.012
        ax.text(mx + dx, my, text, ha="left" if side == "right" else "right", va="center", fontsize=fs,
                color=color)


def _plus(ax, x, y, r=0.018):
    ax.add_patch(plt.Circle((x, y), r, fc="white", ec=INK, lw=1.2))
    ax.text(x, y, "+", ha="center", va="center", fontsize=10)


def _frame(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", loc="left")


def network_overview(cfg: AF3Config = AF3Config(), n_tok: int = 9):
    """The whole pipeline with tensor shapes (N = 9: 8 beads + 1 ligand atom)."""
    N = n_tok
    fig, ax = plt.subplots(figsize=(16, 7))
    _frame(ax, f"Toy AlphaFold 3: the whole network (N = {N} tokens: 8 chain beads + 1 ligand atom)")
    # inputs
    _box(ax, 0.07, 0.62, 0.11, 0.1, f"sequence\ntypes (N,)\nH / P / L", FILL["op"])
    _box(ax, 0.22, 0.62, 0.13, 0.16, "Input embedder\n\none-hot, relpos\n3 Linears", FILL["loop"], weight="bold")
    _arrow(ax, (0.125, 0.62), (0.155, 0.62))
    _box(ax, 0.38, 0.75, 0.1, 0.08, f"s  ({N}, {cfg.c_s})\nsingle", FILL["s"])
    _box(ax, 0.38, 0.49, 0.1, 0.08, f"z  ({N}, {N}, {cfg.c_z})\npair", FILL["z"])
    _arrow(ax, (0.285, 0.66), (0.33, 0.74))
    _arrow(ax, (0.285, 0.58), (0.33, 0.50))
    # pairformer
    _box(ax, 0.55, 0.62, 0.16, 0.34, "", FILL["loop"])
    ax.text(0.55, 0.765, f"Pairformer  ×{cfg.blocks}", ha="center", fontsize=11, weight="bold")
    ax.text(0.55, 0.735, "(AF3: ×48)", ha="center", fontsize=8, color=GREY)
    _box(ax, 0.55, 0.66, 0.13, 0.05, "triangle mult. out / in", FILL["z"], fs=8)
    _box(ax, 0.55, 0.60, 0.13, 0.05, "pair transition", FILL["z"], fs=8)
    _box(ax, 0.55, 0.54, 0.13, 0.05, "attention + pair bias", FILL["s"], fs=8)
    _box(ax, 0.55, 0.48, 0.13, 0.05, "single transition", FILL["s"], fs=8)
    _arrow(ax, (0.43, 0.75), (0.47, 0.70))
    _arrow(ax, (0.43, 0.49), (0.47, 0.54))
    # trunk output
    _box(ax, 0.55, 0.3, 0.16, 0.08, f"s ({N}, {cfg.c_s}),  z ({N}, {N}, {cfg.c_z})\nconditioning, computed once",
         "#FFFFFF", fs=8.5)
    _arrow(ax, (0.55, 0.45), (0.55, 0.34))
    # diffusion
    _box(ax, 0.83, 0.62, 0.22, 0.34, "", FILL["loop"])
    ax.text(0.83, 0.765, "Diffusion module", ha="center", fontsize=11, weight="bold")
    ax.text(0.83, 0.735, "called once per sampling step", ha="center", fontsize=8, color=GREY)
    _box(ax, 0.83, 0.67, 0.19, 0.05, f"EDM: c_in, c_skip, c_out, c_noise(σ)", FILL["op"], fs=8)
    _box(ax, 0.83, 0.61, 0.19, 0.05, f"a = Linear(s, type) + Linear(Fourier) + Linear(c_in x̃)", FILL["x"], fs=7.5)
    _box(ax, 0.83, 0.55, 0.19, 0.05, f"×{cfg.diff_layers}: attention(+ z bias), transition", FILL["x"], fs=8)
    _box(ax, 0.83, 0.49, 0.19, 0.05, "x̂ = c_skip x̃ + c_out Linear(a)", FILL["x"], fs=8)
    _arrow(ax, (0.63, 0.3), (0.80, 0.445), text="conditions every call", side="right")
    # sampling loop
    _box(ax, 0.83, 0.17, 0.2, 0.09, f"noise x_T ~ N(0, σ_max²)  ({N}, 2)\n→ 40 denoising steps →", FILL["op"], fs=8.5)
    _arrow(ax, (0.80, 0.215), (0.80, 0.445), rad=0.0)
    _arrow(ax, (0.86, 0.445), (0.86, 0.215), text="x̃ → x̂, step", side="right")
    _box(ax, 0.83, 0.04, 0.15, 0.05, f"structure x_0  ({N}, 2)", FILL["x"], weight="bold", fs=9)
    _arrow(ax, (0.83, 0.125), (0.83, 0.065))
    # AF2-style head
    _box(ax, 0.3, 0.17, 0.2, 0.09, f"AF2-style baseline head\nx = Linear(LayerNorm(s))  ({N}, 2)", FILL["op"],
         fs=8.5, ls="--")
    _arrow(ax, (0.47, 0.3), (0.40, 0.21), ls="--", text="same trunk", side="left")
    ax.text(0.3, 0.06, "one deterministic answer\n(regression, no noise)", ha="center", fontsize=8, color=GREY)
    # legend
    for k, (lab, c) in enumerate([("single track (per token)", FILL["s"]), ("pair track (per pair i, j)", FILL["z"]),
                                  ("coordinates", FILL["x"])]):
        ax.add_patch(FancyBboxPatch((0.02, 0.33 - 0.05 * k), 0.025, 0.03, boxstyle="round,pad=0.003", fc=c, ec=INK))
        ax.text(0.055, 0.345 - 0.05 * k, lab, va="center", fontsize=9)
    return fig


def block_panels(cfg: AF3Config = AF3Config()):
    """Inside the blocks: Pairformer block, triangle multiplication, attention with pair bias, diffusion."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    _pairformer_block(axes[0, 0], cfg)
    _triangle_mult(axes[0, 1], cfg)
    _attention(axes[1, 0], cfg)
    _diffusion_module(axes[1, 1], cfg)
    fig.tight_layout(h_pad=3)
    return fig


def _pairformer_block(ax, cfg):
    _frame(ax, f"(a) One Pairformer block (residual everywhere; ×{cfg.blocks} here, ×48 in AF3)")
    zx, sx = 0.3, 0.72
    ax.text(zx, 0.96, f"pair z  (N, N, {cfg.c_z})", ha="center", fontsize=11, weight="bold", color=ORANGE)
    ax.text(sx, 0.96, f"single s  (N, {cfg.c_s})", ha="center", fontsize=11, weight="bold", color=BLUE)
    steps_z = [(0.8, "Triangle multiplication, outgoing\nedges i→k and j→k tell i→j"),
               (0.62, "Triangle multiplication, incoming\nedges k→i and k→j tell i→j"),
               (0.44, f"Transition\nLN → {cfg.c_z}→{2 * cfg.c_z} ReLU →{cfg.c_z}")]
    y_prev = 0.93
    for y, t in steps_z:
        _box(ax, zx - 0.11, y, 0.3, 0.1, t, FILL["z"], fs=9)
        ax.plot([zx + 0.08, zx + 0.08], [y_prev - 0.01, y - 0.075], color=INK, lw=1.4)
        _arrow(ax, (zx + 0.08, y + 0.04), (zx + 0.04, y + 0.04))
        _plus(ax, zx + 0.08, y - 0.075)
        _arrow(ax, (zx - 0.11, y - 0.05), (zx + 0.062, y - 0.075), rad=0.25)
        y_prev = y - 0.075
    ax.plot([zx + 0.08, zx + 0.08], [y_prev - 0.018, 0.2], color=INK, lw=1.4)
    _arrow(ax, (zx + 0.08, 0.2), (zx + 0.08, 0.1))
    ax.text(zx + 0.08, 0.07, "z'", ha="center", fontsize=12, weight="bold")
    steps_s = [(0.5, "Attention with pair bias\nbias_h(i,j) = Linear(LN(z'_ij))"),
               (0.3, f"Transition\nLN → {cfg.c_s}→{2 * cfg.c_s} ReLU →{cfg.c_s}")]
    y_prev = 0.93
    for y, t in steps_s:
        _box(ax, sx + 0.1, y, 0.28, 0.1, t, FILL["s"], fs=9)
        ax.plot([sx - 0.08, sx - 0.08], [y_prev - 0.01, y - 0.075], color=INK, lw=1.4)
        _arrow(ax, (sx - 0.08, y + 0.04), (sx - 0.04, y + 0.04))
        _plus(ax, sx - 0.08, y - 0.075)
        _arrow(ax, (sx + 0.1, y - 0.05), (sx - 0.062, y - 0.075), rad=-0.25)
        y_prev = y - 0.075
    ax.plot([sx - 0.08, sx - 0.08], [y_prev - 0.018, 0.2], color=INK, lw=1.4)
    _arrow(ax, (sx - 0.08, 0.2), (sx - 0.08, 0.1))
    ax.text(sx - 0.08, 0.07, "s'", ha="center", fontsize=12, weight="bold")
    _arrow(ax, (zx + 0.08, 0.27), (sx - 0.04, 0.5), color=ORANGE, rad=-0.2)
    ax.text(zx + 0.1, 0.24, "z' biases the attention", fontsize=9, color=ORANGE)
    ax.text(0.5, 0.005, "Information flows z → s only (through the bias). AF2's Evoformer also had an MSA track\n"
            "and s → z via the outer product mean; AF3 dropped both from the main trunk.",
            ha="center", fontsize=9, color=GREY)


def _triangle_mult(ax, cfg):
    _frame(ax, "(b) Triangle multiplication, outgoing (AF3 Alg. 12)")
    c = cfg.c_z
    _box(ax, 0.5, 0.93, 0.22, 0.06, f"z  (N, N, {c})", FILL["z"], weight="bold")
    _box(ax, 0.5, 0.83, 0.16, 0.05, "LayerNorm", FILL["op"])
    _arrow(ax, (0.5, 0.9), (0.5, 0.855))
    for x, nm in ((0.22, "a"), (0.5, "b"), (0.78, "g")):
        txt = (f"{nm} = σ(Linear(z)) ⊙ Linear(z)\n(N, N, {c})" if nm != "g" else f"gate g = σ(Linear(z))\n(N, N, {c})")
        _box(ax, x, 0.7, 0.24, 0.08, txt, FILL["z"], fs=9)
        _arrow(ax, (0.5, 0.805), (x, 0.74))
    _box(ax, 0.36, 0.53, 0.42, 0.09, "t_ij = Σ_k  a_ik · b_jk   (per channel)\n= a · bᵀ for each of the "
         f"{c} channels  (N, N, {c})", FILL["op"], fs=9.5, weight="bold")
    _arrow(ax, (0.22, 0.66), (0.28, 0.575))
    _arrow(ax, (0.5, 0.66), (0.44, 0.575))
    _box(ax, 0.36, 0.4, 0.3, 0.06, "LayerNorm → Linear", FILL["op"])
    _arrow(ax, (0.36, 0.485), (0.36, 0.43))
    _box(ax, 0.5, 0.28, 0.22, 0.06, "⊙", FILL["op"], fs=14)
    _arrow(ax, (0.36, 0.37), (0.46, 0.31))
    _arrow(ax, (0.78, 0.66), (0.56, 0.31), rad=-0.15)
    _box(ax, 0.5, 0.17, 0.22, 0.06, f"update of z  (N, N, {c})", FILL["z"], weight="bold")
    _arrow(ax, (0.5, 0.25), (0.5, 0.2))
    # triangle sketch
    P = {"i": (0.8, 0.5), "j": (0.94, 0.5), "k": (0.87, 0.36)}
    ax.add_patch(Polygon([P["i"], P["j"], P["k"]], closed=True, fill=False, ec=GREY, lw=1, ls=":"))
    _arrow(ax, P["i"], P["j"], color=ORANGE)
    _arrow(ax, P["i"], P["k"], color=BLUE)
    _arrow(ax, P["j"], P["k"], color=BLUE)
    for n, (x, y) in P.items():
        ax.add_patch(plt.Circle((x, y), 0.018, fc=INK))
        ax.text(x, y, n, color="white", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0.87, 0.28, "edge i→j is updated from\ni→k and j→k, for every k", ha="center", fontsize=8.5)
    ax.text(0.5, 0.04, "Incoming version: t_ij = Σ_k a_ki · b_kj.   Cost O(N³ C): the most expensive op in the trunk.",
            ha="center", fontsize=9, color=GREY)


def _attention(ax, cfg):
    _frame(ax, "(c) Attention with pair bias (used in the Pairformer and the diffusion module)")
    c, h = cfg.c_s, cfg.heads
    d = c // h
    _box(ax, 0.3, 0.93, 0.22, 0.06, f"x  (N, C)   C = {c} or {cfg.c_a}", FILL["s"], weight="bold")
    _box(ax, 0.82, 0.93, 0.2, 0.06, f"z  (N, N, {cfg.c_z})", FILL["z"], weight="bold")
    _box(ax, 0.3, 0.83, 0.16, 0.05, "LayerNorm", FILL["op"])
    _arrow(ax, (0.3, 0.9), (0.3, 0.855))
    for x, nm in ((0.1, "q"), (0.3, "k"), (0.58, "v")):
        _box(ax, x, 0.71, 0.16, 0.07, f"{nm} = Linear(x)\n(N, H={h}, d={d})", FILL["s"], fs=8.5)
        _arrow(ax, (0.3, 0.805), (x, 0.745))
    _box(ax, 0.84, 0.71, 0.24, 0.07, f"bias = Linear(LN(z))\n(H={h}, N, N)", FILL["z"], fs=8.5)
    _arrow(ax, (0.82, 0.9), (0.82, 0.745))
    _box(ax, 0.3, 0.56, 0.48, 0.08, "logits_hij = q_hi · k_hj / √d  +  bias_hij\nweights = softmax over j   (H, N, N)",
         FILL["op"], fs=9.5, weight="bold")
    _arrow(ax, (0.1, 0.675), (0.18, 0.6))
    _arrow(ax, (0.3, 0.675), (0.3, 0.6))
    _arrow(ax, (0.82, 0.675), (0.54, 0.58), color=ORANGE)
    _box(ax, 0.45, 0.42, 0.42, 0.07, "out_hi = Σ_j weights_hij · v_hj   → (N, C)", FILL["op"], fs=9.5)
    _arrow(ax, (0.3, 0.52), (0.33, 0.455))
    _arrow(ax, (0.58, 0.675), (0.58, 0.455))
    _box(ax, 0.45, 0.29, 0.42, 0.07, "⊙ σ(Linear(x))  gate,  then Linear  → (N, C)", FILL["op"], fs=9.5)
    _arrow(ax, (0.45, 0.385), (0.45, 0.325))
    _box(ax, 0.45, 0.17, 0.2, 0.06, "update of x", FILL["s"], weight="bold")
    _arrow(ax, (0.45, 0.255), (0.45, 0.2))
    ax.text(0.5, 0.05, "The bias is how geometry steers attention: if z says i and j are close, i attends to j.\n"
            "The diffusion module uses the same op on its token features a, with the trunk's z as the bias.",
            ha="center", fontsize=9, color=GREY)


def _diffusion_module(ax, cfg):
    _frame(ax, "(d) Diffusion module: denoiser D(x̃, σ | s, z)  (EDM preconditioning)")
    _box(ax, 0.15, 0.93, 0.2, 0.06, "noisy x̃  (N, 2)", FILL["x"], weight="bold")
    _box(ax, 0.42, 0.93, 0.14, 0.06, "noise level σ", FILL["op"], weight="bold")
    _box(ax, 0.75, 0.93, 0.3, 0.06, f"trunk: s (N,{cfg.c_s}), type (N,3), z (N,N,{cfg.c_z})", FILL["s"], fs=9,
         weight="bold")
    _box(ax, 0.42, 0.8, 0.38, 0.1, "c_in = 1/√(σ²+σ_d²)     c_skip = σ_d²/(σ²+σ_d²)\n"
         "c_out = σ σ_d/√(σ²+σ_d²)     c_noise = ¼ log(σ/σ_d)", FILL["op"], fs=9)
    _arrow(ax, (0.42, 0.9), (0.42, 0.85))
    _box(ax, 0.42, 0.64, 0.7, 0.09,
         f"a_i = Linear([s_i, type_i]) + Linear(Fourier(c_noise)) + Linear(c_in · x̃_i)     (N, {cfg.c_a})",
         FILL["x"], fs=9.5, weight="bold")
    _arrow(ax, (0.15, 0.9), (0.15, 0.685))
    _arrow(ax, (0.42, 0.75), (0.42, 0.685))
    _arrow(ax, (0.7, 0.9), (0.62, 0.685))
    _box(ax, 0.42, 0.47, 0.6, 0.12, "", FILL["loop"])
    ax.text(0.42, 0.515, f"repeat ×{cfg.diff_layers}  (AF3: 24-block transformer)", ha="center", fontsize=9.5,
            weight="bold")
    ax.text(0.42, 0.455, "a += AttentionWithPairBias(a, z)\na += Transition(a)", ha="center", va="center", fontsize=9)
    _arrow(ax, (0.42, 0.595), (0.42, 0.53))
    _arrow(ax, (0.85, 0.9), (0.73, 0.47), color=ORANGE, text="pair bias", rad=-0.2)
    _box(ax, 0.42, 0.32, 0.38, 0.06, "r_i = Linear(LayerNorm(a_i))     (N, 2)", FILL["x"], fs=9.5)
    _arrow(ax, (0.42, 0.41), (0.42, 0.35))
    _box(ax, 0.42, 0.19, 0.42, 0.07, "x̂ = c_skip · x̃  +  c_out · r", FILL["x"], fs=11, weight="bold")
    _arrow(ax, (0.42, 0.29), (0.42, 0.225))
    ax.plot([0.06, 0.06, 0.06], [0.9, 0.55, 0.19], color=GREY, lw=1.4)
    ax.plot([0.08, 0.06], [0.9, 0.9], color=GREY, lw=1.4)
    _arrow(ax, (0.06, 0.19), (0.21, 0.19))
    ax.text(0.07, 0.3, "skip:\nx̃ itself", fontsize=8.5, color=GREY)
    ax.text(0.5, 0.04, "Small σ: c_skip ≈ 1, the net only adds a small correction (local chemistry).\n"
            "Large σ: c_skip ≈ 0, the net predicts the whole structure (global fold). No frames, no equivariance.",
            ha="center", fontsize=9, color=GREY)


# --------------------------------------------------------------------------- algorithms

TRAIN_ALGO = """\
Training step  (training.diffusion_loss)
------------------------------------------------
input: types (N,), one true structure x0 (N, 2)

s_in, s, z = trunk(types)          # once per structure
repeat for n_sigma = 8 noise levels:      (AF3: 48)
    sigma = sigma_d * exp(-1.2 + 1.5 * randn)
    eps   = randn(N, 2)
    x_noisy = x0 + sigma * eps
    x_hat   = denoise(x_noisy, sigma | s_in, s, z)
    target  = align(x0 onto x_hat)   # rigid fit,
                                     # no gradient
    w = (sigma^2 + sigma_d^2) / (sigma * sigma_d)^2
    loss += w * mean_i |x_hat_i - target_i|^2
loss /= n_sigma
params -= Adam(grad(loss))       # end to end:
                                 # trunk + denoiser
"""

SAMPLE_ALGO = """\
Sampling  (training.sample, AF3 Alg. 18)
------------------------------------------------
input: types (N,).   gamma0=0.8, gamma_min=1,
       noise_scale=1.003, step_scale=1.5

s_in, s, z = trunk(types)          # once
c_0 > c_1 > ... > c_T: rho=7 schedule, T = 40
x = c_0 * randn(N, 2)              # pure noise
for t = 1 .. T:
    x = random rotation + shift of x
    gamma = gamma0 if c_t > gamma_min else 0
    t_hat = c_{t-1} * (gamma + 1)  # add a little
    x_noisy = x + noise_scale             # noise
              * sqrt(t_hat^2 - c_{t-1}^2) * randn
    x_den = denoise(x_noisy, t_hat | s_in, s, z)
    delta = (x_noisy - x_den) / t_hat     # score
    x = x_noisy + step_scale * (c_t - t_hat) * delta
return x - mean(x)                 # one sample;
                                   # new seed = new
                                   # structure
"""


def algorithm_panels():
    """Training step and sampler as pseudocode, line for line with ``training.py``."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
    for ax, txt, col, title in ((axes[0], TRAIN_ALGO, BLUE, "Learn the denoiser"),
                                (axes[1], SAMPLE_ALGO, ORANGE, "Use it: noise → structure")):
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.9, boxstyle="round,pad=0.01", fc="#FAFAF7", ec=col, lw=2))
        ax.text(0.03, 0.87, txt, family="monospace", fontsize=10.5, va="top", color=INK)
        ax.set_title(title, fontsize=13, weight="bold", color=col, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
