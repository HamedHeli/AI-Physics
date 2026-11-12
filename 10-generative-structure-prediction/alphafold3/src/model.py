"""A toy AlphaFold 3: Pairformer trunk + EDM diffusion module (Abramson et al., 2024).

Architecture (AF3 main text Fig. 2a and Supplementary §3-§4, shrunk)
--------------------------------------------------------------------
::

    Input embedder   s_in_i  = one-hot(type_i)                        (N, 3)
                     s_i     = Linear(s_in_i)                         (N, C_S)
                     z_ij    = Linear(s_in_i) + Linear(s_in_j)
                               + Linear(relpos_ij)                     (N, N, C_Z)

    Pairformer       repeat B times (each block has its own weights):
                     z += TriangleMultiplication(outgoing)(z)   "i-k and j-k tell i-j"
                     z += TriangleMultiplication(incoming)(z)   "k-i and k-j tell i-j"
                     z += Transition(z)
                     s += AttentionWithPairBias(s, z)           bias_h(i,j) = Linear(z_ij)
                     s += Transition(s)

    Diffusion        given noisy coordinates x~ (N, 2) at noise level sigma,
    module           EDM preconditioning (Karras et al., 2022):
                       c_in   = 1 / sqrt(sigma^2 + sd^2)
                       c_skip = sd^2 / (sigma^2 + sd^2)
                       c_out  = sigma sd / sqrt(sigma^2 + sd^2)
                       c_noise = log(sigma / sd) / 4
                     a_i = Linear(s_i, s_in_i) + Linear(Fourier(c_noise))
                           + Linear(c_in x~_i)                         (N, C_A)
                     repeat D times: a += AttentionWithPairBias(a, z); a += Transition(a)
                     r_i = Linear(LayerNorm(a_i))                     (N, 2)
                     x^  = c_skip x~ + c_out r                        denoised coordinates

What AF3 has and this toy leaves out: the MSA module and templates (there is no
evolution here), triangle *attention* (only triangle multiplication is kept),
the atom-level encoder/decoder (one atom per token here), adaptive LayerNorm
conditioning, and the confidence head. What it keeps is the part that is new in
AF3: a pair representation feeding a **non-equivariant** denoiser that works on
raw coordinates, with no frames and no torsion angles.

``apply_regression`` is the AF2-flavoured baseline used in the notebook: the
same trunk, but a deterministic head ``x = Linear(LayerNorm(s))`` trained with an
alignment-invariant MSE. It learns a *map* where AF3 learns a *denoiser*.

Design notes
------------
* Pure JAX, no NN framework, like the rest of this repository: parameters are a
  nested dict of ``(W, b)`` tuples.
* LayerNorm has no learned scale/offset, to keep the parameter list short.
* :func:`apply_with_trace` returns every intermediate tensor for the notebook.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

N_TYPES = 3
REL_CLIP = 4
N_REL = 2 * REL_CLIP + 2   # offsets -4..4, plus one bin for "involves the ligand"


class AF3Config(NamedTuple):
    """Static hyperparameters (AF3 values in brackets)."""

    c_s: int = 16          # single channels [384]
    c_z: int = 8           # pair channels [128]
    c_a: int = 32          # diffusion token channels [768]
    heads: int = 2         # attention heads [16]
    blocks: int = 2        # Pairformer blocks [48]
    diff_layers: int = 3   # diffusion transformer blocks [24]
    n_fourier: int = 16    # noise-level Fourier features [256]
    sigma_data: float = 1.3  # [16 Angstrom]


# --------------------------------------------------------------------------- utils

def linear_init(key: Array, i: int, o: int, scale: float = 1.0) -> tuple[Array, Array]:
    return (jax.random.normal(key, (i, o)) * scale / jnp.sqrt(i), jnp.zeros((o,)))


def lin(p: tuple[Array, Array], x: Array) -> Array:
    return x @ p[0] + p[1]


def layer_norm(x: Array, eps: float = 1e-5) -> Array:
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)


def relpos_features(n_tokens: int) -> Array:
    """One-hot relative position ``clip(j - i)``; pairs with the ligand get their own bin."""
    idx = jnp.arange(n_tokens)
    off = jnp.clip(idx[None] - idx[:, None], -REL_CLIP, REL_CLIP) + REL_CLIP
    lig = (idx[:, None] == n_tokens - 1) | (idx[None] == n_tokens - 1)
    b = jnp.where(lig, N_REL - 1, off)
    return jax.nn.one_hot(b, N_REL)


def fourier_embed(c_noise: Array, n: int) -> Array:
    """AF3 uses ``cos(2 pi (t w + b))`` with fixed random ``w, b``; we use fixed frequencies."""
    freqs = 2.0 ** jnp.linspace(-1, 3, n // 2)
    return jnp.concatenate([jnp.sin(freqs * c_noise), jnp.cos(freqs * c_noise)])


# --------------------------------------------------------------------------- blocks

def init_tri(key, c_z):
    ks = jax.random.split(key, 6)
    return {"a": linear_init(ks[0], c_z, c_z), "ag": linear_init(ks[1], c_z, c_z),
            "b": linear_init(ks[2], c_z, c_z), "bg": linear_init(ks[3], c_z, c_z),
            "o": linear_init(ks[4], c_z, c_z, 0.5), "og": linear_init(ks[5], c_z, c_z)}


def triangle_mult(p, z: Array, outgoing: bool, trace: dict | None = None, name: str = "") -> Array:
    """AF2/AF3 triangle multiplicative update (Supplementary Alg. 12/13)."""
    zn = layer_norm(z)
    a = jax.nn.sigmoid(lin(p["ag"], zn)) * lin(p["a"], zn)
    b = jax.nn.sigmoid(lin(p["bg"], zn)) * lin(p["b"], zn)
    if outgoing:
        t = jnp.einsum("ikc,jkc->ijc", a, b)      # edges i->k and j->k close the triangle
    else:
        t = jnp.einsum("kic,kjc->ijc", a, b)      # edges k->i and k->j
    upd = jax.nn.sigmoid(lin(p["og"], zn)) * lin(p["o"], layer_norm(t))
    if trace is not None:
        trace[name + "/a"], trace[name + "/b"], trace[name + "/update"] = a, b, upd
    return upd


def init_transition(key, c, mult=2):
    k1, k2 = jax.random.split(key)
    return [linear_init(k1, c, mult * c), linear_init(k2, mult * c, c, 0.5)]


def transition(p, x):
    return lin(p[1], jax.nn.relu(lin(p[0], layer_norm(x))))


def init_attn(key, c, c_z, heads):
    ks = jax.random.split(key, 6)
    return {"q": linear_init(ks[0], c, c), "k": linear_init(ks[1], c, c),
            "v": linear_init(ks[2], c, c), "g": linear_init(ks[3], c, c),
            "o": linear_init(ks[4], c, c, 0.5), "bias": linear_init(ks[5], c_z, heads)}


def attention_pair_bias(p, x: Array, z: Array, heads: int, trace: dict | None = None,
                        name: str = "") -> Array:
    """Self-attention whose logits get a per-head bias from the pair representation."""
    n, c = x.shape
    d = c // heads
    xn = layer_norm(x)
    q = lin(p["q"], xn).reshape(n, heads, d)
    k = lin(p["k"], xn).reshape(n, heads, d)
    v = lin(p["v"], xn).reshape(n, heads, d)
    bias = lin(p["bias"], layer_norm(z))                       # (N, N, H)
    logits = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(d) + bias.transpose(2, 0, 1)
    w = jax.nn.softmax(logits, axis=-1)                        # (H, N, N)
    out = jnp.einsum("hij,jhd->ihd", w, v).reshape(n, c)
    out = jax.nn.sigmoid(lin(p["g"], xn)) * out
    if trace is not None:
        trace[name + "/pair_bias"], trace[name + "/weights"] = bias.transpose(2, 0, 1), w
    return lin(p["o"], out)


# --------------------------------------------------------------------------- params

def init_params(key: Array, cfg: AF3Config = AF3Config()) -> dict:
    ks = iter(jax.random.split(key, 64))
    p = {
        "embed_s": linear_init(next(ks), N_TYPES, cfg.c_s),
        "embed_zi": linear_init(next(ks), N_TYPES, cfg.c_z),
        "embed_zj": linear_init(next(ks), N_TYPES, cfg.c_z),
        "embed_rel": linear_init(next(ks), N_REL, cfg.c_z),
        "pairformer": [
            {"tri_out": init_tri(next(ks), cfg.c_z), "tri_in": init_tri(next(ks), cfg.c_z),
             "z_trans": init_transition(next(ks), cfg.c_z),
             "attn": init_attn(next(ks), cfg.c_s, cfg.c_z, cfg.heads),
             "s_trans": init_transition(next(ks), cfg.c_s)}
            for _ in range(cfg.blocks)
        ],
        "diff_cond": linear_init(next(ks), cfg.c_s + N_TYPES, cfg.c_a),
        "diff_noise": linear_init(next(ks), cfg.n_fourier, cfg.c_a),
        "diff_pos": linear_init(next(ks), 2, cfg.c_a),
        "diffusion": [
            {"attn": init_attn(next(ks), cfg.c_a, cfg.c_z, cfg.heads),
             "trans": init_transition(next(ks), cfg.c_a)}
            for _ in range(cfg.diff_layers)
        ],
        "diff_out": linear_init(next(ks), cfg.c_a, 2, 0.1),
        "reg_out": linear_init(next(ks), cfg.c_s, 2),     # AF2-style baseline head
    }
    return p


# --------------------------------------------------------------------------- forward

def trunk(params: dict, types: Array, cfg: AF3Config = AF3Config(), trace: dict | None = None):
    """Input embedder + Pairformer. Returns ``(s_in, s, z)``."""
    n = types.shape[0]
    s_in = jax.nn.one_hot(types, N_TYPES)
    s = lin(params["embed_s"], s_in)
    z = (lin(params["embed_zi"], s_in)[:, None] + lin(params["embed_zj"], s_in)[None]
         + lin(params["embed_rel"], relpos_features(n)))
    if trace is not None:
        trace["s_in"], trace["s0"], trace["z0"] = s_in, s, z
    for b, bp in enumerate(params["pairformer"]):
        nm = f"block{b}"
        z = z + triangle_mult(bp["tri_out"], z, True, trace, nm + "/tri_out")
        z = z + triangle_mult(bp["tri_in"], z, False, trace, nm + "/tri_in")
        z = z + transition(bp["z_trans"], z)
        s = s + attention_pair_bias(bp["attn"], s, z, cfg.heads, trace, nm + "/attn")
        s = s + transition(bp["s_trans"], s)
        if trace is not None:
            trace[nm + "/z"], trace[nm + "/s"] = z, s
    return s_in, s, z


def precond(sigma: Array, sd: float):
    """EDM preconditioning coefficients."""
    c_in = 1.0 / jnp.sqrt(sigma**2 + sd**2)
    c_skip = sd**2 / (sigma**2 + sd**2)
    c_out = sigma * sd / jnp.sqrt(sigma**2 + sd**2)
    c_noise = jnp.log(sigma / sd) / 4.0
    return c_in, c_skip, c_out, c_noise


def denoise(params: dict, s_in: Array, s: Array, z: Array, x_noisy: Array, sigma: Array,
            cfg: AF3Config = AF3Config(), trace: dict | None = None) -> Array:
    """The diffusion module: noisy coordinates at level ``sigma`` -> denoised coordinates."""
    c_in, c_skip, c_out, c_noise = precond(sigma, cfg.sigma_data)
    a = (lin(params["diff_cond"], jnp.concatenate([s, s_in], -1))
         + lin(params["diff_noise"], fourier_embed(c_noise, cfg.n_fourier))
         + lin(params["diff_pos"], c_in * x_noisy))
    if trace is not None:
        trace["diff/a0"] = a
    for l, lp in enumerate(params["diffusion"]):
        a = a + attention_pair_bias(lp["attn"], a, z, cfg.heads, trace, f"diff{l}/attn")
        a = a + transition(lp["trans"], a)
    r = lin(params["diff_out"], layer_norm(a))
    x_hat = c_skip * x_noisy + c_out * r
    if trace is not None:
        trace.update({"diff/r": r, "diff/x_hat": x_hat, "diff/c_in": c_in,
                      "diff/c_skip": c_skip, "diff/c_out": c_out, "diff/c_noise": c_noise})
    return x_hat


def apply_regression(params: dict, types: Array, cfg: AF3Config = AF3Config()) -> Array:
    """AF2-flavoured baseline: one deterministic structure from the trunk."""
    _, s, _ = trunk(params, types, cfg)
    return lin(params["reg_out"], layer_norm(s))


def apply_with_trace(params: dict, types: Array, x_noisy: Array, sigma: float,
                     cfg: AF3Config = AF3Config()) -> dict:
    """Trunk + one denoising call, returning every intermediate tensor."""
    tr: dict = {}
    s_in, s, z = trunk(params, types, cfg, tr)
    tr["x_noisy"], tr["sigma"] = x_noisy, sigma
    denoise(params, s_in, s, z, x_noisy, jnp.asarray(sigma), cfg, tr)
    return tr
