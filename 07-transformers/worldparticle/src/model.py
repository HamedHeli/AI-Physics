"""WorldParticle (Wang et al., 2026, "WorldParticle: Unified World Simulation of
Lagrangian Particle Dynamics via Transformer", arXiv:2605.15305), in 2D and small.

One step of the simulator (paper Sec. 3, Eqs. 1-2, Fig. 2)
----------------------------------------------------------
The state of every particle is its position and velocity. One frame forward is
**predict, then correct**::

    predictor (no learning, known forces only, here gravity g):
        V~ = V + dt g
        X~ = X + dt/2 (V + V~)

    corrector (the transformer):
        (dX, dV) = f_theta(X~, V~, attributes, topology, boundary)
        X' = X~ + dX,   V' = V~ + dV

The network never has to learn gravity. It learns the **residual**: everything
the cheap predictor gets wrong (springs, contacts, friction, walls).

The corrector has three stages:

    1. particle tokenizer (Sec. 3.1), local:
         three neighbourhood branches, each a learnable continuous kernel
         (a K x K lattice of weight matrices, bilinear interpolation, compact
         support of radius R):
           particle-particle   neighbours j within R_pp of i,   kernel of x~_j - x~_i
           particle-boundary   wall samples b within R_pb of i, kernel of x_b - x~_i
           topology            mesh neighbours j of i,          kernel of rest_j - rest_i
         h_i = MLP([pp_i, pb_i, topo_i, v~_i, a_i])                        (N, C)

    2. super-token encoder (Sec. 3.2), global and coarse, L_e layers of:
         x = x + SelfAttention(LN(x)) with 2D RoPE on the token positions
         bipartite token merging: split the tokens into even and odd indices,
           pair each even token with its most cosine-similar odd token
           (greedy, one-to-one) and replace each pair by its
           multiplicity-weighted average (features and positions)       N -> N/2
         x = x + MLP(LN(x))
       After L_e layers: N / 2^L_e super tokens with anchor positions.

    3. super-token decoder (Sec. 3.3), back to every particle, L_d layers of:
         y = y + CrossAttention(queries LN(y), keys/values LN(z))   particles <- super tokens
         y = y + SelfAttention(LN(y))                               particles <-> particles
         y = y + MLP(LN(y))
       where y starts as the tokenizer output h (it bypasses the encoder).
       head: MLP(y_i) -> (dx_i, dv_i)

Sizes (paper -> here): token width 1152 -> 32, heads 12 -> 4, encoder layers
6 -> 3, decoder layers 8 -> 2, kernel lattice 4x4x4 -> 4x4, 3D -> 2D, about
157M -> about 60k parameters.

Design notes
------------
* Pure JAX, no NN framework, like the rest of this repository: parameters are
  a pytree, every linear map is ``(W, b)``.
* Missing inputs are zero-filled, as in the paper: the sand has no topology,
  so its adjacency matrix is all zeros and the topology branch returns zeros.
  The architecture does not change.
* The matching inside token merging is an argmax, so it has no gradient; the
  merged *values* do. That is also how ToMe (Bolya et al., 2022) works.
* ``cfg.use_global = False`` drops the encoder and the decoder's cross- and
  self-attention: a purely local model, the paper's "local-only" ablation
  (Fig. 5), close to Deep Lagrangian Fluids.
* :func:`step_with_trace` returns every intermediate tensor. That trace is what
  the notebook visualises.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class WPConfig(NamedTuple):
    """Static hyperparameters.

    Attributes:
        attr_dim: Material attributes per particle (stiffness or friction, normalised).
        width: Token width ``C``.
        branch_width: Output width of each tokenizer branch.
        heads: Attention heads; ``width / heads`` must be divisible by 4 (2D RoPE).
        kernel_size: ``K``, the kernel lattice is ``K x K``.
        r_pp, r_pb, r_topo: Support radius of the particle, boundary and topology kernels.
        enc_layers: ``L_e``; the encoder ends with ``N / 2^L_e`` super tokens.
        dec_layers: ``L_d``.
        mlp_ratio: Hidden width of each block MLP is ``mlp_ratio * C``.
        use_global: ``False`` gives the local-only ablation.
    """

    attr_dim: int = 1
    width: int = 32
    branch_width: int = 16
    heads: int = 4
    kernel_size: int = 4
    r_pp: float = 0.08
    r_pb: float = 0.06
    r_topo: float = 0.08
    enc_layers: int = 3
    dec_layers: int = 2
    mlp_ratio: int = 2
    use_global: bool = True

    @property
    def head_dim(self) -> int:
        return self.width // self.heads


class Scene(NamedTuple):
    """Everything about a scene that is not the state: fixed for a whole sequence.

    Attributes:
        attr: Material attributes per particle, ``(N, attr_dim)``.
        adjacency: Mesh topology, ``(N, N)`` 0/1; all zeros when there is none.
        rest: Rest positions for the topology kernel, ``(N, 2)``; zeros when there is none.
        wall_pts, wall_nrm: Boundary samples and their inward normals, ``(B, 2)`` each.
    """

    attr: Array
    adjacency: Array
    rest: Array
    wall_pts: Array
    wall_nrm: Array


class Norm(NamedTuple):
    """Fixed scales from the training data (not trained).

    Attributes:
        v: Typical speed, divides velocities before they enter the network.
        dx, dv: Typical size of the residuals ``dX`` and ``dV``; the head predicts
            them in these units.
        dt, gravity: The predictor's time step and gravity.
    """

    v: float
    dx: float
    dv: float
    dt: float
    gravity: float


# ----------------------------------------------------------------------------- parameters


def _dense(key: Array, i: int, o: int):
    """``W ~ N(0, 1/fan_in)``, ``b = 0``."""
    return (jax.random.normal(key, (i, o)) / jnp.sqrt(i), jnp.zeros((o,)))


def _mlp(key: Array, sizes: list[int]):
    return [_dense(k, a, b) for k, a, b in zip(jax.random.split(key, len(sizes) - 1), sizes[:-1], sizes[1:])]


def _ln(c: int):
    return {"gamma": jnp.ones((c,)), "beta": jnp.zeros((c,))}


def _attn(key: Array, c: int):
    ks = jax.random.split(key, 4)
    return {"q": _dense(ks[0], c, c), "k": _dense(ks[1], c, c), "v": _dense(ks[2], c, c), "o": _dense(ks[3], c, c)}


# Input channels of each tokenizer branch (see tokenize()).
PP_IN, PB_IN, TOPO_IN = 10, 5, 10

# Springs stretch by a few percent: the stretch feature is divided by this so it is of order 1.
STRAIN_SCALE = 0.05


def init_params(key: Array, cfg: WPConfig) -> dict:
    """Tokenizer kernels and MLP, encoder blocks, decoder blocks and head."""
    C, Cb, K = cfg.width, cfg.branch_width, cfg.kernel_size
    keys = jax.random.split(key, 8)

    def kernel(k, cin):
        return jax.random.normal(k, (K, K, cin, Cb)) / jnp.sqrt(cin * 4.0)

    tok_in = 3 * Cb + 2 + cfg.attr_dim
    enc = []
    for k in jax.random.split(keys[4], cfg.enc_layers if cfg.use_global else 0):
        k1, k2 = jax.random.split(k)
        enc.append({"ln1": _ln(C), "attn": _attn(k1, C), "ln2": _ln(C), "mlp": _mlp(k2, [C, cfg.mlp_ratio * C, C])})
    dec = []
    for k in jax.random.split(keys[5], cfg.dec_layers):
        k1, k2, k3 = jax.random.split(k, 3)
        layer = {"ln_m": _ln(C), "mlp": _mlp(k3, [C, cfg.mlp_ratio * C, C])}
        if cfg.use_global:                                  # the local-only model has no attention at all
            layer.update({"ln_q": _ln(C), "ln_kv": _ln(C), "cross": _attn(k1, C), "ln_s": _ln(C), "self": _attn(k2, C)})
        dec.append(layer)
    head = _mlp(keys[6], [C, C, C, 4])
    head[-1] = (head[-1][0] * 0.1, head[-1][1])           # start close to "no correction"
    return {
        "kernels": {"pp": kernel(keys[0], PP_IN), "pb": kernel(keys[1], PB_IN), "topo": kernel(keys[2], TOPO_IN)},
        "token_mlp": _mlp(keys[3], [tok_in, C, C]),
        "encoder": enc,
        "decoder": dec,
        "ln_head": _ln(C),
        "head": head,
    }


def count_params(params: dict) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


# ----------------------------------------------------------------------------- building blocks


def layer_norm(p: dict, x: Array, eps: float = 1e-5) -> Array:
    """Normalise every row to mean 0, variance 1 over its channels, then scale and shift."""
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * p["gamma"] + p["beta"]


def linear(p, x: Array) -> Array:
    W, b = p
    return x @ W + b


def mlp(layers, x: Array) -> Array:
    """GELU between layers, none after the last."""
    for W, b in layers[:-1]:
        x = jax.nn.gelu(x @ W + b)
    W, b = layers[-1]
    return x @ W + b


def kernel_weights(disp: Array, radius: float, K: int) -> tuple[Array, Array]:
    """Where a displacement lands on the ``K x K`` kernel lattice.

    The lattice covers the square ``[-R, R]^2``. A displacement ``d`` is mapped to
    lattice coordinates ``s = (d / R + 1) / 2 * (K - 1)`` and spread over the
    four surrounding lattice nodes with bilinear ("hat") weights, so the kernel
    is a continuous, piecewise-linear function of ``d``.

    Args:
        disp: Displacements ``(..., 2)``.

    Returns:
        ``(phi, window)``: lattice weights ``(..., K, K)`` (sum to 1) and a smooth
        compact-support window ``max(0, 1 - |d|^2 / R^2)`` that fades
        neighbours out at the radius ``(...)``.
    """
    u = disp / radius
    s = (jnp.clip(u, -1, 1) + 1) / 2 * (K - 1)
    hat = jax.nn.relu(1 - jnp.abs(s[..., None] - jnp.arange(K)))          # (..., 2, K)
    phi = hat[..., 0, :, None] * hat[..., 1, None, :]                      # (..., K, K)
    window = jax.nn.relu(1 - jnp.sum(u ** 2, -1))
    return phi, window


def continuous_conv(W: Array, disp: Array, mask: Array, feats: Array, radius: float, windowed: bool = True) -> Array:
    """One tokenizer branch: ``out_i = sum_j window_ij * sum_ab phi_ijab * feats_ij @ W_ab``.

    Args:
        W: Kernel lattice ``(K, K, C_in, C_out)``.
        disp: Displacement of every neighbour candidate from the particle, ``(N, M, 2)``.
        mask: Which candidates count (``(N, M)``, 0/1).
        feats: Features of every (particle, candidate) pair, ``(N, M, C_in)``.
        windowed: Fade neighbours out towards the radius. The topology branch
            turns this off: its neighbours are given by the mesh, not by distance.
    """
    phi, window = kernel_weights(disp, radius, W.shape[0])
    weight = window * mask if windowed else mask
    G = jnp.einsum("nmab,nmc->nabc", phi * weight[..., None, None], feats)   # features gathered on the lattice
    K, _, cin, cout = W.shape
    return G.reshape(G.shape[0], K * K * cin) @ W.reshape(K * K * cin, cout)


def rope_2d(t: Array, pos: Array) -> Array:
    """2D rotary position embedding on ``t`` of shape ``(H, n, d)``.

    The first half of each head's channels is rotated by angles proportional to
    ``x``, the second half by angles proportional to ``y``, in pairs, at
    frequencies ``2 pi * 16^(k / (d/4 - 1))``. A dot product between a rotated
    query at ``p`` and a rotated key at ``p'`` then depends on ``p - p'`` only:
    attention sees *relative* positions.
    """
    H, n, d = t.shape
    d4 = d // 4
    freqs = 2 * jnp.pi * 16.0 ** (jnp.arange(d4) / max(d4 - 1, 1))
    out = []
    for axis in range(2):
        half = t[..., axis * d // 2:(axis + 1) * d // 2].reshape(H, n, d4, 2)
        ang = pos[:, axis, None] * freqs                                    # (n, d4)
        c, s = jnp.cos(ang)[None], jnp.sin(ang)[None]
        a, b = half[..., 0], half[..., 1]
        out.append(jnp.stack([a * c - b * s, a * s + b * c], -1).reshape(H, n, d // 2))
    return jnp.concatenate(out, -1)


def attention(p: dict, xq: Array, xkv: Array, pos_q: Array, pos_k: Array, heads: int) -> tuple[Array, Array]:
    """Multi-head attention with 2D RoPE. Returns ``(out (n_q, C), weights (H, n_q, n_k))``."""
    def split(x):
        n, c = x.shape
        return x.reshape(n, heads, c // heads).transpose(1, 0, 2)
    q = rope_2d(split(linear(p["q"], xq)), pos_q)
    k = rope_2d(split(linear(p["k"], xkv)), pos_k)
    v = split(linear(p["v"], xkv))
    w = jax.nn.softmax(q @ k.transpose(0, 2, 1) / jnp.sqrt(q.shape[-1]), axis=-1)
    o = (w @ v).transpose(1, 0, 2).reshape(xq.shape[0], -1)
    return linear(p["o"], o), w


def greedy_match(sim: Array) -> Array:
    """One-to-one greedy matching of rows to columns by similarity.

    Rows are visited in order of their best similarity (most confident first);
    each takes its most similar column that is still free.

    Returns:
        ``match (n,)``: the column each row is paired with.
    """
    n = sim.shape[0]
    order = jnp.argsort(-sim.max(1))

    def body(i, carry):
        match, free = carry
        a = order[i]
        b = jnp.argmax(jnp.where(free, sim[a], -jnp.inf))
        return match.at[a].set(b), free.at[b].set(False)

    match, _ = jax.lax.fori_loop(0, n, body, (jnp.zeros(n, jnp.int32), jnp.ones(n, bool)))
    return match


def merge_tokens(x: Array, pos: Array, mult: Array) -> dict:
    """Bipartite token merging, ``n -> ceil(n / 2)`` tokens.

    Even-index tokens form set A and odd-index tokens set B. Each A token is
    paired with a B token (greedy, by cosine similarity of the features) and the
    pair becomes one token: the multiplicity-weighted average of the two
    features and of the two positions; multiplicities add. With an odd count
    the last token passes through unmerged.

    Returns:
        dict with ``x``, ``pos``, ``mult`` (the merged tokens), ``sim`` (A x B
        cosine similarities), ``match`` (B index of each A token) and ``P``,
        the ``(n_new, n)`` averaging matrix with ``x_new = P @ x``.
    """
    n = x.shape[0]
    na = n // 2
    xa, xb = x[0:2 * na:2], x[1:2 * na:2]
    unit = lambda t: t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)
    sim = jax.lax.stop_gradient(unit(xa) @ unit(xb).T)
    match = greedy_match(sim)
    ia, ib = 2 * jnp.arange(na), 2 * match + 1
    tot = mult[ia] + mult[ib]
    P = jnp.zeros((na, n)).at[jnp.arange(na), ia].set(mult[ia] / tot).at[jnp.arange(na), ib].set(mult[ib] / tot)
    if n % 2:
        P = jnp.concatenate([P, jnp.zeros((1, n)).at[0, n - 1].set(1.0)])
    P = jax.lax.stop_gradient(P)
    return dict(x=P @ x, pos=P @ pos, mult=(P > 0).astype(x.dtype) @ mult, sim=sim, match=match, P=P)


# ----------------------------------------------------------------------------- the three stages


def predictor(x: Array, v: Array, norm: Norm) -> tuple[Array, Array]:
    """Eq. 1: integrate the known force (gravity) only."""
    v_t = v + norm.dt * jnp.array([0.0, -norm.gravity])
    return x + norm.dt / 2 * (v + v_t), v_t


def tokenize(params: dict, x: Array, v: Array, scene: Scene, norm: Norm, cfg: WPConfig) -> dict:
    """Sec. 3.1: one token per particle from three neighbourhood branches.

    A branch is linear in its pair features (the kernel only weights them by
    where the neighbour sits), so the features are chosen such that the forces
    of this world are linear combinations of them. With ``d = x_j - x_i``,
    ``n = d / |d|`` and ``dv = (v_j - v_i) / s_v``:

    * particle-particle: ``[dv, (dv . n) n, a_j, 1, n, n |d| / R_pp]``, kernel of ``d``;
    * particle-boundary: ``[n_b, v_i / s_v, 1]``, kernel of ``x_b - x_i``;
    * topology: ``[n, e, e n, dv, (dv . n) n, 1]`` with the stretch
      ``e = (|d| / |r_j - r_i| - 1) / 0.05``, kernel of the rest offset ``r_j - r_i``
      (a spring's force is ``k e L n`` plus damping along ``n``).

    Returns:
        dict with the three branch outputs ``(N, branch_width)`` and the token ``h (N, C)``.
    """
    N = x.shape[0]
    vs = v / norm.v
    one = jnp.ones((N, N, 1))
    eye = jnp.eye(N)
    d_pp = x[None, :, :] - x[:, None, :]                                    # d[i, j] = x_j - x_i
    dist = jnp.sqrt(jnp.sum(d_pp ** 2, -1) + eye)[..., None]                # 1 on the diagonal, unused
    n_pp = d_pp / dist
    dv = vs[None, :, :] - vs[:, None, :]
    mask_pp = (jnp.sum(d_pp ** 2, -1) < cfg.r_pp ** 2) * (1 - eye)
    dv_n = jnp.sum(dv * n_pp, -1, keepdims=True) * n_pp                     # relative velocity along n
    f_pp = jnp.concatenate([dv, dv_n, jnp.broadcast_to(scene.attr[None], (N, N, cfg.attr_dim)), one, n_pp,
                            n_pp * dist / cfg.r_pp], -1)
    pp = continuous_conv(params["kernels"]["pp"], d_pp, mask_pp, f_pp, cfg.r_pp)

    B = scene.wall_pts.shape[0]
    d_pb = scene.wall_pts[None, :, :] - x[:, None, :]
    mask_pb = (jnp.sum(d_pb ** 2, -1) < cfg.r_pb ** 2).astype(x.dtype)
    f_pb = jnp.concatenate([jnp.broadcast_to(scene.wall_nrm[None], (N, B, 2)),
                            jnp.broadcast_to(vs[:, None], (N, B, 2)), jnp.ones((N, B, 1))], -1)
    pb = continuous_conv(params["kernels"]["pb"], d_pb, mask_pb, f_pb, cfg.r_pb)

    d_rest = scene.rest[None, :, :] - scene.rest[:, None, :]
    linked = scene.adjacency[..., None] > 0
    L = jnp.sqrt(jnp.where(linked, jnp.sum(d_rest ** 2, -1, keepdims=True), 1.0))   # rest length (1 if no spring)
    e = jnp.where(linked, (dist / L - 1) / STRAIN_SCALE, 0.0)
    f_topo = jnp.concatenate([n_pp, e, e * n_pp, dv, dv_n, one], -1)
    topo = continuous_conv(params["kernels"]["topo"], d_rest, scene.adjacency, f_topo, cfg.r_topo, windowed=False)

    h = mlp(params["token_mlp"], jnp.concatenate([pp, pb, topo, vs, scene.attr], -1))
    return dict(pp=pp, pb=pb, topo=topo, h=h)


def encode(params: dict, h: Array, pos: Array, cfg: WPConfig) -> dict:
    """Sec. 3.2: attention, merging and an MLP, ``enc_layers`` times.

    Returns:
        dict with the super tokens ``z``, their anchor positions ``z_pos``, their
        multiplicities ``z_mult``, ``members (n_super, N)`` (which particles each
        super token contains), and ``layers``: one record per layer.
    """
    x, mult, members = h, jnp.ones(h.shape[0]), jnp.eye(h.shape[0])
    layers = []
    for p in params["encoder"]:
        a_out, a_w = attention(p["attn"], layer_norm(p["ln1"], x), layer_norm(p["ln1"], x), pos, pos, cfg.heads)
        x = x + a_out
        m = merge_tokens(x, pos, mult)
        members = (m["P"] > 0).astype(x.dtype) @ members
        x, pos, mult = m["x"], m["pos"], m["mult"]
        x = x + mlp(p["mlp"], layer_norm(p["ln2"], x))
        layers.append(dict(attn=a_w, sim=m["sim"], match=m["match"], P=m["P"], x=x, pos=pos, mult=mult,
                           members=members))
    return dict(z=x, z_pos=pos, z_mult=mult, members=members, layers=layers)


def decode(params: dict, h: Array, pos: Array, enc: dict | None, cfg: WPConfig) -> dict:
    """Sec. 3.3: particles query the super tokens, then attend to each other; then the head.

    Returns:
        dict with ``out (N, 4)`` (normalised ``dx, dv``) and ``layers``: the
        cross-attention ``(H, N, n_super)`` and self-attention ``(H, N, N)`` weights.
    """
    y = h
    layers = []
    for p in params["decoder"]:
        rec = {}
        if cfg.use_global:
            c_out, rec["cross"] = attention(p["cross"], layer_norm(p["ln_q"], y), layer_norm(p["ln_kv"], enc["z"]),
                                            pos, enc["z_pos"], cfg.heads)
            y = y + c_out
            s_out, rec["self"] = attention(p["self"], layer_norm(p["ln_s"], y), layer_norm(p["ln_s"], y), pos, pos,
                                           cfg.heads)
            y = y + s_out
        y = y + mlp(p["mlp"], layer_norm(p["ln_m"], y))
        layers.append(rec)
    return dict(y=y, out=mlp(params["head"], layer_norm(params["ln_head"], y)), layers=layers)


# ----------------------------------------------------------------------------- one step


class Trace(NamedTuple):
    """Every intermediate tensor of one step.

    Attributes:
        x_pred, v_pred: The predictor's ``X~, V~``, ``(N, 2)``.
        tokens: Output of :func:`tokenize`.
        enc: Output of :func:`encode` (``None`` for the local-only model).
        dec: Output of :func:`decode`.
        dx, dv: The corrections in physical units, ``(N, 2)``.
        x_next, v_next: The new state.
    """

    x_pred: Array
    v_pred: Array
    tokens: dict
    enc: dict | None
    dec: dict
    dx: Array
    dv: Array
    x_next: Array
    v_next: Array


def step_with_trace(params: dict, x: Array, v: Array, scene: Scene, norm: Norm, cfg: WPConfig) -> Trace:
    """One frame forward, keeping every tensor."""
    x_p, v_p = predictor(x, v, norm)
    tok = tokenize(params, x_p, v_p, scene, norm, cfg)
    enc = encode(params, tok["h"], x_p, cfg) if cfg.use_global else None
    dec = decode(params, tok["h"], x_p, enc, cfg)
    dx, dv = dec["out"][:, :2] * norm.dx, dec["out"][:, 2:] * norm.dv
    return Trace(x_p, v_p, tok, enc, dec, dx, dv, x_p + dx, v_p + dv)


def step(params: dict, x: Array, v: Array, scene: Scene, norm: Norm, cfg: WPConfig) -> tuple[Array, Array]:
    """One frame forward: ``(X, V) -> (X', V')``."""
    tr = step_with_trace(params, x, v, scene, norm, cfg)
    return tr.x_next, tr.v_next


def rollout(params: dict, x0: Array, v0: Array, scene: Scene, norm: Norm, cfg: WPConfig, n_steps: int):
    """Feed the model its own predictions for ``n_steps`` frames.

    Returns:
        ``(X, V)``, each ``(n_steps + 1, N, 2)``, starting with ``x0, v0``.
    """
    def body(state, _):
        nxt = step(params, *state, scene, norm, cfg)
        return nxt, nxt

    _, (X, V) = jax.lax.scan(body, (x0, v0), None, length=n_steps)
    return jnp.concatenate([x0[None], X]), jnp.concatenate([v0[None], V])
