"""Training WorldParticle on simulated sequences (Wang et al., 2026, Sec. 3.4, App. B).

Recipe
------
* **Data:** for one category, ``S`` sequences of ``T`` frames of ``(X, V)``,
  plus the scene of each sequence (attributes, topology, walls).
* **Loss:** the autoregressive rollout loss of the paper. Start from a true
  frame ``t``, let the model run ``W - 1`` steps on its *own* predictions, and
  average the position and velocity errors over the steps:

      L_roll = mean_{k=1..W-1} [ |X^_{t+k} - X_{t+k}|^2 / s_dx^2 + |V^_{t+k} - V_{t+k}|^2 / s_dv^2 ]

  ``s_dx``, ``s_dv`` are the typical sizes of the one-step residuals, so both
  terms are of order 1 and the loss counts errors in "residuals". With
  ``W = 2`` this is plain one-step training; larger windows teach the model
  to cope with its own errors (paper Table 2: the horizon is the biggest lever).
* **Optimiser:** AdamW with linear warm-up and cosine decay, as in the paper
  (App. B), on small batches of random windows.

The paper's optional physics loss (an SPH divergence penalty for fluids) is
left out: neither category here is a fluid.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from . import physics
from .model import Norm, Scene, WPConfig, predictor, step


# ----------------------------------------------------------------------------- data preparation


def material_scale(category: str, p: physics.SimParams) -> tuple[float, float]:
    """``(centre, half-width)`` that maps the training attribute range to ``[-1, 1]``."""
    lo, hi = p.k_range if category == "jelly" else p.mu_range
    return (lo + hi) / 2, (hi - lo) / 2


def make_scene(category: str, material: float, p: physics.SimParams) -> Scene:
    """The fixed inputs of one sequence. The sand has no topology: zero-filled."""
    rest, _, adj = physics.jelly_springs(p)
    wall_pts, wall_nrm = physics.boundary_samples(p)
    c, w = material_scale(category, p)
    attr = np.full((p.n, 1), (material - c) / w, np.float32)
    if category == "sand":
        rest, adj = np.zeros_like(rest), np.zeros_like(adj)
    return Scene(*(jnp.asarray(a) for a in (attr, adj, rest, wall_pts, wall_nrm)))


def stack_scenes(category: str, materials: np.ndarray, p: physics.SimParams) -> Scene:
    """One :class:`Scene` per sequence, stacked along a leading axis."""
    scenes = [make_scene(category, float(m), p) for m in materials]
    return Scene(*(jnp.stack(a) for a in zip(*scenes)))


def residuals(X: np.ndarray, V: np.ndarray, norm: Norm) -> tuple[np.ndarray, np.ndarray]:
    """What the corrector must add after the predictor: ``X_{t+1} - X~``, ``V_{t+1} - V~``."""
    xp, vp = jax.vmap(jax.vmap(lambda x, v: predictor(x, v, norm)))(X[:, :-1], V[:, :-1])
    return np.asarray(X[:, 1:] - xp), np.asarray(V[:, 1:] - vp)


def fit_norm(data: dict, p: physics.SimParams) -> Norm:
    """Typical speed and residual sizes (RMS over the training set)."""
    base = Norm(v=1.0, dx=1.0, dv=1.0, dt=p.frame_dt, gravity=p.gravity)
    dX, dV = residuals(data["X"], data["V"], base)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    return base._replace(v=rms(data["V"]), dx=rms(dX), dv=rms(dV))


# ----------------------------------------------------------------------------- loss and training


def window_loss(params: dict, X: Array, V: Array, scene: Scene, norm: Norm, cfg: WPConfig) -> Array:
    """Rollout loss on one window ``X, V: (W, N, 2)`` that starts at a true frame."""
    def body(state, target):
        x, v = step(params, *state, scene, norm, cfg)
        err = jnp.mean(jnp.sum((x - target[0]) ** 2, -1)) / norm.dx ** 2 \
            + jnp.mean(jnp.sum((v - target[1]) ** 2, -1)) / norm.dv ** 2
        return (x, v), err

    _, errs = jax.lax.scan(body, (X[0], V[0]), (X[1:], V[1:]))
    return jnp.mean(errs)


@partial(jax.jit, static_argnums=(5,))
def batch_loss(params: dict, X: Array, V: Array, scenes: Scene, norm: Norm, cfg: WPConfig) -> Array:
    """Mean of :func:`window_loss` over a batch ``(B, W, N, 2)``."""
    return jnp.mean(jax.vmap(window_loss, in_axes=(None, 0, 0, 0, None, None))(params, X, V, scenes, norm, cfg))


def sample_windows(rng: np.random.Generator, data: dict, scenes: Scene, batch: int, window: int):
    """Random windows of ``window`` consecutive frames from random sequences."""
    S, T = data["X"].shape[:2]
    s = rng.integers(0, S, batch)
    t = rng.integers(0, T - window + 1, batch)
    idx = t[:, None] + np.arange(window)
    X = data["X"][s[:, None], idx]
    V = data["V"][s[:, None], idx]
    return jnp.asarray(X), jnp.asarray(V), jax.tree_util.tree_map(lambda a: a[s], scenes)


def train(params: dict, data: dict, scenes: Scene, norm: Norm, cfg: WPConfig, n_steps: int = 3000,
          window: int = 4, batch_size: int = 8, lr: float = 2e-3, warmup: int = 200, weight_decay: float = 1e-4,
          seed: int = 0, snapshot_every: int = 250, log_every: int = 500):
    """AdamW, warm-up then cosine decay, on random rollout windows.

    Args:
        window: ``W``, frames per window; the model unrolls ``W - 1`` steps.

    Returns:
        ``(params, snapshots, snapshot_steps, losses)``.
    """
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, n_steps, lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=weight_decay))
    opt_state = opt.init(params)

    @jax.jit
    def update(params, opt_state, X, V, sc):
        loss, grads = jax.value_and_grad(batch_loss)(params, X, V, sc, norm, cfg)
        upd, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    rng = np.random.default_rng(seed)
    snapshots, snap_steps, losses = [params], [0], []
    for i in range(1, n_steps + 1):
        X, V, sc = sample_windows(rng, data, scenes, batch_size, window)
        params, opt_state, loss = update(params, opt_state, X, V, sc)
        losses.append(float(loss))
        if i % snapshot_every == 0:
            snapshots.append(params)
            snap_steps.append(i)
        if log_every and i % log_every == 0:
            print(f"step {i:5d}   rollout loss (train batch) = {np.mean(losses[-log_every:]):.4f}")
    return params, snapshots, snap_steps, np.array(losses)


# ----------------------------------------------------------------------------- evaluation


@partial(jax.jit, static_argnums=(4, 5))
def rollout_batch(params: dict, X0: Array, V0: Array, scenes: Scene, cfg: WPConfig, n_steps: int, norm: Norm):
    """Full rollouts from frame 0 for a batch of sequences, ``(S, n_steps + 1, N, 2)`` each."""
    from .model import rollout
    return jax.vmap(lambda x, v, sc: rollout(params, x, v, sc, norm, cfg, n_steps))(X0, V0, scenes)


def rollout_errors(params: dict, data: dict, scenes: Scene, norm: Norm, cfg: WPConfig) -> dict:
    """Roll every test sequence out from its first frame for the whole sequence.

    Returns:
        dict with the predicted ``X, V`` and the position / velocity MSE per frame
        (averaged over particles and sequences), the paper's Table 1 metric.
    """
    T = data["X"].shape[1]
    Xp, Vp = rollout_batch(params, jnp.asarray(data["X"][:, 0]), jnp.asarray(data["V"][:, 0]), scenes, cfg, T - 1,
                           norm)
    Xp, Vp = np.asarray(Xp), np.asarray(Vp)
    pos = np.mean(np.sum((Xp - data["X"]) ** 2, -1), axis=(0, 2))
    vel = np.mean(np.sum((Vp - data["V"]) ** 2, -1), axis=(0, 2))
    return dict(X=Xp, V=Vp, pos_mse=pos, vel_mse=vel)
