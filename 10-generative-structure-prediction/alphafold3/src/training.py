"""Training and sampling for the toy AF3 (Supplementary §3.7 and Alg. 18-20 of AF3).

Training (one step)
-------------------
::

    pick B structures x0 (sequence types, one of its folds)
    random rotation + translation of x0          AF3: CentreRandomAugmentation
    s_in, s, z = trunk(types)                    ONE trunk pass per structure
    for each of K noise levels                   AF3: K = 48 "diffusion batch"
        sigma = sd * exp(-1.2 + 1.5 n),  n ~ N(0, 1)
        x~    = x0 + sigma * eps
        x^    = denoise(s_in, s, z, x~, sigma)
        loss += (sigma^2 + sd^2) / (sigma sd)^2 * MSE(x^, align(x0 -> x^))

The weight makes every noise level contribute equally (EDM). The target is
rigidly aligned onto the prediction first, as in AF3, because at high noise the
global orientation of the answer is arbitrary.

Sampling (AF3 Alg. 18)
----------------------
::

    sigmas: sd * (s_max^(1/7) + t (s_min^(1/7) - s_max^(1/7)))^7, t in [0, 1]
    x = sigma_0 * eps
    for each step: random rotation of x, add a little extra noise (gamma),
                   denoise, take an Euler step scaled by eta = 1.5

The AF2-flavoured baseline is trained with an alignment-invariant MSE of its
one deterministic prediction against a random fold of the same sequence.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from .model import AF3Config, apply_regression, denoise, trunk


# --------------------------------------------------------------------------- geometry

def rot2(theta: Array) -> Array:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def random_rigid(key: Array, x: Array, shift: float = 0.0) -> Array:
    """Centre, rotate by a random angle and shift: AF3's CentreRandomAugmentation."""
    k1, k2 = jax.random.split(key)
    x = x - x.mean(0, keepdims=True)
    return x @ rot2(jax.random.uniform(k1, (), maxval=2 * jnp.pi)).T + shift * jax.random.normal(k2, (2,))


def align(mobile: Array, target: Array) -> Array:
    """Best proper rotation + translation of ``mobile`` onto ``target`` (2D Kabsch).

    Reflections are not allowed, so a mirror image never aligns perfectly.
    """
    mc = mobile - mobile.mean(0)
    tc = target - target.mean(0)
    cross = jnp.sum(mc[:, 0] * tc[:, 1] - mc[:, 1] * tc[:, 0])
    dot = jnp.sum(mc * tc)
    theta = jnp.arctan2(cross, dot)
    return mc @ rot2(theta).T + target.mean(0)


def rmsd(a: Array, b: Array) -> Array:
    """RMSD after the best proper alignment of ``a`` onto ``b``."""
    return jnp.sqrt(jnp.mean(jnp.sum((align(a, b) - b) ** 2, -1)))


# --------------------------------------------------------------------------- losses

def diffusion_loss(params: dict, types: Array, x0: Array, key: Array, cfg: AF3Config,
                   n_sigma: int) -> Array:
    """EDM-weighted, alignment-aware denoising loss for one structure."""
    s_in, s, z = trunk(params, types, cfg)
    k1, k2 = jax.random.split(key)
    sd = cfg.sigma_data
    sigmas = sd * jnp.exp(-1.2 + 1.5 * jax.random.normal(k1, (n_sigma,)))
    eps = jax.random.normal(k2, (n_sigma,) + x0.shape)

    def one(sigma, e):
        x_noisy = x0 + sigma * e
        x_hat = denoise(params, s_in, s, z, x_noisy, sigma, cfg)
        tgt = jax.lax.stop_gradient(align(x0, x_hat))
        w = (sigma**2 + sd**2) / (sigma * sd) ** 2
        return w * jnp.mean(jnp.sum((x_hat - tgt) ** 2, -1))

    return jnp.mean(jax.vmap(one)(sigmas, eps))


def regression_loss(params: dict, types: Array, x0: Array, cfg: AF3Config) -> Array:
    """AF2-flavoured baseline: MSE of one prediction after aligning the target onto it."""
    x_hat = apply_regression(params, types, cfg)
    tgt = jax.lax.stop_gradient(align(x0, x_hat))
    return jnp.mean(jnp.sum((x_hat - tgt) ** 2, -1))


# --------------------------------------------------------------------------- training

def train(params: dict, data: dict, cfg: AF3Config, mode: str = "diffusion", steps: int = 3000,
          batch: int = 32, n_sigma: int = 8, lr: float = 2e-3, snapshot_every: int = 250,
          seed: int = 0, log: Callable[[str], None] | None = print):
    """Adam on either the diffusion loss or the regression loss.

    Returns ``(params, history)`` where ``history`` has ``loss`` per step and
    ``snapshots``: a list of ``(step, params)``.
    """
    types_all, x_all = data["types"], data["x"]
    n_seq, n_samp = x_all.shape[:2]
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, 100, steps, lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    state = opt.init(params)

    def batch_loss(p, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        si = jax.random.randint(k1, (batch,), 0, n_seq)
        ki = jax.random.randint(k2, (batch,), 0, n_samp)
        x0 = jax.vmap(random_rigid)(jax.random.split(k3, batch), x_all[si, ki])
        ty = types_all[si]
        if mode == "diffusion":
            ls = jax.vmap(lambda t, x, k: diffusion_loss(p, t, x, k, cfg, n_sigma))(
                ty, x0, jax.random.split(k4, batch))
        else:
            ls = jax.vmap(lambda t, x: regression_loss(p, t, x, cfg))(ty, x0)
        return jnp.mean(ls)

    @jax.jit
    def step(p, s, key):
        loss, g = jax.value_and_grad(batch_loss)(p, key)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    key = jax.random.PRNGKey(seed)
    hist = {"loss": [], "snapshots": [(0, params)]}
    t0 = time.time()
    for i in range(1, steps + 1):
        key, k = jax.random.split(key)
        params, state, loss = step(params, state, k)
        hist["loss"].append(float(loss))
        if i % snapshot_every == 0:
            hist["snapshots"].append((i, params))
            if log:
                log(f"{mode:9s} step {i:5d}  loss {np.mean(hist['loss'][-snapshot_every:]):.4f}"
                    f"  ({time.time() - t0:.0f}s)")
    return params, hist


# --------------------------------------------------------------------------- sampling

def noise_schedule(n_steps: int, sd: float, s_max: float = 10.0, s_min: float = 4e-4,
                   rho: float = 7.0) -> Array:
    """AF3's schedule, with ``s_max``/``s_min`` in units of ``sigma_data`` (AF3: 160/16 and 4e-4)."""
    t = jnp.linspace(0.0, 1.0, n_steps + 1)
    return sd * (s_max ** (1 / rho) + t * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho


@partial(jax.jit, static_argnames=("cfg", "n_steps", "gamma0", "gamma_min", "noise_scale",
                                   "step_scale", "return_path"))
def sample(params: dict, types: Array, key: Array, cfg: AF3Config, n_steps: int = 40,
           gamma0: float = 0.8, gamma_min: float = 1.0, noise_scale: float = 1.003,
           step_scale: float = 1.5, return_path: bool = False):
    """One structure from noise (AF3 Alg. 18, SampleDiffusion), trunk computed once."""
    s_in, s, z = trunk(params, types, cfg)
    sig = noise_schedule(n_steps, cfg.sigma_data)
    k0, key = jax.random.split(key)
    x = sig[0] * jax.random.normal(k0, (types.shape[0], 2))

    def body(x, inp):
        c_prev, c, k = inp
        ka, kb = jax.random.split(k)
        x = random_rigid(ka, x)
        gamma = jnp.where(c > gamma_min, gamma0, 0.0)
        t_hat = c_prev * (gamma + 1.0)
        x_noisy = x + noise_scale * jnp.sqrt(jnp.maximum(t_hat**2 - c_prev**2, 0.0)) * jax.random.normal(kb, x.shape)
        x_den = denoise(params, s_in, s, z, x_noisy, t_hat, cfg)
        delta = (x_noisy - x_den) / t_hat
        x = x_noisy + step_scale * (c - t_hat) * delta
        return x, (x, x_den)

    x, (path, dens) = jax.lax.scan(body, x, (sig[:-1], sig[1:], jax.random.split(key, n_steps)))
    x = x - x.mean(0)
    if return_path:
        return x, path, dens, sig
    return x


@partial(jax.jit, static_argnames=("n", "cfg", "n_steps"))
def sample_many(params: dict, types: Array, key: Array, n: int, cfg: AF3Config, n_steps: int = 40) -> Array:
    """``n`` independent samples (AF3 runs 5 per seed) for one sequence."""
    return jax.vmap(lambda k: sample(params, types, k, cfg, n_steps))(jax.random.split(key, n))
