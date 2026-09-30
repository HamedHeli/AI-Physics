"""Generic training loop for neural subspaces (Sec. 3.2 + Sec. 4.1).

Every experiment follows the same pattern:

1. build a potential ``E_pot``, a mass matrix ``M`` and a seed ``q_seed``;
2. bind them into a loss with :func:`src.energy.make_loss_fn`;
3. call :func:`train`, which runs Adam while ramping the seed blend ``p``
   from 0 to 1;
4. save the parameters with :func:`save_params` for later viewing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from .model import MLPParams, SubspaceConfig, init_mlp_params, seed_schedule

LossFn = Callable[[MLPParams, Array, Array], tuple[Array, dict[str, Array]]]


def train(
    loss_fn: LossFn,
    config: SubspaceConfig,
    steps: int,
    ramp_steps: int,
    lr: float,
    seed: int = 0,
    log_every: int = 1000,
    clip_norm: float = 1.0,
) -> MLPParams:
    """Fit theta with Adam and a cosine-decayed learning rate.

    The seed blend ``p`` ramps 0 -> 1 over ``ramp_steps`` and then stays at 1,
    so the final ``steps - ramp_steps`` iterations fit the pure MLP. The
    schedule starts at step 1 because at p = 0 the output does not depend on
    theta and the gradient is zero.

    Args:
        loss_fn: ``(params, key, p) -> (loss, aux)`` from ``make_loss_fn``.
        config: Network shape.
        steps: Total optimisation steps.
        ramp_steps: Steps over which ``p`` goes 0 -> 1.
        lr: Peak learning rate (decays to 1% by the end).
        seed: PRNG seed for initialisation and latent sampling.
        log_every: Print progress every this many steps.
        clip_norm: Global gradient-norm clip. Stiff penalty energies can
            produce huge occasional gradients, and clipping keeps them from
            wrecking the network.

    Returns:
        Trained MLP parameters.
    """
    key_init, key = jax.random.split(jax.random.PRNGKey(seed))
    params = init_mlp_params(key_init, config.layer_sizes)

    schedule = optax.cosine_decay_schedule(lr, decay_steps=steps, alpha=1e-2)
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(schedule))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, key, i):
        p = seed_schedule(i + 1, ramp_steps)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, p)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux, p

    t0 = time.time()
    for i in range(steps):
        key, k = jax.random.split(key)
        params, opt_state, loss, aux, p = step(params, opt_state, k, i)
        if i % log_every == 0 or i == steps - 1:
            print(
                f"step {i:6d}  p={float(p):.3f}  loss={float(loss):.4e}  "
                f"E_pot={float(aux['energy']):.4e}  iso={float(aux['isometry']):.4e}  "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )
    return params


def save_params(params: MLPParams, path: Path) -> None:
    """Save MLP parameters as ``W0, b0, W1, b1, ...`` in an ``.npz`` file."""
    flat = {f"W{i}": np.asarray(W) for i, (W, _) in enumerate(params)}
    flat.update({f"b{i}": np.asarray(b) for i, (_, b) in enumerate(params)})
    np.savez(path, **flat)


def load_params(path: Path) -> MLPParams:
    """Inverse of :func:`save_params`."""
    data = np.load(path)
    n_layers = sum(1 for k in data.files if k.startswith("W"))
    return [(jnp.asarray(data[f"W{i}"]), jnp.asarray(data[f"b{i}"])) for i in range(n_layers)]
