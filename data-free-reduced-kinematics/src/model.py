"""Neural subspace maps for data-free reduced-order kinematics.

This module implements the learnable map

    f_theta : R^d -> R^n

from a low-dimensional latent space (dimension ``d``) to the full configuration
space of a physical system (dimension ``n``), as described in Sec. 3.1 and
Sec. 4 of Sharp et al., "Data-Free Learning of Reduced-Order Kinematics".

Architecture
------------
``f_theta`` is a plain multi-layer perceptron (MLP) with ELU activations on
every hidden layer and a linear output layer. ELU is smooth (C^1), which keeps
the subspace map differentiable everywhere -- important because downstream
uses (subspace simulation, Sec. 3.4) differentiate *through* ``f_theta``.

Seeded subspace exploration (Sec. 4.1)
--------------------------------------
Starting from random weights, the MLP's image lies nowhere near the
low-energy manifold, and finding it is a hard optimisation problem. During
training only, the map is therefore blended with a user-supplied seed
configuration ``q_seed``:

    f_theta(z) := p * MLP_theta(z) + (1 - p) * q_seed

where ``p`` ramps linearly from 0 to 1 over training. Early on every latent
maps (almost) to ``q_seed``, a valid low-energy state; as ``p -> 1`` the MLP
gradually takes over and explores outward. At ``p = 1`` the seed has vanished
and the trained map is just ``MLP_theta``.

Design notes
------------
* Pure JAX, no NN framework: parameters are a pytree (list of ``(W, b)``
  tuples), so they work directly with ``jax.grad``, ``jax.jit``, ``jax.vmap``
  and ``optax``.
* All ``apply`` functions act on a *single* latent vector ``z`` of shape
  ``(d,)``. Use ``jax.vmap`` to evaluate batches; this keeps per-sample
  Jacobians (``jax.jacfwd(f)(z)``) straightforward.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import Array

# A dense layer is a (weight, bias) pair; W has shape (fan_in, fan_out).
Layer = tuple[Array, Array]
# The full MLP parameter pytree theta.
MLPParams = list[Layer]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class SubspaceConfig(NamedTuple):
    """Static hyperparameters describing a neural subspace map.

    Attributes:
        latent_dim: ``d``, dimension of the latent / reduced space.
        config_dim: ``n``, dimension of the full configuration space
            (e.g. ``3 * |V|`` for a 3D mesh).
        hidden_dims: Widths of the hidden layers of the MLP.
    """

    latent_dim: int
    config_dim: int
    hidden_dims: Sequence[int] = (128, 128, 128)

    @property
    def layer_sizes(self) -> list[int]:
        """Widths of every layer, input to output: ``[d, *hidden, n]``."""
        return [self.latent_dim, *self.hidden_dims, self.config_dim]


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------


def init_mlp_params(key: Array, layer_sizes: Sequence[int]) -> MLPParams:
    """Randomly initialise MLP weights.

    Weights use variance-scaling (LeCun/He-style) normal initialisation,
    ``W ~ N(0, 1 / fan_in)``, which keeps activations O(1) through ELU layers.
    Biases start at zero.

    Args:
        key: JAX PRNG key.
        layer_sizes: Widths of every layer including input and output,
            e.g. ``SubspaceConfig.layer_sizes``.

    Returns:
        List of ``(W, b)`` tuples, one per dense layer.
    """
    params: MLPParams = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        W = jax.random.normal(k, (fan_in, fan_out)) / jnp.sqrt(fan_in)
        b = jnp.zeros((fan_out,))
        params.append((W, b))
    return params


def mlp_apply(params: MLPParams, z: Array) -> Array:
    """Evaluate ``MLP_theta(z)``.

    Every hidden layer is ``x -> elu(x @ W + b)``; the final layer is linear so
    the output can take any value in R^n (configurations are unbounded).

    Args:
        params: MLP parameters from :func:`init_mlp_params`.
        z: Latent vector of shape ``(d,)``.

    Returns:
        Configuration vector of shape ``(n,)``.
    """
    x = z
    for W, b in params[:-1]:
        x = jax.nn.elu(x @ W + b)
    W_out, b_out = params[-1]
    return x @ W_out + b_out


# -----------------------------------------------------------------------------
# Seeded subspace exploration
# -----------------------------------------------------------------------------


def seed_schedule(step: Array | int, ramp_steps: int) -> Array:
    """Blend weight ``p`` for seeded exploration (Sec. 4.1).

    ``p`` increases linearly from 0 at ``step = 0`` to 1 at
    ``step = ramp_steps`` and stays at 1 afterwards, so any training steps
    beyond the ramp fit the pure MLP.

    Args:
        step: Current training iteration (Python int or traced scalar, so this
            can be called inside ``jax.jit``).
        ramp_steps: Number of iterations over which ``p`` goes 0 -> 1. Must be
            positive.

    Returns:
        Scalar ``p`` in ``[0, 1]``.
    """
    return jnp.clip(jnp.asarray(step, dtype=jnp.float32) / ramp_steps, 0.0, 1.0)


def seeded_subspace_apply(
    params: MLPParams, z: Array, p: Array | float, q_seed: Array
) -> Array:
    """Training-time subspace map ``f_theta(z) = p MLP(z) + (1 - p) q_seed``.

    Args:
        params: MLP parameters ``theta``.
        z: Latent vector of shape ``(d,)``.
        p: Blend weight in ``[0, 1]``, typically from :func:`seed_schedule`.
            Treated as a constant (no gradient flows into the schedule).
        q_seed: Seed configuration of shape ``(n,)``; any valid low-energy
            state of the system (e.g. the rest pose).

    Returns:
        Configuration vector of shape ``(n,)``.
    """
    p = jax.lax.stop_gradient(p)
    return p * mlp_apply(params, z) + (1.0 - p) * q_seed


def subspace_apply(params: MLPParams, z: Array) -> Array:
    """Trained subspace map ``f_theta(z) = MLP_theta(z)``.

    This is the ``p = 1`` limit of :func:`seeded_subspace_apply`: after
    training, the seed configuration is entirely absent.
    """
    return mlp_apply(params, z)


def make_subspace_map(
    params: MLPParams,
    q_seed: Array | None = None,
    p: Array | float = 1.0,
) -> Callable[[Array], Array]:
    """Close over parameters to obtain a plain function ``z -> q``.

    Convenient for downstream use (simulation, interpolation, sampling) where
    only the latent input varies.

    Args:
        params: MLP parameters ``theta``.
        q_seed: Seed configuration. Required only when ``p`` may be < 1.
        p: Blend weight. Defaults to 1 (the trained, seed-free map).

    Returns:
        Function mapping a latent ``(d,)`` vector to a configuration ``(n,)``.
    """
    if q_seed is None:
        return lambda z: subspace_apply(params, z)
    return lambda z: seeded_subspace_apply(params, z, p, q_seed)
