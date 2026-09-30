"""Training objective for data-free neural subspaces (Sec. 3.2).

Given a differentiable potential energy ``E_pot : R^n -> R`` and a mass matrix
``M``, the subspace parameters ``theta`` are fit by stochastic gradient descent
on

    L(theta) = E_{z, z' ~ N(0, I_d)} [ E_pot(f_theta(z))
                 + lambda * ( log( |f(z) - f(z')|_M / (sigma |z - z'|) ) )^2 ]

The two terms pull in opposite directions:

* **Expected potential energy** drives sampled configurations toward low
  energy. On its own it collapses every ``z`` onto the energy minimiser.
* **Soft isometry penalty** asks ``f_theta`` to preserve distances up to the
  scale ``sigma``, measured in the mass-weighted norm ``|x|_M^2 = x^T M x``.
  This prevents collapse and forces the latent space to spread out over a
  diverse set of configurations. ``sigma`` sets how far the subspace reaches;
  ``lambda`` sets how strictly isometry is enforced.

No training data is involved: the only inputs are ``E_pot``, ``M`` and the
latent samples drawn here.

Mass matrix representation
--------------------------
Anywhere ``M`` is accepted it may be:

* ``None``         -> identity (plain Euclidean norm),
* shape ``(n,)``   -> diagonal / lumped mass (the common case; O(n)),
* shape ``(n, n)`` -> dense symmetric positive-definite matrix.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .model import MLPParams, seeded_subspace_apply

# A potential energy maps one configuration (n,) to a scalar.
EnergyFn = Callable[[Array], Array]

# Guards log(0) and the non-differentiable sqrt at 0 in the isometry term.
_EPS = 1e-12


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------


class ObjectiveConfig(NamedTuple):
    """Hyperparameters of the training objective.

    Attributes:
        sigma: Target scale of the isometry, ``|f(z)-f(z')|_M ~ sigma |z-z'|``.
            Small values concentrate the subspace near low-energy states.
        lam: ``lambda``, weight of the isometry penalty relative to energy.
        batch_size: Number of latent pairs ``(z, z')`` per Monte Carlo
            estimate of the expectation.
    """

    sigma: float = 1.0
    lam: float = 1.0
    batch_size: int = 32


# -----------------------------------------------------------------------------
# Mass-weighted norms
# -----------------------------------------------------------------------------


def mass_sq_norm(x: Array, M: Array | None = None) -> Array:
    """Squared mass-weighted norm ``|x|_M^2 = x^T M x`` of a vector ``x``.

    Args:
        x: Vector of shape ``(n,)``.
        M: Mass matrix: ``None`` (identity), diagonal ``(n,)`` or dense
            ``(n, n)``. See the module docstring.

    Returns:
        Non-negative scalar.
    """
    if M is None:
        return jnp.dot(x, x)
    if M.ndim == 1:
        return jnp.dot(x, M * x)
    return jnp.dot(x, M @ x)


def mass_norm(x: Array, M: Array | None = None) -> Array:
    """Mass-weighted norm ``|x|_M``."""
    return jnp.sqrt(mass_sq_norm(x, M))


# -----------------------------------------------------------------------------
# Objective terms
# -----------------------------------------------------------------------------


def isometry_penalty(
    q: Array,
    q_prime: Array,
    z: Array,
    z_prime: Array,
    sigma: float,
    M: Array | None = None,
) -> Array:
    """Soft isometry penalty for one latent pair.

        ( log( |q - q'|_M / (sigma |z - z'|) ) )^2

    The log-ratio penalises stretching and compression symmetrically: mapping
    two latents twice too far apart costs the same as half as far. It
    diverges as ``|q - q'| -> 0``, which is what stops collapse onto a single
    configuration.

    Implemented as ``0.5 * (log|dq|_M^2 - log(sigma^2 |dz|^2))`` so no square
    root is differentiated; ``_EPS`` keeps the logs finite.

    Args:
        q, q_prime: Configurations ``f(z)``, ``f(z')`` of shape ``(n,)``.
        z, z_prime: Latent vectors of shape ``(d,)``.
        sigma: Isometry scale.
        M: Mass matrix (see module docstring).

    Returns:
        Non-negative scalar penalty.
    """
    log_dq_sq = jnp.log(mass_sq_norm(q - q_prime, M) + _EPS)
    log_dz_sq = jnp.log(sigma**2 * jnp.dot(z - z_prime, z - z_prime) + _EPS)
    return (0.5 * (log_dq_sq - log_dz_sq)) ** 2


def sample_latents(key: Array, batch_size: int, latent_dim: int) -> tuple[Array, Array]:
    """Draw independent batches ``z, z' ~ N(0, I_d)``.

    Returns:
        Two arrays of shape ``(batch_size, latent_dim)``.
    """
    key_z, key_zp = jax.random.split(key)
    z = jax.random.normal(key_z, (batch_size, latent_dim))
    z_prime = jax.random.normal(key_zp, (batch_size, latent_dim))
    return z, z_prime


def subspace_loss(
    params: MLPParams,
    key: Array,
    p: Array | float,
    *,
    energy_fn: EnergyFn,
    q_seed: Array,
    latent_dim: int,
    config: ObjectiveConfig,
    M: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Monte Carlo estimate of the full training objective.

        L = mean[ E_pot(f(z)) ] + lambda * mean[ isometry_penalty(z, z') ]

    ``f`` is the seeded map ``p MLP(z) + (1 - p) q_seed`` (Sec. 4.1), so this
    is the loss to use throughout training; at ``p = 1`` it reduces to the
    objective on the plain MLP.

    Both ``f(z)`` and ``f(z')`` are needed for the isometry term, so the
    energy term is averaged over both batches. Since ``z`` and ``z'`` are
    identically distributed this is still an unbiased estimate of
    ``E_z[E_pot(f(z))]``, with twice the samples at no extra network cost.

    Args:
        params: MLP parameters ``theta`` (differentiated with respect to).
        key: PRNG key for this step's latent samples.
        p: Seed blend weight from :func:`model.seed_schedule`.
        energy_fn: Differentiable potential energy ``E_pot(q) -> scalar``.
        q_seed: Seed configuration, shape ``(n,)``.
        latent_dim: ``d``.
        config: ``sigma``, ``lambda`` and batch size.
        M: Mass matrix (see module docstring).

    Returns:
        ``(loss, aux)`` where ``aux`` holds the unweighted ``energy`` and
        ``isometry`` terms for logging. Suitable for
        ``jax.value_and_grad(..., has_aux=True)``.
    """
    z, z_prime = sample_latents(key, config.batch_size, latent_dim)

    f = jax.vmap(lambda zi: seeded_subspace_apply(params, zi, p, q_seed))
    q, q_prime = f(z), f(z_prime)

    energies = jax.vmap(energy_fn)(jnp.concatenate([q, q_prime], axis=0))
    energy_term = jnp.mean(energies)

    iso = jax.vmap(isometry_penalty, in_axes=(0, 0, 0, 0, None, None))(
        q, q_prime, z, z_prime, config.sigma, M
    )
    isometry_term = jnp.mean(iso)

    loss = energy_term + config.lam * isometry_term
    return loss, {"energy": energy_term, "isometry": isometry_term}


def make_loss_fn(
    energy_fn: EnergyFn,
    q_seed: Array,
    latent_dim: int,
    config: ObjectiveConfig,
    M: Array | None = None,
) -> Callable[[MLPParams, Array, Array | float], tuple[Array, dict[str, Array]]]:
    """Bind the system description into a loss ``(params, key, p) -> (L, aux)``.

    The returned function has only array arguments, so it can be passed
    straight to ``jax.jit(jax.value_and_grad(loss_fn, has_aux=True))``.
    """

    def loss_fn(params: MLPParams, key: Array, p: Array | float):
        return subspace_loss(
            params,
            key,
            p,
            energy_fn=energy_fn,
            q_seed=q_seed,
            latent_dim=latent_dim,
            config=config,
            M=M,
        )

    return loss_fn


# -----------------------------------------------------------------------------
# Generic potential building blocks (Sec. 5)
# -----------------------------------------------------------------------------


def penalty_energy(
    eq_constraints: Array | None = None,
    ineq_constraints: Array | None = None,
    w_eq: float = 1.0,
    w_ineq: float = 1.0,
) -> Array:
    """Quadratic penalty for constraints (joints, collisions, ...).

        w_eq |C_eq(q)|^2 + w_ineq |min(C_ineq(q), 0)|^2

    Takes already-evaluated constraint values so it is independent of any
    particular system. Equality constraints are satisfied at ``C_eq = 0``;
    inequality constraints at ``C_ineq >= 0`` (only violations are penalised).

    Args:
        eq_constraints: Values ``C_eq(q)``, any shape, or ``None``.
        ineq_constraints: Values ``C_ineq(q)``, any shape, or ``None``.
        w_eq, w_ineq: Penalty stiffnesses.

    Returns:
        Scalar penalty energy, to be added to a system's ``E_pot``.
    """
    total = jnp.asarray(0.0)
    if eq_constraints is not None:
        total = total + w_eq * jnp.sum(eq_constraints**2)
    if ineq_constraints is not None:
        total = total + w_ineq * jnp.sum(jnp.minimum(ineq_constraints, 0.0) ** 2)
    return total
