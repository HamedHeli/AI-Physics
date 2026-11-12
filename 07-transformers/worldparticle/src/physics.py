"""Ground-truth physics: a jelly block and a sand pile falling in a 2D box.

These two small simulators play the role of the six classical solvers
(DFSPH, GIPC, MPM, VBD, OpenMM) in Wang et al. (2026), "WorldParticle: Unified
World Simulation of Lagrangian Particle Dynamics via Transformer". Like the
paper, the two **categories** share one particle representation and one
network architecture, but each category gets its own training set and its own
weights (paper Sec. 5, "Limitations").

Both scenes live in the unit box ``[0, 1]^2`` with gravity ``g`` pointing down
and are made of ``N = 64`` particles of mass 1.

Jelly (an elastic solid, topology branch used)
----------------------------------------------
An 8 x 8 lattice of particles joined by springs to its 8 neighbours (structural
and diagonal). The springs are the jelly's **topology**: they are fixed for the
whole sequence and are given to the model, like the rest-shape mesh of the
paper's cloth and elastic-solid categories. For the spring between ``i`` and
``j`` with rest length ``L_ij``::

    d   = x_j - x_i,   n = d / |d|
    F_i = k (|d| - L_ij) n + c_s ((v_j - v_i) . n) n

The **stiffness** ``k`` changes from sequence to sequence and is the jelly's
material attribute (like ``E`` for the paper's cloth).

Sand (a granular material, topology branch zero-filled)
-------------------------------------------------------
64 grains of radius ``r`` with no springs. Grains interact only when they
overlap, through a spring-dashpot normal force plus a tangential friction force
that opposes sliding, capped by Coulomb's law::

    overlap = 2r - |d| > 0
    F_n     = k_c overlap + c_n (relative normal speed)
    F_t     = -min(c_t |v_t|, mu F_n) v_t / |v_t|

The **friction coefficient** ``mu`` changes from sequence to sequence and is
the sand's material attribute (as in the paper's MPM sand).

Walls (both categories)
-----------------------
The four walls push back on any particle closer than ``r`` with the same
spring-dashpot-plus-friction law (wall friction uses the category's own
``mu``, a fixed 0.5 for the jelly). The model sees the walls only through
**boundary samples**: points along each wall with an outward normal, the
paper's particle-boundary branch.

Integration
-----------
Semi-implicit Euler with ``substeps`` small steps per saved frame, so the
stiff springs and contacts stay stable. The learned model will take **one**
step per frame, ``substeps`` times larger: it has to learn what the many small
steps add up to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class SimParams(NamedTuple):
    """Constants shared by both categories.

    Attributes:
        n_side: The jelly is an ``n_side x n_side`` lattice; the sand has as many grains.
        spacing: Rest distance between jelly neighbours.
        radius: Particle radius ``r`` (contact with walls and, for sand, between grains).
        gravity: Downward acceleration ``g``.
        frame_dt: Time between saved frames, the step the model learns.
        substeps: Small solver steps per frame.
        n_frames: Saved frames per sequence (the initial state is frame 0).
        k_contact: Contact spring constant (walls and grains).
        c_contact: Contact normal damping.
        c_spring: Damping along each jelly spring.
        c_tangent: Viscous friction coefficient before the Coulomb cap.
        k_range: Jelly stiffness range ``(k_min, k_max)`` for training sequences.
        mu_range: Sand friction range ``(mu_min, mu_max)`` for training sequences.
        n_wall: Boundary samples per wall.
    """

    n_side: int = 8
    spacing: float = 0.05
    radius: float = 0.025
    gravity: float = 3.0
    frame_dt: float = 0.005
    substeps: int = 10
    n_frames: int = 240
    k_contact: float = 3000.0
    c_contact: float = 8.0
    c_spring: float = 2.0
    c_tangent: float = 200.0
    k_range: tuple = (4000.0, 12000.0)
    mu_range: tuple = (0.1, 0.9)
    n_wall: int = 21

    @property
    def n(self) -> int:
        return self.n_side * self.n_side


# ----------------------------------------------------------------------------- topology and boundary


def jelly_springs(p: SimParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest lattice and spring list of the jelly.

    Returns:
        ``(rest, edges, adjacency)``: rest positions ``(N, 2)`` centred at the
        origin, spring endpoints ``(E, 2)``, and the symmetric 0/1 adjacency
        matrix ``(N, N)`` the model's topology branch uses.
    """
    s = p.n_side
    ij = np.stack(np.meshgrid(np.arange(s), np.arange(s), indexing="ij"), -1).reshape(-1, 2)
    rest = (ij - (s - 1) / 2) * p.spacing
    edges = []
    for a in range(s * s):
        ia, ja = ij[a]
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            ib, jb = ia + di, ja + dj
            if 0 <= ib < s and 0 <= jb < s:
                edges.append((a, ib * s + jb))
    edges = np.array(edges)
    adj = np.zeros((s * s, s * s), np.float32)
    adj[edges[:, 0], edges[:, 1]] = adj[edges[:, 1], edges[:, 0]] = 1.0
    return rest.astype(np.float32), edges, adj


def boundary_samples(p: SimParams) -> tuple[np.ndarray, np.ndarray]:
    """Points along the four walls and their inward normals, ``(4 n_wall, 2)`` each.

    The normal points into the box, the direction a wall pushes.
    """
    t = np.linspace(0, 1, p.n_wall)
    z, o = np.zeros_like(t), np.ones_like(t)
    pts = np.concatenate([np.c_[t, z], np.c_[t, o], np.c_[z, t], np.c_[o, t]])
    nrm = np.concatenate([np.c_[z, o], np.c_[z, -o], np.c_[o, z], np.c_[-o, z]])
    return pts.astype(np.float32), nrm.astype(np.float32)


# ----------------------------------------------------------------------------- forces


def _friction(v_t: Array, f_n: Array, mu: Array, c_t: float) -> Array:
    """Tangential force opposing the sliding velocity ``v_t``: viscous, capped at ``mu * f_n``."""
    speed = jnp.linalg.norm(v_t, axis=-1, keepdims=True)
    mag = jnp.minimum(c_t * speed, mu * f_n[..., None])
    return -mag * v_t / (speed + 1e-9)


def wall_force(x: Array, v: Array, mu: Array, p: SimParams) -> Array:
    """Force of the four walls on every particle, ``(N, 2)``."""
    f = jnp.zeros_like(x)
    for axis, side in [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)]:
        sign = 1.0 if side == 0.0 else -1.0                      # inward normal component
        n = jnp.zeros(2).at[axis].set(sign)
        gap = sign * (x[:, axis] - side)                          # distance to the wall
        overlap = jnp.maximum(p.radius - gap, 0.0)
        vn = v @ n
        f_n = jnp.where(overlap > 0, p.k_contact * overlap - p.c_contact * vn, 0.0)
        f_n = jnp.maximum(f_n, 0.0)
        v_t = v - vn[:, None] * n
        f_t = jnp.where((overlap > 0)[:, None], _friction(v_t, f_n, mu, p.c_tangent), 0.0)
        f = f + f_n[:, None] * n + f_t
    return f


def spring_force(x: Array, v: Array, k: Array, edges: Array, rest_len: Array, p: SimParams) -> Array:
    """Jelly springs, ``(N, 2)``."""
    d = x[edges[:, 1]] - x[edges[:, 0]]
    L = jnp.linalg.norm(d, axis=-1)
    n = d / L[:, None]
    dv = v[edges[:, 1]] - v[edges[:, 0]]
    f = (k * (L - rest_len) + p.c_spring * jnp.sum(dv * n, -1))[:, None] * n
    return jnp.zeros_like(x).at[edges[:, 0]].add(f).at[edges[:, 1]].add(-f)


def grain_force(x: Array, v: Array, mu: Array, p: SimParams) -> Array:
    """Sand grain contacts (all pairs; ``N = 64`` is small), ``(N, 2)``."""
    d = x[None, :, :] - x[:, None, :]                             # d[i, j] = x_j - x_i
    dist = jnp.linalg.norm(d + jnp.eye(x.shape[0])[..., None], axis=-1)
    n = d / dist[..., None]                                       # from i towards j
    overlap = jnp.maximum(2 * p.radius - dist, 0.0) * (1 - jnp.eye(x.shape[0]))
    dv = v[None, :, :] - v[:, None, :]
    vn = jnp.sum(dv * n, -1)                                      # > 0: separating
    f_n = jnp.where(overlap > 0, p.k_contact * overlap - p.c_contact * vn, 0.0)
    f_n = jnp.maximum(f_n, 0.0)
    v_t = dv - vn[..., None] * n
    f_t = jnp.where((overlap > 0)[..., None], -_friction(v_t, f_n, mu, p.c_tangent), 0.0)
    return jnp.sum(-f_n[..., None] * n + f_t, axis=1)


# ----------------------------------------------------------------------------- rollouts


def simulate(x0: Array, v0: Array, material: Array, category: str, p: SimParams) -> tuple[Array, Array]:
    """Ground-truth trajectory.

    Args:
        x0, v0: Initial positions and velocities, ``(N, 2)``.
        material: The category's attribute: stiffness ``k`` (jelly) or friction ``mu`` (sand).
        category: ``"jelly"`` or ``"sand"``.

    Returns:
        ``(X, V)``, each ``(n_frames, N, 2)``; frame 0 is the initial state.
    """
    h = p.frame_dt / p.substeps
    g = jnp.array([0.0, -p.gravity])
    if category == "jelly":
        rest, edges, _ = jelly_springs(p)
        rest_len = jnp.asarray(np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=-1))
        edges = jnp.asarray(edges)
        internal = lambda x, v: spring_force(x, v, material, edges, rest_len, p)
        mu_wall = 0.5
    else:
        internal = lambda x, v: grain_force(x, v, material, p)
        mu_wall = material

    def substep(state, _):
        x, v = state
        a = g + internal(x, v) + wall_force(x, v, mu_wall, p)
        v = v + h * a
        return (x + h * v, v), None

    def frame(state, _):
        state, _ = jax.lax.scan(substep, state, None, length=p.substeps)
        return state, state

    _, (X, V) = jax.lax.scan(frame, (x0, v0), None, length=p.n_frames - 1)
    return jnp.concatenate([x0[None], X]), jnp.concatenate([v0[None], V])


def initial_state(rng: np.random.Generator, category: str, p: SimParams) -> tuple[np.ndarray, np.ndarray]:
    """A random starting configuration.

    Jelly: the rest lattice, rotated by up to 30 degrees, dropped from a random
    height with a random sideways velocity. Sand: a tall, jittered column of
    grains standing on the floor, given a random sideways shove so it
    collapses into a pile whose spread depends on the friction.
    """
    rest, _, _ = jelly_springs(p)
    if category == "jelly":
        th = rng.uniform(-0.5, 0.5)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        c = np.array([rng.uniform(0.3, 0.7), rng.uniform(0.55, 0.7)])
        x = rest @ R.T + c
        v = np.tile([rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.0)], (p.n, 1))
    else:
        w = p.n_side // 2                                         # a column w wide, n / w tall
        ij = np.stack(np.meshgrid(np.arange(w), np.arange(p.n // w), indexing="ij"), -1).reshape(-1, 2)
        grid = ij * 2.1 * p.radius
        c = np.array([rng.uniform(0.2, 0.8) - grid[:, 0].max() / 2, p.radius + 0.005])
        x = grid + c + rng.normal(0, 0.002, grid.shape)
        v = np.tile([rng.uniform(-0.5, 0.5), 0.0], (p.n, 1)) + rng.normal(0, 0.05, grid.shape)
    return x.astype(np.float32), v.astype(np.float32)


def make_dataset(category: str, n_seq: int, seed: int, p: SimParams, material: np.ndarray | None = None) -> dict:
    """``n_seq`` simulated sequences of one category.

    Args:
        material: Optional per-sequence attribute; by default drawn uniformly from
            ``k_range`` (jelly) or ``mu_range`` (sand).

    Returns:
        ``{"X": (S, T, N, 2), "V": (S, T, N, 2), "material": (S,)}``.
    """
    rng = np.random.default_rng(seed)
    lo, hi = p.k_range if category == "jelly" else p.mu_range
    if material is None:
        material = rng.uniform(lo, hi, n_seq).astype(np.float32)
    starts = [initial_state(rng, category, p) for _ in range(n_seq)]
    x0 = jnp.asarray(np.stack([s[0] for s in starts]))
    v0 = jnp.asarray(np.stack([s[1] for s in starts]))
    sim = jax.jit(jax.vmap(lambda x, v, m: simulate(x, v, m, category, p)))
    X, V = sim(x0, v0, jnp.asarray(material))
    return {"X": np.asarray(X), "V": np.asarray(V), "material": np.asarray(material)}
