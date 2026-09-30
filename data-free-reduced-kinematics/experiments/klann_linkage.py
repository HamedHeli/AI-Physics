"""Klann linkage: a 1D neural subspace learned from the energy alone (Sec. 5).

Every body in the linkage floats freely, and constraint penalties are the only
thing holding it together. The network is never told that the mechanism has
one degree of freedom. Training should discover that the low-energy
configurations form a 1D curve (the crank cycle) and parameterise it with a
single latent coordinate.

System
------
7 rigid bodies in 3D, each stored as a free 12-vector

    q_i = [ t_i (3) | R_i row-major (9) ],   world point x = R_i p + t_i

with ``p`` a body-local point (relative to the body's centre of mass). ``R_i``
is an unconstrained 3x3 matrix: no angles, quaternions or relative
coordinates. 7 x 12 = 84 = n.

Bodies:
    0 anchor          fixed to the world by penalty
    1 frame           welded to the anchor; holds the three ground pivots
    2 crank           frame -- crank pin
    3 connecting arm  ternary: crank, lower rocker, leg
    4 lower rocker    frame -- connecting arm
    5 upper rocker    frame -- leg
    6 leg             ternary: connecting arm, upper rocker, foot

The Klann mechanism itself is the 6-bar (bodies 1-6, 7 pin joints). The
separate anchor body brings the count to 7 bodies, and so to n = 84.

Energy ("rigid + penalty")
--------------------------
Rigid bodies store no elastic energy, so the potential consists only of the
equality penalty ``w_eq |C_eq(q)|^2`` (``src.energy.penalty_energy``), plus
optional gravity. ``C_eq`` stacks:

* orthonormality   R_i^T R_i - I             (keeps each body rigid)
* world anchor     t_0 - t_0^seed, R_0 - I
* weld             4 frame points == the same 4 anchor points
* pin joints       2 points on each hinge axis (z = +-h) coincide, which
                   makes a revolute joint about z and keeps the motion planar

Mass metric
-----------
Each body is a cloud of sample points of equal mass. Its kinetic energy is
``m |t'|^2 + tr(R' J R'^T)`` with ``J = sum_k m_k p_k p_k^T``, so the exact
mass matrix is block-diagonal with blocks ``m I_3`` (translation) and
``I_3 (x) J`` (rotation rows). The isometry penalty measures distances in
this metric.

Usage
-----
    python experiments/klann_linkage.py                   # train + evaluate
    python experiments/klann_linkage.py --steps 60000 --w-eq 1e4
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.energy import ObjectiveConfig, make_loss_fn, penalty_energy  # noqa: E402
from src.model import MLPParams, SubspaceConfig, subspace_apply  # noqa: E402
from src.training import save_params, train  # noqa: E402

# -----------------------------------------------------------------------------
# Experiment specification (paper settings)
# -----------------------------------------------------------------------------

N_BODIES = 7
DOF_PER_BODY = 12  # translation (3) + unconstrained 3x3 matrix (9)
CONFIG_DIM = N_BODIES * DOF_PER_BODY  # n = 84
LATENT_DIM = 1  # d
COND_DIM = 0  # no conditional parameters (Sec. 3.5 unused)
HIDDEN_DIMS = (64,) * 5  # 5 hidden layers of width 64
SIGMA = 1.0
LAMBDA = 1.0

assert CONFIG_DIM == 84

# -----------------------------------------------------------------------------
# Linkage geometry (seed / pinned-joint configuration)
# -----------------------------------------------------------------------------

# Joint locations in the plane (z = 0) with the crank at angle 0. These
# proportions were chosen so that the crank can turn a full 360 deg and the
# linkage stays well away from lock-up at every angle. Link lengths are not
# listed separately: they follow from these points.
JOINTS_2D: dict[str, tuple[float, float]] = {
    "O": (0.00, 0.00),  # crank pivot (frame)
    "A": (0.25, 0.00),  # crank pin
    "F1": (-0.65, -0.23),  # lower rocker pivot (frame)
    "F2": (-0.46, 0.35),  # upper rocker pivot (frame)
    "B": (-0.27, -0.68),  # lower rocker -- connecting arm
    "C": (-0.32, -0.07),  # connecting arm -- leg
    "D": (-0.94, 0.16),  # upper rocker -- leg
    "E": (-0.18, -1.22),  # foot (on the leg, not a joint)
}

# Each body is drawn as the polygon through these points (a closed triangle
# for three points, a single bar for two). The anchor is a small plate.
BODY_NAMES = ["anchor", "frame", "crank", "conn_arm", "lower_rocker", "upper_rocker", "leg"]
BODY_OUTLINES: dict[str, list[str]] = {
    "frame": ["O", "F1", "F2"],
    "crank": ["O", "A"],
    "conn_arm": ["A", "B", "C"],
    "lower_rocker": ["F1", "B"],
    "upper_rocker": ["F2", "D"],
    "leg": ["C", "D", "E"],
}
ANCHOR_HALF_SIZE = 0.1  # anchor plate is a square centred on O

# Revolute joints: (body_a, body_b, joint point).
PIN_JOINTS: list[tuple[str, str, str]] = [
    ("frame", "crank", "O"),
    ("crank", "conn_arm", "A"),
    ("frame", "lower_rocker", "F1"),
    ("lower_rocker", "conn_arm", "B"),
    ("frame", "upper_rocker", "F2"),
    ("upper_rocker", "leg", "D"),
    ("conn_arm", "leg", "C"),
]

# Link cross-section used to build the sample-point clouds (affects mass and
# inertia only; the linkage is still planar).
LINK_WIDTH = 0.04
LINK_DEPTH = 0.04
HINGE_HALF_LENGTH = 0.05  # h: axial offset of the two hinge points
DENSITY = 1.0  # mass per unit link length


def _p3(name: str) -> np.ndarray:
    x, y = JOINTS_2D[name]
    return np.array([x, y, 0.0])


# -----------------------------------------------------------------------------
# Rigid body description
# -----------------------------------------------------------------------------


class RigidBody(NamedTuple):
    """Mass properties of one body, all in world coordinates at the seed."""

    mass: float
    com: np.ndarray  # (3,) centre of mass = seed translation
    J: np.ndarray  # (3, 3) second moment sum m_k p_k p_k^T about the COM


def _thicken(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Replace each centre-line point by 4 points offset in width and depth."""
    normal = np.array([-direction[1], direction[0], 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    offsets = [
        sw * 0.5 * LINK_WIDTH * normal + sd * 0.5 * LINK_DEPTH * ez
        for sw, sd in itertools.product((-1, 1), (-1, 1))
    ]
    return np.concatenate([points + o for o in offsets])


def _sample_body_points(name: str, per_unit_length: int = 40) -> tuple[np.ndarray, float]:
    """Sample points over a body's outline; returns (points, total length)."""
    if name == "anchor":
        s = ANCHOR_HALF_SIZE
        corners = [np.array(c) for c in [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)]]
    else:
        corners = [_p3(p) for p in BODY_OUTLINES[name]]
    edges = list(zip(corners, corners[1:] + corners[:1])) if len(corners) > 2 else [tuple(corners)]
    clouds, total_len = [], 0.0
    for a, b in edges:
        length = float(np.linalg.norm(b - a))
        k = max(2, int(np.ceil(length * per_unit_length)))
        t = np.linspace(0.0, 1.0, k)[:, None]
        clouds.append(_thicken(a + t * (b - a), (b - a) / length))
        total_len += length
    return np.concatenate(clouds), total_len


def build_bodies() -> list[RigidBody]:
    """Compute mass, centre of mass and second moment for every body."""
    bodies = []
    for name in BODY_NAMES:
        pts, length = _sample_body_points(name)
        mass = DENSITY * length
        m_k = mass / len(pts)
        com = pts.mean(axis=0)
        p = pts - com
        bodies.append(RigidBody(mass, com, m_k * p.T @ p))
    return bodies


def build_mass_matrix(bodies: list[RigidBody]) -> np.ndarray:
    """Exact kinetic-energy metric on the 84-vector (block diagonal).

    For ``q_i = [t, vec(R)]`` (``R`` row-major) the kinetic energy of the
    point cloud is ``m |t'|^2 + sum_rows r' J r'^T``, giving blocks ``m I_3``
    and ``kron(I_3, J)``.
    """
    M = np.zeros((CONFIG_DIM, CONFIG_DIM))
    for i, b in enumerate(bodies):
        s = i * DOF_PER_BODY
        M[s : s + 3, s : s + 3] = b.mass * np.eye(3)
        M[s + 3 : s + 12, s + 3 : s + 12] = np.kron(np.eye(3), b.J)
    return M


def seed_configuration(bodies: list[RigidBody]) -> np.ndarray:
    """The pinned-joint seed: every body at its assembled pose (t = COM, R = I)."""
    q = np.zeros((N_BODIES, DOF_PER_BODY))
    for i, b in enumerate(bodies):
        q[i, :3] = b.com
        q[i, 3:] = np.eye(3).ravel()
    return q.ravel()


def unpack(q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split ``(..., 84)`` into translations ``(..., 7, 3)`` and matrices ``(..., 7, 3, 3)``."""
    batch = q.shape[:-1]
    q = q.reshape(*batch, N_BODIES, DOF_PER_BODY)
    return q[..., :3], q[..., 3:].reshape(*batch, N_BODIES, 3, 3)


# -----------------------------------------------------------------------------
# Energy: rigid + penalty
# -----------------------------------------------------------------------------


class ConstraintData(NamedTuple):
    """Body-local constraint points, precomputed from the seed geometry.

    Pairs ``(idx_a, idx_b, local_a, local_b)``: body ``idx_a`` at local point
    ``local_a`` must coincide with body ``idx_b`` at ``local_b``.
    """

    idx_a: np.ndarray  # (K,)
    idx_b: np.ndarray  # (K,)
    local_a: np.ndarray  # (K, 3)
    local_b: np.ndarray  # (K, 3)
    anchor_t: np.ndarray  # (3,) seed translation of the anchor


def build_constraints(bodies: list[RigidBody]) -> ConstraintData:
    idx = {n: i for i, n in enumerate(BODY_NAMES)}
    ez = np.array([0.0, 0.0, HINGE_HALF_LENGTH])
    ia, ib, la, lb = [], [], [], []

    def coincide(a: str, b: str, world_pt: np.ndarray) -> None:
        ia.append(idx[a])
        ib.append(idx[b])
        la.append(world_pt - bodies[idx[a]].com)
        lb.append(world_pt - bodies[idx[b]].com)

    # Revolute joints: two points on the z-axis through the joint.
    for a, b, pt in PIN_JOINTS:
        for sign in (-1.0, 1.0):
            coincide(a, b, _p3(pt) + sign * ez)

    # Weld frame to anchor: 4 non-coplanar points fix the relative transform.
    origin = _p3("O")
    for offset in [np.zeros(3), *np.eye(3) * 0.1]:
        coincide("frame", "anchor", origin + offset)

    return ConstraintData(
        np.array(ia), np.array(ib), np.array(la), np.array(lb), bodies[0].com.copy()
    )


def make_energy_fn(
    cons: ConstraintData,
    bodies: list[RigidBody],
    w_eq: float,
    gravity: float = 0.0,
):
    """Build ``E_pot(q) = w_eq |C_eq(q)|^2 + gravity`` for the linkage."""
    idx_a, idx_b = jnp.asarray(cons.idx_a), jnp.asarray(cons.idx_b)
    local_a, local_b = jnp.asarray(cons.local_a), jnp.asarray(cons.local_b)
    anchor_t = jnp.asarray(cons.anchor_t)
    masses = jnp.asarray([b.mass for b in bodies])
    eye = jnp.eye(3)

    def constraint_residuals(q: jax.Array) -> jax.Array:
        t, R = unpack(q)
        # x = R p + t for every constraint point on both sides.
        xa = jnp.einsum("kij,kj->ki", R[idx_a], local_a) + t[idx_a]
        xb = jnp.einsum("kij,kj->ki", R[idx_b], local_b) + t[idx_b]
        joint = (xa - xb).ravel()
        ortho = (jnp.einsum("bki,bkj->bij", R, R) - eye).ravel()
        anchor = jnp.concatenate([t[0] - anchor_t, (R[0] - eye).ravel()])
        return jnp.concatenate([joint, ortho, anchor])

    def energy_fn(q: jax.Array) -> jax.Array:
        e = penalty_energy(eq_constraints=constraint_residuals(q), w_eq=w_eq)
        if gravity:
            t, _ = unpack(q)
            e = e + gravity * jnp.sum(masses * t[:, 1])
        return e

    return energy_fn, constraint_residuals


# -----------------------------------------------------------------------------
# Analytic reference kinematics (evaluation only, never used in training)
# -----------------------------------------------------------------------------


def _circle_intersect(P, r, Q, s, sign):
    d = np.linalg.norm(Q - P)
    a = (r * r - s * s + d * d) / (2 * d)
    h = np.sqrt(max(r * r - a * a, 0.0))
    u = (Q - P) / d
    return P + a * u + sign * h * np.array([-u[1], u[0]])


def _carry(P1, P2, Q1, Q2, X):
    """Move point X rigidly with the segment P1P2 -> Q1Q2 (2D)."""
    t = np.arctan2(*(Q2 - Q1)[::-1]) - np.arctan2(*(P2 - P1)[::-1])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return Q1 + R @ (X - P1)


class KlannReference:
    """Solve loop closure for a given crank angle (ground truth).

    Used only to check the learned subspace. The assembly branch (which of
    the two circle-intersection solutions) is the one containing the seed.
    """

    def __init__(self) -> None:
        g = {k: np.array(v) for k, v in JOINTS_2D.items()}
        self.g = g
        self.r = np.linalg.norm(g["A"] - g["O"])
        self.phi0 = np.arctan2(*(g["A"] - g["O"])[::-1])
        self.lB = np.linalg.norm(g["B"] - g["F1"])
        self.lAB = np.linalg.norm(g["B"] - g["A"])
        self.lD = np.linalg.norm(g["D"] - g["F2"])
        self.lCD = np.linalg.norm(g["D"] - g["C"])
        self.sB = self.sD = None
        for sB, sD in itertools.product((1, -1), (1, -1)):
            self.sB, self.sD = sB, sD
            j = self.solve(self.phi0)
            if np.allclose(j["B"], g["B"], atol=1e-6) and np.allclose(j["D"], g["D"], atol=1e-6):
                return
        raise RuntimeError("seed geometry not on any assembly branch")

    def solve(self, phi: float) -> dict[str, np.ndarray]:
        g = self.g
        A = g["O"] + self.r * np.array([np.cos(phi), np.sin(phi)])
        B = _circle_intersect(g["F1"], self.lB, A, self.lAB, self.sB)
        C = _carry(g["A"], g["B"], A, B, g["C"])
        D = _circle_intersect(g["F2"], self.lD, C, self.lCD, self.sD)
        E = _carry(g["C"], g["D"], C, D, g["E"])
        return {"A": A, "B": B, "C": C, "D": D, "E": E}


# -----------------------------------------------------------------------------
# Post-training sampling and verification
# -----------------------------------------------------------------------------


def world_joints(q: np.ndarray, bodies: list[RigidBody]) -> dict[str, np.ndarray]:
    """Planar world positions of the joints/foot implied by configuration q.

    Every point is read off the body it belongs to (A from the crank, B from
    the lower rocker, C and E from the leg, D from the upper rocker).
    """
    t, R = (np.asarray(x) for x in unpack(jnp.asarray(q)))
    idx = {n: i for i, n in enumerate(BODY_NAMES)}
    owner = {"O": "crank", "A": "crank", "B": "lower_rocker", "C": "leg",
             "D": "upper_rocker", "E": "leg", "F1": "lower_rocker", "F2": "upper_rocker"}
    out = {}
    for pt, body in owner.items():
        i = idx[body]
        out[pt] = (R[i] @ (_p3(pt) - bodies[i].com) + t[i])[:2]
    return out


def evaluate(
    params: MLPParams,
    bodies: list[RigidBody],
    energy_fn,
    residual_fn,
    M: np.ndarray,
    out_dir: Path,
    n_samples: int = 241,
    z_max: float = 3.0,
) -> dict[str, np.ndarray]:
    """Sample the 1D latent line and check it traces the Klann motion.

    For z on a uniform grid in [-z_max, z_max] (~99.7% of N(0, 1) mass) this
    reports:

    * energy and the worst constraint residual (joint gap, non-rigidity),
    * the crank angle, read from the crank's matrix R. This is a diagnostic
      only; the model itself never uses angles,
    * the error against analytic loop closure at that crank angle. A small
      error means every sample is a genuine Klann configuration, i.e. the
      subspace lies on the mechanism's 1-DOF curve,
    * how far the latent line travels in the mass metric compared with
      sigma * |dz|.
    """
    zs = np.linspace(-z_max, z_max, n_samples)[:, None]
    f = jax.jit(jax.vmap(lambda z: subspace_apply(params, z)))
    qs = np.asarray(f(jnp.asarray(zs)))

    energies = np.asarray(jax.vmap(energy_fn)(qs))
    residuals = np.abs(np.asarray(jax.vmap(residual_fn)(qs)))
    _, R = unpack(jnp.asarray(qs))
    R = np.asarray(R)
    crank = BODY_NAMES.index("crank")
    ref = KlannReference()
    crank_angle = np.unwrap(np.arctan2(R[:, crank, 1, 0], R[:, crank, 0, 0])) + ref.phi0

    ref_err = np.zeros(n_samples)
    joints = []
    for k, q in enumerate(qs):
        wj = world_joints(q, bodies)
        rj = ref.solve(crank_angle[k])
        ref_err[k] = max(np.linalg.norm(wj[n] - rj[n]) for n in "ABCDE")
        joints.append(wj)

    dq = np.diff(qs, axis=0)
    step_len = np.sqrt(np.einsum("ki,ij,kj->k", dq, M, dq))
    dz = np.diff(zs[:, 0])

    print("\n=== Latent sampling (z in [-%.1f, %.1f], %d samples) ===" % (z_max, z_max, n_samples))
    print(f"potential energy          : mean {energies.mean():.3e}   max {energies.max():.3e}")
    print(f"max |constraint residual| : {residuals.max():.3e}")
    print(f"crank angle swept         : {np.degrees(np.ptp(crank_angle)):.1f} deg"
          f"   (monotone: {bool(np.all(np.diff(crank_angle) > 0) or np.all(np.diff(crank_angle) < 0))})")
    # z ~ N(0, 1): |z| <= 1, 2, 3 hold ~68%, 95%, 99.7% of training samples.
    for band in (1.0, 2.0, 3.0):
        m = np.abs(zs[:, 0]) <= band
        print(f"|z| <= {band:.0f}: max joint error vs analytic loop closure {ref_err[m].max():.3e}"
              f"   max E_pot {energies[m].max():.3e}")
    print(f"isometry ratio |dq|_M / (sigma |dz|): "
          f"mean {np.mean(step_len / (SIGMA * dz)):.3f}   "
          f"min {np.min(step_len / (SIGMA * dz)):.3f}   max {np.max(step_len / (SIGMA * dz)):.3f}")

    results = dict(z=zs[:, 0], q=qs, energy=energies, crank_angle=crank_angle,
                   ref_error=ref_err, foot=np.array([j["E"] for j in joints]))
    np.savez(out_dir / "samples.npz", **results)
    _plot(results, joints, ref, out_dir)
    return results


def _plot(results, joints, ref: KlannReference, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    z = results["z"]

    ax = axes[0]
    ref_foot = np.array([ref.solve(a)["E"] for a in np.linspace(0, 2 * np.pi, 361)])
    ax.plot(*ref_foot.T, "k--", lw=1, zorder=4, label="analytic foot path (full cycle)")
    for k in np.linspace(0, len(joints) - 1, 9).astype(int):
        j = joints[k]
        g = ref.g
        segs = [(g["O"], j["A"]), (j["A"], j["B"]), (j["B"], j["C"]), (j["C"], j["A"]),
                (g["F1"], j["B"]), (g["F2"], j["D"]), (j["C"], j["D"]), (j["D"], j["E"]),
                (j["E"], j["C"])]
        color = plt.cm.viridis((z[k] - z.min()) / np.ptp(z))
        for a, b in segs:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.5, alpha=0.8)
    sc = ax.scatter(*results["foot"].T, c=z, s=6, cmap="viridis", zorder=3)
    fig.colorbar(sc, ax=ax, label="latent z")
    ax.set_aspect("equal")
    ax.set_title("Linkage sampled along the 1D latent")
    ax.legend(loc="upper right", fontsize=8)

    axes[1].plot(z, np.degrees(results["crank_angle"]))
    axes[1].set_xlabel("latent z")
    axes[1].set_ylabel("crank angle (deg)")
    axes[1].set_title("Crank angle vs latent")

    axes[2].semilogy(z, results["ref_error"], label="joint error vs analytic")
    axes[2].semilogy(z, np.maximum(results["energy"], 1e-12), label="E_pot")
    axes[2].set_xlabel("latent z")
    axes[2].set_title("Accuracy along the latent")
    axes[2].legend()

    fig.tight_layout()
    path = out_dir / "klann_latent_samples.png"
    fig.savefig(path, dpi=130)
    print(f"saved plot to {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--ramp-frac", type=float, default=0.5,
                        help="fraction of training over which p ramps 0 -> 1")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--w-eq", type=float, default=1e4, help="joint penalty stiffness")
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "experiments" / "outputs" / "klann_linkage")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bodies = build_bodies()
    M = build_mass_matrix(bodies)
    q_seed = seed_configuration(bodies)
    cons = build_constraints(bodies)
    energy_fn, residual_fn = make_energy_fn(cons, bodies, args.w_eq, args.gravity)

    print(f"Klann linkage: {N_BODIES} free rigid bodies, n={CONFIG_DIM}, "
          f"d={LATENT_DIM}, cond={COND_DIM}, MLP {list(HIDDEN_DIMS)}, "
          f"sigma={SIGMA}, lambda={LAMBDA}, w_eq={args.w_eq:g}")
    print(f"seed energy (should be ~0): {float(energy_fn(jnp.asarray(q_seed))):.3e}")

    subspace = SubspaceConfig(LATENT_DIM + COND_DIM, CONFIG_DIM, HIDDEN_DIMS)
    objective = ObjectiveConfig(sigma=SIGMA, lam=LAMBDA, batch_size=args.batch_size)
    loss_fn = make_loss_fn(energy_fn, jnp.asarray(q_seed), LATENT_DIM, objective, M=jnp.asarray(M))

    params = train(loss_fn, subspace, args.steps, int(args.ramp_frac * args.steps),
                   args.lr, args.seed, args.log_every)

    save_params(params, args.out_dir / "params.npz")

    evaluate(params, bodies, energy_fn, residual_fn, M, args.out_dir)


if __name__ == "__main__":
    main()
