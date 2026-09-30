"""Cloth ball (Fig. 1): a 3D neural subspace for a ball rolling on pinned cloth.

A heavy ball rests on a circular cloth pinned around its rim, like a ball on
a trampoline. Training discovers a 3-dimensional latent space of low-energy
configurations (ball rolling around the sagging cloth, with the cloth
deforming to follow it) from the energy alone, starting from one seed.

System
------
* Cloth: triangulated disk. The rim vertices are pinned (fixed boundary
  values, not degrees of freedom); the 2022 interior vertices are free.
* Ball: rigid sphere represented by its centre only. A sphere's rotation
  changes neither gravity nor contact, so rotational DOFs would be
  zero-energy directions the isometry term could exploit; they are left out.

    q = [ x_1 .. x_2022 (3 each) | ball centre (3) ],   n = 6066 + 3 = 6069

Energy ("cloth + penalty")
--------------------------
    E_pot = E_stretch + E_bend + E_gravity + E_contact (+ E_floor)

* StVK stretching, constant strain per triangle:
  ``A0 (mu |E|_F^2 + lam_L/2 tr(E)^2)`` with Green strain
  ``E = (F^T F - I) / 2`` and ``F = Ds Dm^-1`` (3x2).
* Bending on interior edges (discrete shells):
  ``k_b * 3 |e|^2 / (A1 + A2) * theta^2``, where ``theta`` is the dihedral
  angle (rest angle 0: flat cloth).
* Gravity on lumped cloth masses and the ball.
* Contact: inequality penalty ``w_ineq |min(C_ineq, 0)|^2`` with
  ``C_ineq = |x_i - c| - (r + thickness)`` for every cloth vertex
  (``src.energy.penalty_energy``).
* Floor (safety): the same inequality penalty on ``c_z - r >= z_floor``.
  Without it, a ball placed beyond the rim could fall forever and make the
  energy unbounded below. The floor sits well below the cloth and is inactive
  near the seed.

Mass metric
-----------
Diagonal (lumped): each free vertex gets a third of the area of its adjacent
triangles times the areal density, and the ball its mass.

Usage
-----
    python experiments/cloth_ball.py                # seed solve + train + evaluate
    python experiments/cloth_ball.py --steps 5000   # quicker run

Outputs go to ``experiments/outputs/cloth_ball/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.energy import ObjectiveConfig, make_loss_fn, penalty_energy  # noqa: E402
from src.model import SubspaceConfig, subspace_apply  # noqa: E402
from src.training import save_params, train  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "experiments" / "outputs" / "cloth_ball"

# -----------------------------------------------------------------------------
# Experiment specification (paper settings)
# -----------------------------------------------------------------------------

N_FREE_VERTICES = 2022
BALL_DOF = 3
CONFIG_DIM = 3 * N_FREE_VERTICES + BALL_DOF  # n = 6069
LATENT_DIM = 3  # d
COND_DIM = 0
HIDDEN_DIMS = (128,) * 5  # 5 hidden layers of width 128
SIGMA = 0.05
LAMBDA = 1.0

assert CONFIG_DIM == 6069


# -----------------------------------------------------------------------------
# Physical parameters (SI units)
# -----------------------------------------------------------------------------


class PhysicalParams(NamedTuple):
    """Material, contact and scene constants. All can be overridden from the CLI."""

    mesh_spacing: float = 0.042  # triangle edge length (m); gives a ~1 m radius disk
    cloth_density: float = 0.1  # kg / m^2
    stretch_mu: float = 20.0  # StVK shear modulus (N / m)
    stretch_lambda: float = 20.0  # StVK first Lame parameter (N / m)
    bend_stiffness: float = 1e-3  # k_b (N m)
    ball_radius: float = 0.2  # m
    ball_mass: float = 0.2  # kg
    gravity: float = 9.81  # m / s^2
    contact_stiffness: float = 1e4  # w_ineq
    contact_thickness: float = 0.005  # m; cloth "skin" around the ball
    floor_height: float = -1.0  # m; lowest point the ball's bottom can reach


# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------


class ClothMesh(NamedTuple):
    rest: np.ndarray  # (V, 3) rest positions, free vertices first, rim last
    tris: np.ndarray  # (T, 3) vertex indices
    n_free: int
    radius: float  # rim radius


def build_disk_mesh(n_free: int, spacing: float) -> ClothMesh:
    """Triangulated disk with exactly ``n_free`` interior vertices.

    Interior vertices are the ``n_free`` points of a triangular lattice
    closest to the origin (a small offset breaks ties, so any count is
    exact). A ring of rim vertices just outside them is added and everything
    is Delaunay-triangulated. The rim ring is convex, so the triangulation
    covers exactly the disk.
    """
    k = int(np.ceil(1.5 * np.sqrt(n_free)))
    ii, jj = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    lattice = np.stack([(ii + 0.5 * (jj % 2)) * spacing, jj * spacing * np.sqrt(3) / 2], -1)
    lattice = lattice.reshape(-1, 2) + np.array([0.0137, 0.0071]) * spacing / 0.042
    free = lattice[np.argsort(np.linalg.norm(lattice, axis=1))[:n_free]]

    radius = np.linalg.norm(free, axis=1).max() + 0.8 * spacing
    n_rim = int(round(2 * np.pi * radius / spacing))
    theta = np.linspace(0.0, 2 * np.pi, n_rim, endpoint=False)
    rim = radius * np.stack([np.cos(theta), np.sin(theta)], -1)

    xy = np.vstack([free, rim])
    tris = Delaunay(xy).simplices
    # Orient every triangle counter-clockwise so normals point up (+z).
    a, b, c = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    cw = ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) < 0
    tris[cw] = tris[cw][:, [0, 2, 1]]

    rest = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
    return ClothMesh(rest, tris, n_free, float(radius))


def interior_hinges(tris: np.ndarray) -> np.ndarray:
    """Edges shared by two triangles, as ``(e0, e1, a, b)`` rows.

    ``a`` and ``b`` are the vertices opposite the edge, ordered so that for
    counter-clockwise triangles the normals ``(e1 - e0) x (a - e0)`` and
    ``(b - e0) x (e1 - e0)`` agree when the cloth is flat.
    """
    owner: dict[tuple[int, int], tuple[int, int, int]] = {}
    hinges = []
    for tri in tris:
        for k in range(3):
            i, j, opp = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            key = (min(i, j), max(i, j))
            if key in owner:
                oi, oj, oopp = owner.pop(key)
                # (oi, oj, oopp) is CCW, so oopp lies left of oi -> oj.
                hinges.append((oi, oj, oopp, opp))
            else:
                owner[key] = (i, j, opp)
    return np.array(hinges)


# -----------------------------------------------------------------------------
# System assembly
# -----------------------------------------------------------------------------


class ClothBallSystem(NamedTuple):
    """Everything the energy needs, precomputed from the rest mesh."""

    mesh: ClothMesh
    phys: PhysicalParams
    rim: np.ndarray  # (V - n_free, 3) fixed rim positions
    dm_inv: np.ndarray  # (T, 2, 2) inverse rest edge matrices
    rest_area: np.ndarray  # (T,)
    hinges: np.ndarray  # (H, 4)
    hinge_weight: np.ndarray  # (H,) 3 |e|^2 / (A1 + A2)
    vertex_mass: np.ndarray  # (V,)


def build_system(phys: PhysicalParams) -> ClothBallSystem:
    mesh = build_disk_mesh(N_FREE_VERTICES, phys.mesh_spacing)
    X = mesh.rest[:, :2]
    t = mesh.tris

    Dm = np.stack([X[t[:, 1]] - X[t[:, 0]], X[t[:, 2]] - X[t[:, 0]]], axis=-1)  # (T, 2, 2)
    area = 0.5 * np.abs(np.linalg.det(Dm))
    dm_inv = np.linalg.inv(Dm)

    vertex_mass = np.zeros(len(X))
    np.add.at(vertex_mass, t.ravel(), np.repeat(phys.cloth_density * area / 3.0, 3))

    hinges = interior_hinges(t)
    tri_area = {tuple(sorted(tri)): a for tri, a in zip(t.tolist(), area)}
    e_len2 = np.sum((X[hinges[:, 1]] - X[hinges[:, 0]]) ** 2, axis=1)
    a1 = np.array([tri_area[tuple(sorted((h[0], h[1], h[2])))] for h in hinges])
    a2 = np.array([tri_area[tuple(sorted((h[0], h[1], h[3])))] for h in hinges])

    return ClothBallSystem(
        mesh=mesh,
        phys=phys,
        rim=mesh.rest[mesh.n_free :],
        dm_inv=dm_inv,
        rest_area=area,
        hinges=hinges,
        hinge_weight=3.0 * e_len2 / (a1 + a2),
        vertex_mass=vertex_mass,
    )


def mass_vector(system: ClothBallSystem) -> np.ndarray:
    """Diagonal (lumped) mass matrix as an ``(n,)`` vector."""
    cloth = np.repeat(system.vertex_mass[: system.mesh.n_free], 3)
    return np.concatenate([cloth, np.full(BALL_DOF, system.phys.ball_mass)])


def unpack(q: jax.Array, rim: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split ``q`` into all cloth vertices ``(V, 3)`` (rim appended) and ball centre ``(3,)``."""
    free = q[: 3 * N_FREE_VERTICES].reshape(N_FREE_VERTICES, 3)
    return jnp.concatenate([free, rim], axis=0), q[3 * N_FREE_VERTICES :]


# -----------------------------------------------------------------------------
# Energy: cloth + penalty
# -----------------------------------------------------------------------------


def make_energy_fn(system: ClothBallSystem):
    """Build ``E_pot(q)`` and a function returning its individual terms."""
    ph = system.phys
    tris = jnp.asarray(system.mesh.tris)
    dm_inv = jnp.asarray(system.dm_inv)
    area = jnp.asarray(system.rest_area)
    hinges = jnp.asarray(system.hinges)
    hinge_w = jnp.asarray(system.hinge_weight)
    vmass = jnp.asarray(system.vertex_mass)
    rim = jnp.asarray(system.rim)
    eye2 = jnp.eye(2)

    def stretch(x):
        Ds = jnp.stack([x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]]], axis=-1)
        F = Ds @ dm_inv  # (T, 3, 2)
        E = 0.5 * (jnp.einsum("tki,tkj->tij", F, F) - eye2)
        tr = E[:, 0, 0] + E[:, 1, 1]
        psi = ph.stretch_mu * jnp.sum(E**2, axis=(1, 2)) + 0.5 * ph.stretch_lambda * tr**2
        return jnp.sum(area * psi)

    def bend(x):
        x0, x1, xa, xb = (x[hinges[:, k]] for k in range(4))
        e = x1 - x0
        n1 = jnp.cross(e, xa - x0)
        n2 = jnp.cross(xb - x0, e)
        e_hat = e / jnp.linalg.norm(e, axis=1, keepdims=True)
        sin = jnp.sum(jnp.cross(n1, n2) * e_hat, axis=1)
        cos = jnp.sum(n1 * n2, axis=1)
        theta = jnp.arctan2(sin, cos)  # scale-free: |n1||n2| cancels
        return ph.bend_stiffness * jnp.sum(hinge_w * theta**2)

    def gravity(x, c):
        return ph.gravity * (jnp.sum(vmass * x[:, 2]) + ph.ball_mass * c[2])

    def contact(x, c):
        dist = jnp.sqrt(jnp.sum((x - c) ** 2, axis=1) + 1e-12)
        gap = dist - (ph.ball_radius + ph.contact_thickness)
        floor_gap = c[2] - ph.ball_radius - ph.floor_height
        return penalty_energy(
            ineq_constraints=jnp.append(gap, floor_gap), w_ineq=ph.contact_stiffness
        )

    def terms(q):
        x, c = unpack(q, rim)
        return {
            "stretch": stretch(x),
            "bend": bend(x),
            "gravity": gravity(x, c),
            "contact": contact(x, c),
        }

    def energy_fn(q):
        return sum(terms(q).values())

    return energy_fn, terms


# -----------------------------------------------------------------------------
# Seed: static equilibrium of the ball resting on the pinned cloth
# -----------------------------------------------------------------------------


def solve_seed(system: ClothBallSystem, energy_fn, steps: int = 20_000, lr: float = 2e-3) -> np.ndarray:
    """Relax flat cloth + ball touching its centre to static equilibrium.

    Starts from the flat pinned cloth with the ball just touching it at the
    centre and minimises ``E_pot``, giving the neutral pinned configuration
    with the ball resting in the sag.

    Uses Adam with a decaying step size rather than L-BFGS: Adam moves each
    coordinate by at most about ``lr`` per step, so the ball can never jump
    past the cloth vertices in a single step. L-BFGS's first step can be of
    order 1 m, and it happily tunnels the ball through the cloth (the tunnelled
    state has lower energy, so the line search accepts it).
    """
    import optax

    ph = system.phys
    x0 = system.mesh.rest[: system.mesh.n_free]
    c0 = np.array([0.0, 0.0, ph.ball_radius + ph.contact_thickness])
    q0 = jnp.asarray(np.concatenate([x0.ravel(), c0]), dtype=jnp.float32)

    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=1e-3))
    grad = jax.grad(energy_fn)

    @jax.jit
    def run(q):
        def body(carry, _):
            q, state = carry
            updates, state = opt.update(grad(q), state)
            return (optax.apply_updates(q, updates), state), None

        (q, _), _ = jax.lax.scan(body, (q, opt.init(q)), None, length=steps)
        return q

    q = run(q0)
    g = np.asarray(grad(q))
    ball_force = np.linalg.norm(g[-3:])
    print(f"seed solve: {steps} Adam steps, E_pot {float(energy_fn(q)):.4e}, "
          f"residual force on ball {ball_force:.2e} N (weight {ph.ball_mass * ph.gravity:.2f} N), "
          f"max on a cloth vertex {np.abs(g[:-3]).max():.2e} N")
    return np.asarray(q)


# -----------------------------------------------------------------------------
# Rendering (shared with visualize_cloth_ball.py)
# -----------------------------------------------------------------------------


def sphere_mesh(center: np.ndarray, radius: float, n: int = 16) -> tuple[np.ndarray, ...]:
    """Grid points of a sphere surface for ``Axes3D.plot_surface``."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2 + 1))
    return (center[0] + radius * np.cos(u) * np.sin(v),
            center[1] + radius * np.sin(u) * np.sin(v),
            center[2] + radius * np.cos(v))


class ClothBallArtist:
    """Draws one configuration into a matplotlib 3D axes and can redraw it.

    The cloth is a ``Poly3DCollection`` whose vertices and colours are updated
    in place (colour = height). The ball surface is re-created on each update,
    which is cheap at this resolution.

    Matplotlib depth-sorts whole artists rather than individual triangles, so
    the ball is always drawn on top of the cloth. That is correct for the
    intended views (from above, with the ball resting on the cloth).
    """

    def __init__(self, ax, system: ClothBallSystem, z_limits=(-0.6, 0.3), color_limits=(-0.4, 0.0)):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt

        self.ax, self.system = ax, system
        self.rim = np.asarray(system.rim)
        self.tris = system.mesh.tris
        self.cmap = plt.get_cmap("viridis")
        self.color_limits = color_limits
        ax.computed_zorder = False
        self.cloth = Poly3DCollection([], edgecolor="none", linewidth=0.0, zorder=1)
        ax.add_collection3d(self.cloth)
        self.ball = None
        R = system.mesh.radius
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(*z_limits)
        ax.set_box_aspect((2 * R, 2 * R, z_limits[1] - z_limits[0]), zoom=1.4)
        ax.view_init(elev=22, azim=-60)
        ax.set_axis_off()

    def update(self, q: np.ndarray) -> None:
        x, c = (np.asarray(a) for a in unpack(jnp.asarray(q), jnp.asarray(self.rim)))
        tri_xyz = x[self.tris]
        lo, hi = self.color_limits
        self.cloth.set_verts(tri_xyz)
        self.cloth.set_facecolor(self.cmap(np.clip((tri_xyz[:, :, 2].mean(1) - lo) / (hi - lo), 0, 1)))
        if self.ball is not None:
            self.ball.remove()
        self.ball = self.ax.plot_surface(*sphere_mesh(c, self.system.phys.ball_radius),
                                         color="#d62728", linewidth=0, shade=True, zorder=2)


# -----------------------------------------------------------------------------
# Post-training sampling
# -----------------------------------------------------------------------------


def describe(q: np.ndarray, system: ClothBallSystem, terms) -> dict[str, float]:
    """Scalar diagnostics for one configuration."""
    x, c = (np.asarray(a) for a in unpack(jnp.asarray(q), jnp.asarray(system.rim)))
    dist = np.linalg.norm(x - c, axis=1) - system.phys.ball_radius
    t = {k: float(v) for k, v in terms(jnp.asarray(q)).items()}
    return {
        **t,
        "ball_x": c[0], "ball_y": c[1], "ball_z": c[2],
        "cloth_min_z": x[:, 2].min(),
        "penetration": max(0.0, -dist.min()),
    }


def evaluate(params, system: ClothBallSystem, energy_fn, terms, q_seed, out_dir: Path) -> None:
    """Sample the latent space and report what each axis does.

    * Energy statistics over 512 random ``z ~ N(0, I)``.
    * A sweep along each latent axis (others at 0) for z in [-2, 2],
      reporting where the ball goes and how deep the cloth sags.
    * A grid of rendered samples (rows: latent axis, columns: z value).
    """
    f = jax.jit(jax.vmap(lambda z: subspace_apply(params, z)))
    e = jax.jit(jax.vmap(energy_fn))

    zs = jax.random.normal(jax.random.PRNGKey(123), (512, LATENT_DIM))
    energies = np.asarray(e(f(zs)))
    seed_e = float(energy_fn(jnp.asarray(q_seed)))
    print("\n=== Latent sampling ===")
    print(f"seed E_pot                 : {seed_e:.4e}")
    print(f"E_pot over z ~ N(0, I)     : median {np.median(energies):.4e}   "
          f"95th pct {np.percentile(energies, 95):.4e}   max {energies.max():.4e}")
    print(f"  (relative to seed)       : median {np.median(energies) - seed_e:+.3e}   "
          f"95th pct {np.percentile(energies, 95) - seed_e:+.3e}")

    sweep = np.linspace(-2.0, 2.0, 5)
    grid = np.zeros((LATENT_DIM, len(sweep), CONFIG_DIM))
    for axis in range(LATENT_DIM):
        z = np.zeros((len(sweep), LATENT_DIM))
        z[:, axis] = sweep
        grid[axis] = np.asarray(f(jnp.asarray(z)))
        print(f"\nlatent axis {axis}:")
        print("     z    ball (x, y, z)              cloth min z   penetration   E_pot - seed")
        for zi, q in zip(sweep, grid[axis]):
            d = describe(q, system, terms)
            print(f"  {zi:+.1f}   ({d['ball_x']:+.3f}, {d['ball_y']:+.3f}, {d['ball_z']:+.3f})"
                  f"      {d['cloth_min_z']:+.3f}       {d['penetration']:.1e}"
                  f"      {float(energy_fn(jnp.asarray(q))) - seed_e:+.3e}")

    np.savez(out_dir / "samples.npz", sweep=sweep, grid=grid, random_z=np.asarray(zs),
             random_energy=energies, q_seed=q_seed)
    _plot_grid(grid, sweep, system, out_dir / "cloth_ball_latent_samples.png")


def _plot_grid(grid, sweep, system: ClothBallSystem, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    fig = plt.figure(figsize=(3.2 * len(sweep), 3.0 * LATENT_DIM))
    for axis in range(LATENT_DIM):
        for j, zi in enumerate(sweep):
            ax = fig.add_subplot(LATENT_DIM, len(sweep), axis * len(sweep) + j + 1, projection="3d")
            ClothBallArtist(ax, system).update(grid[axis, j])
            ax.set_title(f"z[{axis}] = {zi:+.1f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nsaved plot to {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--ramp-frac", type=float, default=0.5,
                        help="fraction of training over which p ramps 0 -> 1")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed-only", action="store_true",
                        help="solve and report the seed state, then stop")
    for name, default in PhysicalParams._field_defaults.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phys = PhysicalParams(**{k: getattr(args, k) for k in PhysicalParams._fields})
    system = build_system(phys)
    energy_fn, terms = make_energy_fn(system)
    M = mass_vector(system)

    print(f"Cloth ball: {system.mesh.n_free} free + {len(system.rim)} pinned cloth vertices, "
          f"{len(system.mesh.tris)} triangles, {len(system.hinges)} bending hinges")
    print(f"n={CONFIG_DIM}, d={LATENT_DIM}, cond={COND_DIM}, MLP {list(HIDDEN_DIMS)}, "
          f"sigma={SIGMA}, lambda={LAMBDA}")
    print(f"cloth mass {system.vertex_mass.sum():.3f} kg, ball mass {phys.ball_mass} kg, "
          f"rim radius {system.mesh.radius:.3f} m")

    t0 = time.time()
    q_seed = solve_seed(system, energy_fn)
    d = describe(q_seed, system, terms)
    print(f"seed ({time.time() - t0:.0f}s): ball centre z {d['ball_z']:+.3f}, "
          f"cloth sag {d['cloth_min_z']:+.3f} m, penetration {d['penetration']:.1e} m")
    print("seed energy terms: " + ", ".join(f"{k} {d[k]:+.3e}" for k in
                                            ("stretch", "bend", "gravity", "contact")))
    np.save(args.out_dir / "q_seed.npy", q_seed)
    # The viewer rebuilds the identical system from this file.
    (args.out_dir / "physical_params.json").write_text(json.dumps(phys._asdict(), indent=2))
    if args.seed_only:
        return

    subspace = SubspaceConfig(LATENT_DIM + COND_DIM, CONFIG_DIM, HIDDEN_DIMS)
    objective = ObjectiveConfig(sigma=SIGMA, lam=LAMBDA, batch_size=args.batch_size)
    loss_fn = make_loss_fn(energy_fn, jnp.asarray(q_seed, dtype=jnp.float32), LATENT_DIM,
                           objective, M=jnp.asarray(M, dtype=jnp.float32))

    params = train(loss_fn, subspace, args.steps, int(args.ramp_frac * args.steps),
                   args.lr, args.seed, args.log_every)
    save_params(params, args.out_dir / "params.npz")

    evaluate(params, system, energy_fn, terms, q_seed, args.out_dir)


if __name__ == "__main__":
    main()
