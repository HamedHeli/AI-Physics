"""Orbits on a rubber sheet: dynamics of the ball, simulated in the 3D latent space.

The heavy ball sags the pinned cloth into a well, like the classic rubber-sheet
picture of gravity. Give the ball a sideways kick and it circles the well like
a planet around a star. The whole 6069-DOF system (cloth + ball) is advanced
using only the 3 latent coordinates ``z`` of the trained subspace
``q = f_theta(z)``.

Integrator
----------
A variational (discrete Euler-Lagrange) integrator restricted to the subspace.
With the discrete Lagrangian

    L_d(z_k, z_k+1) = h [ 1/2 |(f(z_k+1) - f(z_k)) / h|_M^2
                          - (E(f(z_k)) + E(f(z_k+1))) / 2 ]

the update ``D2 L_d(z_k-1, z_k) + D1 L_d(z_k, z_k+1) = 0`` reads

    J_k^T M (f(z_k+1) - f(z_k)) = (1 - gamma h) J_k^T M (f(z_k) - f(z_k-1)) - h^2 grad_z E(f(z_k))

with ``J_k = df/dz`` at ``z_k``: 3 nonlinear equations for ``z_k+1``, solved by
Newton's method (3x3 systems). Unlike the implicit-Euler step of the paper
(Sec. 3.4), which bleeds energy, this scheme is symplectic, so the total energy
stays bounded and an orbit keeps going. ``gamma`` adds optional damping, which
makes the ball spiral into the bottom of the well.

Initial conditions
------------------
* Position: the lowest-energy subspace state with the ball centre at the
  requested ``(x, y)``.
* Velocity: the requested ball velocity, realised with the least kinetic
  energy: ``zdot = Mz^-1 Jb^T (Jb Mz^-1 Jb^T)^-1 v`` where ``Mz = J^T M J`` and
  ``Jb`` is the ball-(x, y) block of ``J``.
* By default the kick is tangential with the circular-orbit speed
  ``v_c = sqrt(r a_r)``, where ``a_r`` is the inward acceleration of the ball
  released from rest at the start; ``--orbit`` scales it (1 ~ circle,
  < 1 = ellipse dipping towards the centre, > 1 = ellipse swinging outwards).

Usage
-----
    python experiments/simulate_cloth_ball.py                         # precessing orbit (default)
    python experiments/simulate_cloth_ball.py --orbit 1.0 --start 0.2 0
    python experiments/simulate_cloth_ball.py --orbit 0.6 --save ellipse.gif
    python experiments/simulate_cloth_ball.py --damping 0.4 --save spiral.gif

Run ``experiments/cloth_ball.py`` first. Relative ``--save`` paths go to
``experiments/outputs/cloth_ball/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # tiny per-step displacements; float32 drifts

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR.parent))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import cloth_ball as cb  # noqa: E402
from src.model import subspace_apply  # noqa: E402
from src.training import load_params  # noqa: E402
from visualize_cloth_ball import load_system  # noqa: E402

OUTPUT_DIR = cb.OUTPUT_DIR
BALL = slice(-3, None)  # ball centre inside q
BALL_XY = slice(-3, -1)


# -----------------------------------------------------------------------------
# Reduced dynamics
# -----------------------------------------------------------------------------


class ReducedSystem:
    """Jitted pieces of the subspace mechanics: ``f``, ``J``, ``E(f(z))`` and the step."""

    def __init__(self, params, system: cb.ClothBallSystem):
        params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), params)
        energy_fn, _ = cb.make_energy_fn(system)
        M = jnp.asarray(cb.mass_vector(system))

        f = lambda z: subspace_apply(params, z)  # noqa: E731
        self.M = M
        self.f = jax.jit(f)
        self.jac = jax.jit(jax.jacfwd(f))  # (n, d)
        self.energy = jax.jit(lambda z: energy_fn(f(z)))
        self.grad_energy = jax.jit(jax.grad(lambda z: energy_fn(f(z))))

        def step(z_prev, z, h, damping):
            """One discrete Euler-Lagrange step ``(z_k-1, z_k) -> z_k+1``."""
            fz, J = f(z), jax.jacfwd(f)(z)
            JtM = J.T * M
            rhs = (1.0 - damping * h) * JtM @ (fz - f(z_prev)) - h**2 * jax.grad(
                lambda w: energy_fn(f(w)))(z)

            def newton(_, w):
                r = JtM @ (f(w) - fz) - rhs
                return w - jnp.linalg.solve(JtM @ jax.jacfwd(f)(w), r)

            return jax.lax.fori_loop(0, 4, newton, 2 * z - z_prev)

        self.step = jax.jit(step)

    def reduced_mass(self, z) -> np.ndarray:
        J = np.asarray(self.jac(z))
        return J.T @ (np.asarray(self.M)[:, None] * J)

    def ball_xy(self, z) -> np.ndarray:
        return np.asarray(self.f(z))[BALL_XY]


def rest_state_at(rs: ReducedSystem, target_xy, weight: float = 1e4) -> np.ndarray:
    """Lowest-energy latent ``z`` whose ball centre sits at ``target_xy``."""
    target = jnp.asarray(target_xy, jnp.float64)
    obj = jax.jit(jax.value_and_grad(
        lambda z: rs.energy(z) + weight * jnp.sum((rs.f(z)[BALL_XY] - target) ** 2)))
    res = minimize(lambda z: tuple(np.asarray(a) for a in obj(z)),
                   np.zeros(cb.LATENT_DIM), jac=True, method="L-BFGS-B")
    return res.x


def kick(rs: ReducedSystem, z0: np.ndarray, v_xy: np.ndarray) -> np.ndarray:
    """Latent velocity giving the ball velocity ``v_xy`` with least kinetic energy."""
    Mz_inv = np.linalg.inv(rs.reduced_mass(z0))
    Jb = np.asarray(rs.jac(z0))[BALL_XY]
    return Mz_inv @ Jb.T @ np.linalg.solve(Jb @ Mz_inv @ Jb.T, v_xy)


def circular_speed(rs: ReducedSystem, z0: np.ndarray) -> float:
    """``sqrt(r a_r)`` for the ball released from rest at ``z0``."""
    zdd = -np.linalg.solve(rs.reduced_mass(z0), np.asarray(rs.grad_energy(z0)))
    a = np.asarray(rs.jac(z0))[BALL_XY] @ zdd
    xy = rs.ball_xy(z0)
    r = np.linalg.norm(xy)
    a_r = -a @ xy / r
    if a_r <= 0:
        sys.exit(f"ball at r={r:.3f} m is not pulled towards the centre (a_r={a_r:.3g}); "
                 "pick a different --start")
    return float(np.sqrt(r * a_r))


def simulate(rs: ReducedSystem, z0, zdot0, h: float, n_steps: int, damping: float) -> np.ndarray:
    """Latent trajectory ``(n_steps + 1, d)``."""
    zs = np.zeros((n_steps + 1, cb.LATENT_DIM))
    zs[0] = z0
    z_prev, z = jnp.asarray(z0 - h * zdot0), jnp.asarray(z0)
    for k in range(n_steps):
        z_prev, z = z, rs.step(z_prev, z, h, damping)
        zs[k + 1] = np.asarray(z)
    return zs


def energies(rs: ReducedSystem, zs: np.ndarray, h: float):
    """Kinetic (central difference, full space), potential, and their sum per step."""
    qs = np.asarray(jax.jit(jax.vmap(rs.f))(jnp.asarray(zs)))
    pe = np.asarray(jax.jit(jax.vmap(rs.energy))(jnp.asarray(zs)))
    v = np.gradient(qs, h, axis=0)
    ke = 0.5 * np.sum(np.asarray(rs.M) * v**2, axis=1)
    return qs, ke, pe


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def render(qs, ke, pe, zs, h, system, path: Path, frame_every: int, fps: int, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    t = np.arange(len(qs)) * h
    ball = qs[:, BALL]
    pe_rel = pe - pe.min()
    total = ke + pe_rel

    fig = plt.figure(figsize=(12, 6.4))
    ax3d = fig.add_axes([-0.04, 0.0, 0.66, 0.92], projection="3d")
    artist = cb.ClothBallArtist(ax3d, system)
    ax3d.view_init(elev=38, azim=-60)
    trail3d, = ax3d.plot([], [], [], color="white", lw=1.6, alpha=0.9, zorder=3)
    fig.text(0.02, 0.97, title, fontsize=13, weight="bold", va="top")
    info = fig.text(0.02, 0.91, "", family="monospace", fontsize=9, va="top")

    R = system.mesh.radius
    ax_top = fig.add_axes([0.66, 0.47, 0.31, 0.45])
    ax_top.add_patch(plt.Circle((0, 0), R, fill=False, color="0.6", lw=1, ls="--"))
    ax_top.plot(ball[:, 0], ball[:, 1], color="0.85", lw=1)
    trail_top, = ax_top.plot([], [], color="#d62728", lw=1.5)
    dot_top, = ax_top.plot([], [], "o", color="#d62728", ms=8)
    ax_top.plot(0, 0, "+", color="0.4", ms=10)
    lim = max(0.35, 1.25 * np.abs(ball[:, :2]).max())
    ax_top.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
               title="ball path, seen from above", xlabel="x (m)", ylabel="y (m)")
    ax_top.title.set_fontsize(10)

    ax_e = fig.add_axes([0.66, 0.08, 0.31, 0.28])
    ax_e.plot(t, ke, color="#1f77b4", lw=1.2, label="kinetic")
    ax_e.plot(t, pe_rel, color="#2ca02c", lw=1.2, label="potential")
    ax_e.plot(t, total, color="black", lw=1.6, label="total")
    cursor = ax_e.axvline(0, color="0.5", lw=1)
    ax_e.set(xlim=(0, t[-1]), ylim=(0, 1.45 * max(total.max(), 1e-6)), xlabel="time (s)", ylabel="J")
    ax_e.set_title("energy (potential relative to its minimum)", fontsize=10)
    ax_e.legend(fontsize=8, loc="upper right", ncols=3)

    def update(k):
        artist.update(qs[k])
        trail3d.set_data_3d(ball[: k + 1, 0], ball[: k + 1, 1], ball[: k + 1, 2])
        trail_top.set_data(ball[: k + 1, 0], ball[: k + 1, 1])
        dot_top.set_data([ball[k, 0]], [ball[k, 1]])
        cursor.set_xdata([t[k], t[k]])
        info.set_text(f"t = {t[k]:5.2f} s     z = ({', '.join(f'{v:+.2f}' for v in zs[k])})\n"
                      f"{cb.CONFIG_DIM} DOFs driven by {cb.LATENT_DIM} latent numbers")

    frames = range(0, len(qs), frame_every)
    anim = FuncAnimation(fig, update, frames=frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, fps=fps, dpi=80)
    print(f"saved {path} ({len(frames)} frames)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start", type=float, nargs=2, default=[0.25, 0.0],
                        help="initial ball centre (x, y) in metres")
    parser.add_argument("--orbit", type=float, default=1.2,
                        help="tangential kick as a multiple of the circular-orbit speed")
    parser.add_argument("--velocity", type=float, nargs=2, default=None,
                        help="explicit initial ball velocity (vx, vy) in m/s; overrides --orbit")
    parser.add_argument("--damping", type=float, default=0.0, help="gamma (1/s)")
    parser.add_argument("--duration", type=float, default=6.0, help="seconds")
    parser.add_argument("--dt", type=float, default=1 / 240)
    parser.add_argument("--frame-every", type=int, default=6, help="simulation steps per frame")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--save", type=Path, default=Path("cloth_ball_orbit.gif"))
    args = parser.parse_args()
    if not args.save.is_absolute():
        args.save = OUTPUT_DIR / args.save

    params_path = args.run_dir / "params.npz"
    if not params_path.exists():
        sys.exit(f"{params_path} not found; run experiments/cloth_ball.py first")
    system = load_system(args.run_dir)
    rs = ReducedSystem(load_params(params_path), system)

    z0 = rest_state_at(rs, args.start)
    xy0 = rs.ball_xy(z0)
    if args.velocity is not None:
        v = np.asarray(args.velocity)
        title = "Ball kicked across a rubber sheet"
    else:
        tangent = np.array([-xy0[1], xy0[0]]) / np.linalg.norm(xy0)
        v_c = circular_speed(rs, z0)
        v = args.orbit * v_c * tangent
        print(f"circular-orbit speed at r={np.linalg.norm(xy0):.3f} m: {v_c:.3f} m/s")
        title = "Orbiting a rubber-sheet 'star'" + (" (damped)" if args.damping > 0 else "")
    print(f"start: z0={np.round(z0, 3)}, ball xy={np.round(xy0, 3)}, kick v={np.round(v, 3)} m/s")

    n_steps = int(round(args.duration / args.dt))
    t0 = time.time()
    zs = simulate(rs, z0, kick(rs, z0, v), args.dt, n_steps, args.damping)
    print(f"simulated {n_steps} steps in {time.time() - t0:.1f}s "
          f"({n_steps / (time.time() - t0):.0f} steps/s); max |z| = {np.abs(zs).max():.2f}")

    qs, ke, pe = energies(rs, zs, args.dt)
    tot = ke + pe
    r = np.linalg.norm(qs[:, BALL_XY], axis=1)
    print(f"ball radius from centre: min {r.min():.3f}  max {r.max():.3f} m")
    print(f"total energy: start {tot[1]:+.5f} J, end {tot[-2]:+.5f} J, "
          f"max swing {np.ptp(tot[1:-1]):.2e} J (kinetic peak {ke.max():.4f} J)")

    render(qs, ke, pe, zs, args.dt, system, args.save, args.frame_every, args.fps, title)


if __name__ == "__main__":
    main()
