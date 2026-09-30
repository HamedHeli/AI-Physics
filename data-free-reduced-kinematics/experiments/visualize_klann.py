"""Interactive viewer for the trained Klann linkage neural subspace.

Drag the slider (or type a value) to set the 1D latent coordinate ``z``. The
viewer evaluates the trained MLP ``f_theta(z)``, splits the 84-vector into
seven 3x4 rigid transforms ``[R | t]``, and redraws every body by applying
its transform to the body's rest geometry.

Run ``experiments/klann_linkage.py`` first to produce ``params.npz``.

Usage
-----
    python experiments/visualize_klann.py                      # interactive
    python experiments/visualize_klann.py --z-range 3.0        # wider slider
    python experiments/visualize_klann.py --save klann.gif     # headless GIF sweep
    python experiments/visualize_klann.py --save klann.png --z 0.5

With no display (e.g. SSH, codespaces), use ``--save``: ``.gif`` / ``.mp4``
sweeps z across the slider range, and any other extension saves a still
image at ``--z``. Files are written to ``experiments/outputs/klann_linkage/``
unless ``--save`` is given an absolute path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR.parent))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import klann_linkage as klann  # noqa: E402
from src.model import MLPParams, subspace_apply  # noqa: E402
from src.training import load_params  # noqa: E402

OUTPUT_DIR = EXPERIMENTS_DIR / "outputs" / "klann_linkage"
DEFAULT_PARAMS = OUTPUT_DIR / "params.npz"

# Colour per body, in klann.BODY_NAMES order.
BODY_COLORS = {
    "anchor": "#7f7f7f",
    "frame": "#404040",
    "crank": "#d62728",
    "conn_arm": "#1f77b4",
    "lower_rocker": "#2ca02c",
    "upper_rocker": "#9467bd",
    "leg": "#ff7f0e",
}


# -----------------------------------------------------------------------------
# Model evaluation
# -----------------------------------------------------------------------------


def make_evaluator(params: MLPParams):
    """Return ``z (float) -> q (84,)``, jit-compiled once up front."""
    f = jax.jit(lambda z: subspace_apply(params, z))

    def evaluate(z: float) -> np.ndarray:
        q = f(jnp.array([z], dtype=jnp.float32))
        return np.asarray(q)

    evaluate(0.0)  # compile now so the first slider move is instant
    return evaluate


# -----------------------------------------------------------------------------
# Geometric decoding
# -----------------------------------------------------------------------------


def decode_transforms(q: np.ndarray) -> np.ndarray:
    """Reshape the 84-vector into seven 3x4 transforms ``[R | t]``.

    Per body the configuration stores ``[t (3), R row-major (9)]``
    (see ``klann_linkage.unpack``). ``R`` is the raw learned matrix and is
    not re-orthonormalised, so any residual non-rigidity is shown as is.

    Returns:
        Array of shape ``(7, 3, 4)`` with body ``i``'s world point
        ``x = T[i] @ [p, 1]`` for body-local point ``p``.
    """
    t, R = klann.unpack(q)
    return np.concatenate([np.asarray(R), np.asarray(t)[..., None]], axis=-1)


def base_geometry(bodies: list[klann.RigidBody]) -> list[np.ndarray]:
    """Rest-pose outline of each body in its local frame (relative to the COM).

    Each entry is an ``(k, 3)`` polyline; triangles are closed. The anchor is
    drawn as a small square plate.
    """
    outlines = []
    for i, name in enumerate(klann.BODY_NAMES):
        if name == "anchor":
            s = klann.ANCHOR_HALF_SIZE
            pts = np.array([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0), (-s, -s, 0)])
            pts = pts + klann._p3("O")
        else:
            names = klann.BODY_OUTLINES[name]
            pts = np.array([klann._p3(n) for n in names])
            if len(pts) > 2:
                pts = np.vstack([pts, pts[:1]])
        outlines.append(pts - bodies[i].com)
    return outlines


def apply_transforms(T: np.ndarray, outlines: list[np.ndarray]) -> list[np.ndarray]:
    """World-space 2D polylines: ``x = R p + t`` for every outline point."""
    return [(outline @ Ti[:, :3].T + Ti[:, 3])[:, :2] for Ti, outline in zip(T, outlines)]


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------


class KlannViewer:
    """Matplotlib figure with a latent slider and a typed-value box."""

    def __init__(self, evaluate, bodies, energy_fn, z_range: float, z0: float, interactive: bool):
        import matplotlib.pyplot as plt

        self.evaluate = evaluate
        self.energy_fn = jax.jit(energy_fn)
        self.outlines = base_geometry(bodies)
        self.foot_local = klann._p3("E") - bodies[klann.BODY_NAMES.index("leg")].com
        self.z_range = z_range

        self.fig, self.ax = plt.subplots(figsize=(7, 7.5))
        self.fig.subplots_adjust(bottom=0.18)
        ax = self.ax
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 0.6)
        ax.set_ylim(-1.6, 0.8)
        ax.grid(alpha=0.2)

        # Static context: the analytic foot path, and the foot trace over the slider range.
        ref = klann.KlannReference()
        ref_foot = np.array([ref.solve(a)["E"] for a in np.linspace(0, 2 * np.pi, 361)])
        ax.plot(*ref_foot.T, "k--", lw=0.8, alpha=0.5, label="analytic foot path")
        trace = np.array([self._foot(self.evaluate(z)) for z in np.linspace(-z_range, z_range, 121)])
        ax.plot(*trace.T, color="#ff7f0e", lw=1, alpha=0.4, label=f"learned foot, |z| <= {z_range:g}")

        self.lines = [
            ax.plot([], [], "-o", color=BODY_COLORS[n], lw=3, ms=4, label=n)[0]
            for n in klann.BODY_NAMES
        ]
        (self.foot_dot,) = ax.plot([], [], "o", color="k", ms=7, zorder=5)
        self.info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace")
        ax.legend(loc="upper right", fontsize=7, ncol=2)

        self.slider = self.textbox = None
        self._syncing = False  # set while update() writes the text box
        if interactive:
            from matplotlib.widgets import Slider, TextBox

            self.slider = Slider(
                self.fig.add_axes([0.15, 0.07, 0.55, 0.03]),
                "latent z", -z_range, z_range, valinit=z0,
            )
            self.slider.on_changed(self.update)
            self.textbox = TextBox(self.fig.add_axes([0.80, 0.065, 0.12, 0.04]), "z = ", initial=f"{z0:g}")
            self.textbox.on_submit(self._on_text)

        self.update(z0)

    def _foot(self, q: np.ndarray) -> np.ndarray:
        T = decode_transforms(q)[klann.BODY_NAMES.index("leg")]
        return (T[:, :3] @ self.foot_local + T[:, 3])[:2]

    def _on_text(self, text: str) -> None:
        if self._syncing:  # TextBox.set_val fires submit; ignore our own echo
            return
        try:
            z = float(text)
        except ValueError:
            return
        # Setting the slider triggers update(); values outside its range are clamped.
        self.slider.set_val(float(np.clip(z, -self.z_range, self.z_range)))

    def update(self, z: float) -> None:
        """Evaluate f(z), decode the transforms and redraw every body."""
        q = self.evaluate(float(z))
        T = decode_transforms(q)
        for line, poly in zip(self.lines, apply_transforms(T, self.outlines)):
            line.set_data(poly[:, 0], poly[:, 1])
        self.foot_dot.set_data(*[[c] for c in self._foot(q)])

        R = T[:, :, :3]
        crank = klann.BODY_NAMES.index("crank")
        angle = np.degrees(np.arctan2(R[crank, 1, 0], R[crank, 0, 0]))
        ortho = np.abs(np.einsum("bki,bkj->bij", R, R) - np.eye(3)).max()
        energy = float(self.energy_fn(jnp.asarray(q)))
        self.info.set_text(
            f"z          = {z:+.3f}\n"
            f"crank      = {angle:+7.1f} deg\n"
            f"E_pot      = {energy:.2e}\n"
            f"|R^T R - I| = {ortho:.1e}"
        )
        if self.textbox is not None:
            self._syncing = True
            self.textbox.set_val(f"{z:.3g}")
            self._syncing = False
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Klann subspace viewer")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--z-range", type=float, default=2.0,
                        help="slider spans [-z_range, z_range]; ~95%% of N(0,1) at 2.0")
    parser.add_argument("--z", type=float, default=0.0, help="initial latent value")
    parser.add_argument("--w-eq", type=float, default=1e4,
                        help="penalty stiffness for the E_pot readout (match training)")
    parser.add_argument("--save", type=Path, default=None,
                        help="render to file instead of opening a window (.gif/.mp4 = sweep); "
                             "relative paths are placed in experiments/outputs/klann_linkage/")
    args = parser.parse_args()
    if args.save is not None and not args.save.is_absolute():
        args.save = OUTPUT_DIR / args.save

    if not args.params.exists():
        sys.exit(f"{args.params} not found; run experiments/klann_linkage.py first")

    import matplotlib

    if args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bodies = klann.build_bodies()
    energy_fn, _ = klann.make_energy_fn(klann.build_constraints(bodies), bodies, args.w_eq)
    evaluate = make_evaluator(load_params(args.params))

    interactive = args.save is None
    viewer = KlannViewer(evaluate, bodies, energy_fn, args.z_range, args.z, interactive)

    if interactive:
        plt.show()
        return

    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.save.suffix.lower() in (".gif", ".mp4"):
        from matplotlib.animation import FuncAnimation

        # There and back, so the loop is seamless.
        zs = np.concatenate([np.linspace(-args.z_range, args.z_range, 60),
                             np.linspace(args.z_range, -args.z_range, 60)])
        anim = FuncAnimation(viewer.fig, lambda z: viewer.update(z), frames=zs, interval=40)
        anim.save(args.save, fps=25)
    else:
        viewer.fig.savefig(args.save, dpi=120)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
