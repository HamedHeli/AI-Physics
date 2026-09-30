"""Interactive viewer for the trained cloth-ball neural subspace.

Three sliders (one per latent coordinate) set ``z = (z0, z1, z2)``. The viewer
evaluates the trained MLP ``f_theta(z)``, splits the 6069-vector into the
cloth vertex positions (2022 free + the fixed pinned rim) and the ball
centre, and redraws the cloth surface and the ball.

Run ``experiments/cloth_ball.py`` first to produce ``params.npz`` (and
``physical_params.json``, used to rebuild the identical mesh and materials).

Usage
-----
    python experiments/visualize_cloth_ball.py                        # interactive
    python experiments/visualize_cloth_ball.py --save cloth_ball.gif  # sweep each axis in turn
    python experiments/visualize_cloth_ball.py --save still.png --z 1.0 -0.5 0.0

With no display (e.g. SSH, codespaces), use ``--save``. Relative paths are
written to ``experiments/outputs/cloth_ball/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR.parent))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import cloth_ball as cb  # noqa: E402
from src.model import subspace_apply  # noqa: E402
from src.training import load_params  # noqa: E402

OUTPUT_DIR = cb.OUTPUT_DIR


# -----------------------------------------------------------------------------
# Model evaluation and decoding
# -----------------------------------------------------------------------------


def make_evaluator(params):
    """Return ``z (3,) -> q (6069,)``, jit-compiled once up front."""
    f = jax.jit(lambda z: subspace_apply(params, z))

    def evaluate(z) -> np.ndarray:
        return np.asarray(f(jnp.asarray(z, dtype=jnp.float32)))

    evaluate(np.zeros(cb.LATENT_DIM))
    return evaluate


def load_system(out_dir: Path) -> cb.ClothBallSystem:
    """Rebuild the training system, using saved physical parameters if present."""
    path = out_dir / "physical_params.json"
    if path.exists():
        phys = cb.PhysicalParams(**json.loads(path.read_text()))
    else:
        print(f"{path} not found; assuming default physical parameters")
        phys = cb.PhysicalParams()
    return cb.build_system(phys)


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------


class ClothBallViewer:
    """3D view with one slider per latent coordinate and a live readout."""

    def __init__(self, evaluate, system, z_range: float, z0, interactive: bool):
        import matplotlib.pyplot as plt

        self.evaluate = evaluate
        self.system = system
        energy_fn, terms = cb.make_energy_fn(system)
        self.terms = jax.jit(terms)
        self.z = np.array(z0, dtype=float)

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_axes([0.0, 0.2, 1.0, 0.8], projection="3d")
        self.artist = cb.ClothBallArtist(self.ax, system)
        self.info = self.fig.text(0.02, 0.97, "", va="top", family="monospace", fontsize=9)

        self.sliders = []
        if interactive:
            from matplotlib.widgets import Slider

            for k in range(cb.LATENT_DIM):
                s = Slider(self.fig.add_axes([0.15, 0.13 - 0.045 * k, 0.7, 0.03]),
                           f"z[{k}]", -z_range, z_range, valinit=self.z[k])
                s.on_changed(lambda v, k=k: self.set_component(k, v))
                self.sliders.append(s)

        self.update()

    def set_component(self, k: int, value: float) -> None:
        self.z[k] = value
        self.update()

    def update(self, z=None) -> None:
        """Evaluate f(z), redraw cloth and ball, refresh the readout."""
        if z is not None:
            self.z = np.asarray(z, dtype=float)
        q = self.evaluate(self.z)
        self.artist.update(q)

        t = {k: float(v) for k, v in self.terms(jnp.asarray(q)).items()}
        x, c = (np.asarray(a) for a in cb.unpack(jnp.asarray(q), jnp.asarray(self.system.rim)))
        pen = max(0.0, -(np.linalg.norm(x - c, axis=1) - self.system.phys.ball_radius).min())
        self.info.set_text(
            f"z = ({', '.join(f'{v:+.2f}' for v in self.z)})\n"
            f"ball = ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) m\n"
            f"cloth lowest point = {x[:, 2].min():+.3f} m   penetration = {pen:.1e} m\n"
            f"E_pot = {sum(t.values()):+.4f}  (stretch {t['stretch']:.3f}, bend {t['bend']:.4f}, "
            f"gravity {t['gravity']:+.3f}, contact {t['contact']:.1e})"
        )
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive cloth-ball subspace viewer")
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_DIR,
                        help="training output folder (params.npz, physical_params.json)")
    parser.add_argument("--z-range", type=float, default=2.0,
                        help="each slider spans [-z_range, z_range]")
    parser.add_argument("--z", type=float, nargs=cb.LATENT_DIM, default=[0.0] * cb.LATENT_DIM,
                        help="initial latent vector")
    parser.add_argument("--save", type=Path, default=None,
                        help="render to file instead of opening a window "
                             "(.gif/.mp4 = sweep each axis); relative paths go to "
                             "experiments/outputs/cloth_ball/")
    args = parser.parse_args()
    if args.save is not None and not args.save.is_absolute():
        args.save = OUTPUT_DIR / args.save

    params_path = args.run_dir / "params.npz"
    if not params_path.exists():
        sys.exit(f"{params_path} not found; run experiments/cloth_ball.py first")

    import matplotlib

    if args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    system = load_system(args.run_dir)
    evaluate = make_evaluator(load_params(params_path))
    interactive = args.save is None
    viewer = ClothBallViewer(evaluate, system, args.z_range, args.z, interactive)

    if interactive:
        plt.show()
        return

    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.save.suffix.lower() in (".gif", ".mp4"):
        from matplotlib.animation import FuncAnimation

        # For each axis in turn: 0 -> +r -> -r -> 0, other coordinates at 0.
        r = args.z_range
        path = np.concatenate([np.linspace(0, r, 15), np.linspace(r, -r, 30), np.linspace(-r, 0, 15)])
        frames = []
        for k in range(cb.LATENT_DIM):
            for v in path:
                z = np.zeros(cb.LATENT_DIM)
                z[k] = v
                frames.append(z)
        anim = FuncAnimation(viewer.fig, viewer.update, frames=frames, interval=50)
        anim.save(args.save, fps=20)
    else:
        viewer.fig.savefig(args.save, dpi=110)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
