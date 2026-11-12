# WorldParticle: a jelly and a sand pile in a 2D box

A minimal, fully inspectable version of **WorldParticle** from Wang, Guo, Chen et al., *WorldParticle: Unified World
Simulation of Lagrangian Particle Dynamics via Transformer* (2026, [arXiv:2605.15305](https://arxiv.org/abs/2605.15305)),
in pure JAX like the rest of this repository.

The paper trains one transformer architecture on six kinds of physics, all written as particles. Here we use two of
them, in 2D with 64 particles each, so that every tensor fits on screen:

* a **jelly**: an 8×8 block joined by springs, thrown into a box (like the paper's elastic solids and cloth: it has a
  mesh **topology**, and its material attribute is the spring stiffness `k`);
* a **sand** column: 64 grains with friction and no springs, which collapses into a pile (like the paper's MPM sand:
  no topology, so those inputs are **zero-filled**, and the attribute is the friction `μ`).

**What is learned:** not the whole time step, as in [GNS](../../05-graph-neural-networks/gnn/), but only what a cheap
predictor gets wrong. The predictor applies gravity (`Ṽ = V + Δt g`, `X̃ = X + Δt/2 (V + Ṽ)`), and the transformer
predicts the residual `(ΔX, ΔV)` per particle.

## Start here

[`notebooks/worldparticle_explained.ipynb`](notebooks/worldparticle_explained.ipynb) walks through:

1. the ground truth: two small simulators (springs and walls; grains with Coulomb friction), 10 substeps per frame,
2. what goes in (positions, velocities, attributes, a 64×64 adjacency, rest positions, 84 wall samples) and what
   comes out (`ΔX`, `ΔV`), and why the sand's topology inputs are zeros,
3. the predictor alone, and how big the residual is that the network has to supply,
4. the architecture with every shape (tokenizer → 3 encoder layers 64 → 32 → 16 → 8 → 2 decoder layers → head),
5. one forward pass, tensor by tensor, with worked examples checked against the code: the **continuous kernel**
   (where a neighbour lands on the 4×4 lattice), the three **tokenizer branches**, the particle token, **bipartite
   token merging** (cosine similarities, greedy pairing, the averaging matrix), the **super tokens**, the decoder's
   **cross-attention**, the head,
6. the autoregressive **rollout loss** (window `W = 4`) and training (about 5 minutes per model on a laptop CPU),
7. 240-frame rollouts against the solver and against the predictor alone, with the error **frame by frame**,
8. what it learned: are the super tokens spatial regions, and are they **stable over time**?
9. generalization to unseen stiffness and to a 10×10 jelly (100 particles) with the same weights,
10. the paper's **local-only ablation**,
11. the sand: the same architecture with its own weights, and whether it uses the friction attribute,
12. an **interactive explorer** (category, test sequence, frame),
13. a table mapping each finding to the paper's claims and the reviewer talk.

The sliders need a running Jupyter kernel (press *Run Interact*). The saved outputs, including a static version of
the explorer, render on GitHub. The whole notebook runs in about 15 minutes on a laptop CPU.

## Results (saved run)

Position MSE is averaged over particles, frames and the 16 test sequences, rolled out from frame 0 for 239 steps
(the paper's Table 1 metric).

| | position MSE |
| --- | --- |
| jelly, WorldParticle | 2.6e-3 |
| jelly, predictor only (gravity, no correction) | 1.1 |
| jelly at frame 100 / frame 239 | 3.3e-4 / 1.5e-2 (RMS about 0.02 / 0.12; lattice spacing 0.05) |
| jelly, unseen stiffness k = 3000 / 15000 (trained on 4000 to 12000) | 7.0e-3 / 1.0e-2 (in range: 3.2e-3) |
| jelly, 100 particles with the 64-particle weights | 3.5e-3 |
| jelly, local-only model (no encoder, no attention; a third of the parameters) | 1.9e-3 |
| sand, WorldParticle | 4.8e-3 |
| sand spread, μ = 0.15 → 0.85 | truth 0.30 → 0.23, model 0.30 → 0.19 |
| super-token grouping unchanged from one frame to the next | about 0.82 after the first impact (two random groupings: 0.80) |

What the small version says about the paper:

* **The predictor–corrector split works.** The predictor alone falls through the floor; with the corrector the jelly
  holds together and bounces plausibly for 240 frames. Errors grow by orders of magnitude over a rollout, which a
  single averaged number hides.
* **"Unified" means the architecture.** The jelly and the sand run identical code, but each has its own weights.
* **The global stage is not always needed.** In this world every force is local, and the local-only model is as good
  as the full one. The paper's argument for super tokens is long-range fluid pressure.
* **Super tokens are feature groups, not patches.** Each holds 8 particles spread over the jelly, and the grouping
  reshuffles almost randomly from frame to frame after contact, while the rollout still works.

## Layout

```
worldparticle/
├── src/
│   ├── physics.py     ground truth: jelly (springs) and sand (Coulomb friction) in a box, walls, boundary samples
│   ├── model.py       predictor, tokenizer (3 continuous-kernel branches), super-token encoder with token merging,
│   │                  decoder with cross-attention, 2D RoPE; step_with_trace returns every tensor
│   ├── training.py    scenes and normalisation, rollout-window loss, AdamW warm-up + cosine, full-rollout evaluation
│   └── visualize.py   particles and springs, kernel lattice, merge levels, cross-attention, error curves, diagram
├── notebooks/
│   └── worldparticle_explained.ipynb
└── requirements.txt
```

## Run

```bash
cd 07-transformers/worldparticle
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/worldparticle_explained.ipynb
```

## Differences from the paper (all for size and readability)

| | WorldParticle paper | here |
| --- | --- | --- |
| dimension | 3D | 2D |
| categories | 6 (fluid, cloth, sand, non-Newtonian, elastic, protein) | 2 (jelly, sand) |
| particles | about 2k to 60k DOFs | 64 (100 in the generalization test) |
| ground truth | DFSPH, GIPC, MPM, VBD, OpenMM | our own spring and contact simulators |
| token width, heads | 1152, 12 | 32, 4 |
| encoder / decoder layers | 6 / 8 | 3 / 2 |
| kernel lattice | 4×4×4 | 4×4 |
| pair features of the kernels | not fully specified | chosen so spring and contact forces are linear in them |
| parameters | 156.7M | about 63k |
| rollout window | up to 5 | 4 |
| physics loss | optional SPH divergence (fluids) | none |
| training | 8 A100s, 400k steps, batch size 1 | laptop CPU, 2000 steps, batch size 8 |
