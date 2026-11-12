# 07 Transformers

**Week 8 (Oct 29), Unit II: learning all the way** · [back to the graph](../README.md)

**What is learned:** a general-purpose surrogate for many PDEs at once, trained like a foundation model. The
network still maps input fields to output fields, but one backbone covers many equation families. WorldParticle
does the same on particles instead of grids, and learns only the *residual* of a cheap physics step.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Holzschuh+ 2025] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations.** Benjamin Holzschuh, Qiang Liu, Georg Kohl, Nils Thuerey | ICML 2025 | [arXiv:2505.24717](https://arxiv.org/abs/2505.24717) · [PMLR](https://proceedings.mlr.press/v267/holzschuh25a.html) |
| R2 | **[Wang+ 2026] WorldParticle: Unified World Simulation of Lagrangian Particle Dynamics via Transformer.** Caoliwen Wang, Minghao Guo, Siyuan Chen, et al. (corresponding: Peter Yichen Chen) | preprint | [arXiv:2605.15305](https://arxiv.org/abs/2605.15305) |

Week 8 also has the first project check-in. WorldParticle was assigned on the course Slack (it is not on the schedule
sheet); Hamed is its Reviewer.

## Key ideas

- **Diffusion-transformer backbone for physics.** Fields are patchified into tokens on regular grids and processed
  by a multi-scale transformer with shifted-window attention, so cost stays manageable at high resolution.
- **Mixed-channel vs separate-channel embeddings.** Embedding each physical channel separately lets one model handle
  PDEs with different numbers and kinds of variables.
- **Pre-train, then fine-tune.** Trained on a large mix of PDE datasets, it transfers to new downstream PDEs better
  than training from scratch.
- **WorldParticle: learn only the correction to a physics step.** A predictor integrates the known forces (gravity),
  and a transformer predicts the residual `(ΔX, ΔV)` per particle. One architecture covers six kinds of physics
  (fluid, cloth, sand, non-Newtonian flow, elastic solids, proteins), with one set of weights per category.
- **Local tokens, global super tokens.** A tokenizer turns each particle into a token with learned continuous kernels
  over three neighbourhoods (particles, walls, mesh edges). An encoder alternates self-attention with bipartite
  token merging (ToMe), halving the tokens each layer into a few *super tokens*; in the decoder every particle
  cross-attends to them. Inputs a category lacks (a mesh for sand, walls for proteins) are zero-filled.

## Code in this folder

- [`worldparticle/`](worldparticle/): a minimal WorldParticle in pure JAX on a 2D jelly (springs, topology branch
  used) and a 2D sand column (friction, topology zero-filled), with a notebook that shows the kernels, the token
  merging 64 → 8 and the cross-attention tensor by tensor, rollout training, 240-frame rollouts against the solver
  and the predictor alone, generalization to unseen stiffness and 100 particles, and a local-only ablation.

## Connections

- ← [06 Neural operators](../06-neural-operators/): Transolver is attention on general meshes; PDE-Transformer is
  attention on grids at scale. [`transolver/`](../06-neural-operators/transolver/) walks through Physics-Attention
  tensor by tensor on an irregular point cloud.
- ← [05 Graph neural networks](../05-graph-neural-networks/): WorldParticle is a learned particle simulator like
  GNS, but it predicts a correction on top of a physics predictor and gets global context from super tokens in one
  step instead of many message-passing hops.
- ↔ [09 One PDE, many ways](../09-one-pde-many-ways/): §8 uses Transolver's Physics-Attention on Burgers, next to an
  FNO, next to every other method of the course.
