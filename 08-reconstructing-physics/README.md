# 08 Reconstructing physics

**Week 11 (Nov 19), Unit III: generative AI** · [back to the graph](../README.md)

**What is learned:** the physics behind observations. Geometry and appearance come from a neural radiance field, and
the unknown material parameters are recovered by fitting a differentiable simulator to multi-view video.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Li+ 2023] PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification.** Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, Chuang Gan | ICLR 2023 | [arXiv:2303.05512](https://arxiv.org/abs/2303.05512) |

Week 10 opens Unit III with AlphaFold 3, which has its own folder: [10 Generative structure prediction](../10-generative-structure-prediction/).

## Key ideas

- **Eulerian-Lagrangian NeRF.** The radiance field is stored on a grid for rendering, and its density is carried by
  MPM particles so it advects with the simulated material.
- **Geometry-agnostic.** No mesh or template is needed: the first frame's NeRF provides the initial shape.
- **System identification by gradient descent.** A differentiable MPM simulator (elastic, plastic, granular,
  Newtonian and non-Newtonian fluids) is rolled out, rendered, and compared with the video; gradients update
  Young's modulus, viscosity, friction angle and similar parameters.

## Connections

- ← [01 Differentiable simulation](../01-differentiable-simulation/): the differentiable MPM is the DiffTaichi idea
  applied to an inverse problem from images.
