# Data-Free Learning of Reduced-Order Kinematics

**Authors:** NICHOLAS SHARP, CRISTIAN ROMERO, ALEC JACOBSON, ETIENNE VOUGA, PAUL G. KRY, DAVID I.W. LEVIN, JUSTIN SOLOMON

## Abstract
Physical systems ranging from elastic bodies to kinematic linkages are defined on high-dimensional configuration spaces, yet their typical low-energy configurations are concentrated on much lower-dimensional subspaces. This work addresses the challenge of identifying such subspaces automatically. Given as input an energy function for a high-dimensional system, we produce a low-dimensional map whose image parameterizes a diverse yet low-energy submanifold of configurations. The only additional input needed is a single seed configuration for the system to initialize our procedure; no dataset of trajectories is required. We represent subspaces as neural networks that map a low-dimensional latent vector to the full configuration space, and propose a training scheme to fit network parameters to any system of interest. This formulation is effective across a very general range of physical systems; our experiments demonstrate not only nonlinear and very low-dimensional elastic body and cloth subspaces, but also more general systems like colliding rigid bodies and linkages. We briefly explore applications built on this formulation, including manipulation, latent interpolation, and sampling.

## 1 INTRODUCTION
Physical simulation algorithms perennially achieve new heights of detail and fidelity. Modern computer graphics techniques successfully capture phenomena from elasticity to fluid motion, producing visual effects that are nearly indistinguishable from real life. With this added realism, however, comes substantial computational expense, often placing detailed physical simulation in the realm of offline computations involving many degrees of freedom.

In settings like interactive graphics, however, it is advantageous to reparameterize the system with a much smaller number of degrees of freedom which describe only the states that are actually of interest. These subspaces, or reduced order models enable downstream tasks, most traditionally fast simulation in reduced coordinates, but also other operations such as user-guided animation, interpolation, or sampling states of the system.

However, identifying such subspaces is inevitably challenging, because they must trade-off between the conciseness and expressivity. Classical reduced-order simulation methods such as linear modal analysis or modal derivatives have typically focused on perturbative motions about a rest state for a deformable object. These methods are highly effective numerical schemes for fast forward-integration of system dynamics; our approach will seek a complementary technique in two senses.

First, such approaches typically only approximate object behavior in a truncated region about the rest pose, and dramatic nonlinear motions are not well-represented in the subspace. This concern is already impactful for the classic case of deformable bodies undergoing large motions, but is a total show-stopper when seeking reduced kinematics for more general physical systems, such as rigid bodies under collision penalties. In these settings, the kinematic landscape is so nonlinear that an approximation in terms of a local expansion does not capture any significant behavior.

Second, our subspaces will parameterize only the desired configuration space of the system. The challenge of large motions can be mitigated in perturbative methods by using a moderately large reduced basis. However, this returns to the original problem with the full configuration space: the relevant system configurations again lie only on narrow submanifold of the space. In contrast, we seek very low-dimensional but highly nonlinear subspaces, such that even large motions and physical systems with irregular potential landscapes can be directly parameterized; an important property for applications like animation and sampling.

This work is not the first to propose using a richer class of highly nonlinear models to fit reduced kinematics. Our motivating goal is to do so without data-driven fitting; we do not require any dataset of representative simulation trajectories or states as input. Collecting such a high-quality dataset is challenging and labor-intensive, both in the sense of engineering effort and user input. It is a significant obstacle for past methods which otherwise offer excellent properties. To be clear, although our method leverages tools from machine learning, it is not data-driven in the usual sense. Instead, it mirrors recent "overfit" neural networks, where models are fit in isolation to each example, and neural networks are used simply as a general and easy-to-optimize nonlinear function space.

**Summary.** In this paper, we apply machine learning to identify a nonlinear reduced model for physical motion. Our approach is designed around two significant properties:
* We do not assume input data such as simulation trajectories are provided. Instead, our method is self-supervised, taking the energy function as input and automatically sampling it to explore the low-energy subspace.
* Our method is very general, and avoids specific assumptions about e.g. deformable bodies. It applies broadly across systems such as rigid bodies and linkages under penalty potentials, or even multi-physics combinations of several different interacting systems.

Provided a differentiable potential energy function describing a given physical system and a single seed state from which to begin the search, our learning algorithm automatically determines an effective nonlinear low-order model, trading off between staying in low-potential energy configurations and coverage of the configuration space. 

## 2 RELATED WORK
*(Omitted for brevity - covers Subspaces for Simulation and Neural/Data-Driven Methods)*

## 3 METHOD
We present a straightforward approach to fit a neural network modeling low-energy kinematics of a physical system. 

### 3.1 Neural Subspace Maps
Consider a map $f_{\theta}$, which takes a low-dimensional subspace $\mathbb{R}^{d}$ to the high-dimensional configuration space $\mathbb{R}^{n}$ of some physical system ($d \ll n$), so $f_{\theta}:\mathbb{R}^{d}\rightarrow\mathbb{R}^{n}$. For example, $\mathbb{R}^{n}$ might represent the set of all possible vertex configurations for a given triangle mesh (so, $n=3|V|$ where $|V|$ is the number of vertices). The vector $\theta \in \mathbb{R}^{k}$ contains learnable parameters specific to the physical system, e.g. neural network weights.

Classical simulation algorithms operate on $\mathbb{R}^{n}$, where the potential energy $E_{pot}:\mathbb{R}^{n}\rightarrow\mathbb{R}$ and external forces can be evaluated directly. However, $\mathbb{R}^{n}$ contains many unlikely configurations, corresponding to high-energy deformations under the potential energy $E_{pot}$. In many settings, we can reasonably expect the kinematics to stay in the image $f_{\theta}(\mathbb{R}^{d})$ of some map $f_{\theta}$ parameterizing typical configurations.

### 3.2 Objective Function
For any choice of subspace architecture $f_{\theta}$, we will fit the parameters $\theta$ via stochastic gradient descent on an objective function. We optimize for $\theta$ directly using the analytical description of the system—in particular, the potential energy function $E_{pot}(\cdot)$—rather than requiring training data.

We might seek low-energy subspaces of a system by minimizing the expected potential energy of randomly-sampled subspace configurations $z$ as follows:

$$ \mathbb{E}_{z \sim \mathcal{N}}[E_{pot}(f_{\theta}(z))] $$

Here, $\mathcal{N}$ denotes the Gaussian distribution over $\mathbb{R}^{d}$ with mean $0$ and variance $I_{d \times d}$. 

Minimizing this yields an uninteresting map $f_{\theta}$ that maps all latent variables $z$ to the lowest-energy configuration (the minimizer of $E_{pot}$). To combat this degeneracy, we attempt to impose isometry up to scale on $f_{\theta}$, enforcing that $|f_{\theta}(z)-f_{\theta}(z^{\prime})|_{M} \approx \sigma|z-z^{\prime}|$ for typical $z,z^{\prime} \in \mathbb{R}^{d}$. Here the distance in configuration space $\mathbb{R}^{n}$ is measured with respect to the system's mass matrix $M \in \mathbb{R}^{n \times n}$: $|x|_{M}^{2} := x^{\top}Mx$. Equivalently, we write:

$$ \log \frac{|f_{\theta}(z)-f_{\theta}(z^{\prime})|_{M}}{\sigma|z-z^{\prime}|} \approx 0 $$

Enforcing strict equality is a stiff constraint, so we use a soft penalty to avoid degeneracies:

$$ \mathbb{E}_{z, z^{\prime} \sim \mathcal{N}} \left[ \left( \log \frac{|f_{\theta}(z)-f_{\theta}(z^{\prime})|_{M}}{\sigma|z-z^{\prime}|} \right)^2 \right] $$

Combining these terms and using $\lambda \in \mathbb{R}$ as a weight, we optimize for the parameters $\theta$ as follows:

$$ \min_{\theta} \mathbb{E}_{z, z^{\prime} \sim \mathcal{N}} \left[ E_{pot}(f_{\theta}(z)) + \lambda \left( \log \frac{|f_{\theta}(z)-f_{\theta}(z^{\prime})|_{M}}{\sigma|z-z^{\prime}|} \right)^2 \right] $$

The hyperparameter $\sigma$ adjusts the size of the subspace. Small $\sigma$ yields subspaces tightly concentrated around low-energy configurations.

### 3.3 Reduction to Modal Analysis
PROPOSITION 3.1. Suppose $f_{\theta}$ has the capacity to represent affine functions. Then, as $\sigma\rightarrow0$ and $\lambda\rightarrow\infty$ the solution satisfies:

$$ f(z) = Az + b $$
$$ b = \arg \min_{b} E_{pot}(b) $$
$$ A = \text{TOP-d-GENERALIZED-EIGENVECTORS}(M, H(b)) $$

In words, as we push to preserve geometry exactly and prioritize small neighborhoods, we recover a linearization about the minimum-energy state (classic linear modal analysis).

### 3.4 Subspace Simulation
We optimize to obtain the subspace configuration $z$ in the next timestep as:

$$ \hat{z} = \arg \min_{z} \left[ \frac{1}{2h^{2}}|f_{\theta}(z)-\overline{q}|_{M}^{2} + E_{pot}(f_{\theta}(z)) \right] $$

where $h$ is the timestep and $\overline{q}$ is an inertial guess computed from previous configurations.

### 3.5 Conditional Subspaces
We can generalize $f_{\theta}$ to incorporate conditional parameters (e.g., material stiffness) as additional inputs: $f_{\theta}:\mathbb{R}^{d} \times \mathbb{R}^{m} \rightarrow \mathbb{R}^{n}$, yielding $q \leftarrow f_{\theta}([z,c])$.

## 4 ARCHITECTURES AND TRAINING
We use multi-layer perceptrons (MLPs).

### 4.1 Seeded Subspace Exploration
When starting from a randomly-initialized neural subspace map, finding any point on the low-energy submanifold is a hard optimization problem. We propose exploring outward from an initial seed configuration $q_{seed} \in \mathbb{R}^{n}$. During training only, we parameterize the neural subspace map as:

$$ f_{\theta}(z) := p \cdot \text{MLP}_{\theta}(z) + (1-p)q_{seed} $$

where $p$ is a scheduling parameter which linearly increases from $0 \rightarrow 1$ as training proceeds. At the conclusion of training, this seed state is entirely absent.

## 5 EVALUATIONS
*   **FEM (Deformable Objects):** Used neo-Hookean material model for 2D/3D elements.
*   **Cloth Model:** Simple energy model with bending term at edges and StVK stretching term on faces.
*   **Penalty Functions:** Enforce constraints (like collisions or joints) via penalties: $w_{eq}|C_{eq}(q)|^{2} + w_{ineq}|\min(C_{ineq}(q),0)|^{2}$. 
*   **Rigid Bodies & Linkages:** Complex linkages modeled as free-floating rigid objects with strong penalties at joints, successfully learning low-dimensional parameterizations (e.g., planar Klann linkage).

**Comparisons:** Compared against linear modal analysis and modal derivatives on a heterogeneous 3D bar and rigid body chain. Classic local methods struggle to capture the non-linear rotations and bending, whereas the neural subspace accurately matches reference physics.

## 6 CONCLUSION
Introduces a promising approach for fitting kinematic subspaces directly to physical systems without gathering datasets of trajectories. Limitations include vulnerability to local minima and lack of theoretical guarantees regarding which physical effects are truncated.