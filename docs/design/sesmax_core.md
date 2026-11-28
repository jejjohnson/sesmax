---
title: "sesmax-core: what does every model have to be?"
subtitle: "Design · the contract of a differentiable model, and the pieces every component shares"
short_title: sesmax-core
description: "Design of sesmax-core: motivation, user stories, the maths of a model as a differentiable map, its numerics in JAX, the API as it is and as proposed, examples that run, what is available today, and references."
date: 2026-11-26
keywords:
  - sesmax
  - core
  - differentiable model
  - term algebra
  - data assimilation
---

26 Nov 2026 · J. Emmanuel Johnson (drafted with Claude Code)

## Decision

Decided on 30 October 2026, and built: `sesmax-core` is the bottom package of sesmax. It holds the contract that makes a model usable by an integrator, an optimiser, a filter and a run loop at once, and the pieces that every component of the Earth system shares.
Decided by the owner on 26 November: the core composes the machinery of the stack and writes none. Time stepping is diffrax and integrix, stencils and grids are gridax, linear algebra is lineax and gaussx, bases are functax, constants are geophysix.
Proposed, for review: the core shrinks to the contract. The toys move to [`sesmax-testing`](sesmax_testing.md), drag, forcing and bases to [`sesmax-param`](sesmax_param.md), and the known fields stay as `Covariates`.
Proposed on 27 November, for review: a component says what its models show, as named `Observable`s, and a model can be seen through any encoder and decoder, of which `ScaledModel` is the affine case. Training either map stays out of sesmax.
Alternative: keep one core and add modules to it, which is fewer releases, but then everyone who wants the model contract installs 75 toy names and eight basis constructors.
Status: released at 0.3.1, at sesmax `a34e46c`. The section on where it came from has the history of the move from somax and what changed since. This page is the design of the package.

## Motivation

A forecast, a gradient, an ensemble and a year-long run are four uses of one function: the map from a state now to a state later.
Most model codes are written for the first use, and the other three are added around them.
The adjoint is then a second code that drifts from the first, the ensemble is a loop over processes, and the parameters are numbers in a namelist that no optimiser can reach.

sesmax is for short-window data assimilation, where all four uses happen in the same hour.
So the model has to be one object that serves them all, and the question for the core is what such an object must promise.

- **To an integrator:** a tendency, and where the boundaries are.
- **To an optimiser:** which numbers are parameters, which of those are free, and which are positive.
- **To a filter or a variational method:** a step from one time to the next, on a flat vector, with a known time.
- **To a run loop:** a way to build it from a file, and something to check and to report.
- **To the next person:** a place to add a term without rewriting the model.

A second reason is that these promises do not depend on the fluid.
An ocean, an atmosphere and a land surface differ in their equations and share the rest: non-dimensional scales, layered vertical structure, elliptic inversions, forcing by a field in time.
What two components share goes here, so that they never import each other.

## User stories

- As a **model author**, I want to write a tendency and boundary conditions and get integration, stepping, gradients and batching, so that I write physics and not plumbing.
- As someone **composing physics**, I want to add and scale terms, and mark one as stiff, so that a different closure is a different sum and an IMEX solver gets the right split.
- As someone **calibrating**, I want a parameter declared positive once, so that an optimiser cannot make a viscosity negative and I never write a softplus.
- As someone **assimilating**, I want any model handed to filterax or vardax through one adapter, so that a new model needs no new assimilation code.
- As someone **scaling a problem**, I want to run in dimensionless units and read results in SI, so that a network or an optimiser sees numbers of order one.
- As a **component author**, I want to register my models under a name, so that a run loop finds them without importing my package.
- As a **reviewer**, I want each promise to be a test.

## Background: the maths

### A model is a vector field

Let $\mathbf{x}(t)$ be the state, a pytree with $D_ x$ values in all, and $\boldsymbol{\theta}$ the parameters.
A model is a tendency $f$ and a boundary operator $B$:

```math
\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = f(t, B[\mathbf{x}]; \boldsymbol{\theta})
```

$B$ fills what the interior does not determine (ghost cells, wall faces) and is a projection, $B[B[\mathbf{x}]] = B[\mathbf{x}]$.
The flow over a step $\Delta t$ is the map

```math
\mathbf{x}_{k+1} = \mathcal{M}_{\Delta t}(\mathbf{x}_k; \boldsymbol{\theta})
```

and everything else is built from $\mathcal{M}$: a trajectory is its composition, a forecast ensemble its vectorisation, and a gradient its transpose.
The tangent-linear model is $\mathbf{M} = \partial \mathcal{M} / \partial \mathbf{x}$ and the adjoint is $\mathbf{M}^\top$, which automatic differentiation gives from the same code ([Griewank and Walther, 2008](https://doi.org/10.1137/1.9780898717761); [Kidger, 2022](https://arxiv.org/abs/2202.02435)).

### A tendency is a sum of terms

```math
f(t, \mathbf{x}) = \sum_i c_i f_i(t, \mathbf{x})
```

with coefficients $c_ i$ that are leaves, so a gradient reaches them.
Each term is explicit or implicit. With $f = f_ \text{E} + f_ \text{I}$, an implicit-explicit scheme takes the non-stiff part explicitly and the stiff part (a diffusion, a fast wave) implicitly ([Ascher et al., 1997](https://hdl.handle.net/10.1016/S0168-9274(97)00056-1)):

```math
\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t f_\text{E}(\mathbf{x}_k) + \Delta t f_\text{I}(\mathbf{x}_{k+1})
```

in its first-order form, so the step is bound by the advection and not by the diffusion.

### Constrained parameters

A positive parameter $\theta > 0$ is stored as $\rho \in \mathbb{R}$ with $\theta = \log(1 + e^\rho)$, and one in an interval $(a, b)$ as $\theta = a + (b - a) \sigma(\rho)$ with $\sigma$ the logistic function.
An optimiser moves $\rho$ freely and the model always sees a valid $\theta$.

### Scales

With a length $L$, a velocity $U$, a depth $H$ and a Coriolis parameter $f_ 0$, a state is non-dimensionalised leaf by leaf by an affine map

```math
\mathbf{x}' = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}
```

and time by $t = T t'$.
The time scale depends on the regime: $T = L / U$ for the advective set, $T = 1 / f_ 0$ for the inertial one.
The same map with a sample mean and standard deviation standardises a state for a network or a filter, so the two compose.
The numbers that say which regime a run is in are $Ro = U / (f_ 0 L)$, $Bu = g H / (f_ 0 L)^2$ and $Fr = U / \sqrt{g H}$.

### A model in other coordinates

The affine map is one case of a pair of maps, an encoder $g_ \text{enc}: \mathbb{R}^{D_ x} \to \mathbb{R}^{D_ z}$ and a decoder $g_ \text{dec}: \mathbb{R}^{D_ z} \to \mathbb{R}^{D_ x}$. A model $\mathrm{d}\mathbf{x}/\mathrm{d}t = f(\mathbf{x}, t)$ seen in the coordinates $\mathbf{z} = g_ \text{enc}(\mathbf{x})$ is again a model, by the chain rule:

```math
\frac{\mathrm{d}\mathbf{z}}{\mathrm{d}t} = \mathbf{J}_\text{enc}(\mathbf{x}) f(\mathbf{x}, t)
```

where $\mathbf{x} = g_ \text{dec}(\mathbf{z})$ and $\mathbf{J}_ \text{enc} \in \mathbb{R}^{D_ z \times D_ x}$ is the Jacobian of the encoder. Three cases differ in what is lost.

| Pair | $\mathbf{J}_ \text{enc}$ | $D_ z$ | Exact? |
| --- | --- | --- | --- |
| scales, standardisation | $\mathrm{diag}(1 / \boldsymbol{\sigma})$ | $D_ x$ | yes |
| a basis, $\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\Phi} \mathbf{z}$ with orthonormal columns | $\boldsymbol{\Phi}^\top$ | the number of modes | only with every mode |
| a learned encoder and decoder | by automatic differentiation | the latent size | no |

The second is a Galerkin projection, $\mathrm{d}\mathbf{z}/\mathrm{d}t = \boldsymbol{\Phi}^\top f(\boldsymbol{\mu} + \boldsymbol{\Phi} \mathbf{z})$ ([Benner et al., 2015](https://doi.org/10.1137/130932715)). It drops what the kept modes exchange with the others, so a reduced model drifts from the full one even from a state inside the span. The vertical modes of the next section are the one reduction in the core that loses nothing, because the modes diagonalise the coupling.

A model that is learned in the latent space, $\mathrm{d}\mathbf{z}/\mathrm{d}t = f_ \theta(\mathbf{z})$, is not a projection of $f$. It is a model in its own right, whose decoder is what it shows. It fits the contract as it is, and training it is not the core's work.

### Layers and inversions

Two pieces of geophysical fluid dynamics are shared by an ocean and an atmosphere.

A layered fluid has a coupling matrix $\mathbf{A} \in \mathbb{R}^{N_ l \times N_ l}$, built from the thicknesses and the reduced gravities, whose eigenvectors are the vertical modes and whose eigenvalues $\lambda_ m$ give the deformation radii $R_ m = 1 / (f_ 0 \sqrt{\lambda_ m})$.
$\mathbf{A}$ is not symmetric unless the layers are equal, so it is diagonalised through its symmetric similarity $\mathbf{D}^{1/2} \mathbf{A} \mathbf{D}^{-1/2}$.

A balanced flow is recovered from its vorticity by an elliptic problem,

```math
(\nabla^2 - \lambda) \psi = q
```

with $\lambda = 0$ for a streamfunction and $\lambda = 1 / R_ m^2$ for the potential vorticity of mode $m$.

### Forcing by a field in time

A forcing is a field $u(\mathbf{s}, t)$ added to one component of the tendency.
The estimated ones are an expansion in a fixed dictionary,

```math
u(\mathbf{s}, t) = \sum_{a=1}^{D_w} w_a b_a(t) \phi_a(\mathbf{s})
```

where the coefficients $\mathbf{w}$ are the only unknowns and each mode carries a prior standard deviation.
[The param design](sesmax_param.md) has the theory, and takes this over.

### What an assimilation method needs

A filter needs $\mathcal{M}$ on a flat vector, and its ensemble.
A strong-constraint variational method minimises

```math
\mathcal{J}(\mathbf{x}_0) = \frac{1}{2} \Vert \mathbf{x}_0 - \boldsymbol{\mu}_x \Vert^2_{\boldsymbol{\Sigma}_x^{-1}} + \frac{1}{2} \sum_{t \leq T} \Vert \mathbf{y}_t - h(\mathbf{x}_t) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}}
```

where $\mathbf{y}_ t$ are observations, $h$ the observation operator, and $\boldsymbol{\Sigma}_ x$, $\boldsymbol{\Sigma}_ \varepsilon$ the background and observation covariances ([Le Dimet and Talagrand, 1986](https://doi.org/10.3402/tellusa.v38i2.11706)).
For a control that is not the state (a forcing, a parameter), the model maps the control through the whole window to the stacked observations, $G(\boldsymbol{\theta}) = \mathrm{vec}[h(\mathbf{x}_ t(\boldsymbol{\theta}))]$, and the cost is that of 3D-Var over $\boldsymbol{\theta}$.
The core supplies $\mathcal{M}$ and $G$; the methods are filterax's and vardax's.

The observation operator is a composition, $h = m \circ \mathbf{L} \circ o$, which [the experiment design](sesmax_experiment.md) sets out. Only its first map is the model's: the observable $o$ takes the state to a field on the grid, such as a sea-surface height from a streamfunction. The reading $\mathbf{L}$ at places and times is functax's, and the instrument $m$ is vardax's or radtrax's. So the core supplies one more thing, a way for a model to say what it shows and on which point of which grid.

## Background: the numerics

**Leaves and static fields.**
A model is an Equinox module, which is a pytree. An array field is a leaf: `jax.grad` differentiates it, `vmap` batches it, and changing its value does not recompile. A field marked static (a grid size, a boundary type, a scheme) is part of the structure: changing it compiles again, and no gradient reaches it. Putting a number on the wrong side is the commonest mistake, so the rule is in the contract: physical parameters are leaves, shapes and flags are static.

**Compile once.**
A solve compiles for a given structure and set of shapes. `Model.integrate` reuses the compiled solve across calls at other times and other parameter values (sesmax#196), so an optimiser or an ensemble pays for compilation once. The library itself never calls `jit`: the caller does, once, at the top.

**Where the boundary conditions act.**
`integrate` applies $B$ to the initial state, to the input of every tendency evaluation, and to every state it saves (sesmax#201). Between two stages the integrator does not, so a tendency that is non-zero on a wall face accumulates there. A term may fill its own ghosts when advection and diffusion need different values at the same wall.

**Time stepping.**
`integrate` is `diffrax.diffeqsolve`, by default with the pair of [Tsitouras (2011)](https://doi.org/10.1016/j.camwa.2011.06.002) at a fixed step. `step` advances one increment and returns a bare state, and the adapters and `rollout` call only `step`, so a model changes how it integrates by overriding `build_terms`. `imex_solver` and `stable_step` are integrix's, re-exported.

**Adjoints.**
Reverse mode through a solve stores the trajectory. diffrax's default checkpoints it, and supports reverse mode only; forward mode, which Gauss–Newton and Hessian-vector products need, takes `DirectAdjoint`. `rollout` stores every state by default. With `checkpoint=True` it recomputes each block of `save_every` steps on the backward pass and stores about $n / s + s$ states for a block of $s$ steps, which is about $2\sqrt{n}$ when $s$ is near $\sqrt{n}$. `rollout_loss` checkpoints by default, but its block is its `stride`, one step unless set, so it saves memory only when the stride is near $\sqrt{n}$. `rollout` passes no adjoint to the model today, which [the experiment design](sesmax_experiment.md) fixes.

**Located fields.**
A field is an array or a `gridax.CField`, which knows its point on the C-grid. The helpers `each`, `values` and `placed` let one formula serve both, and fields on different points raise instead of adding silently.

**Flattening.**
filterax and vardax work on vectors and sesmax on pytrees. An adapter keeps a template state and flattens on each call, so it stays a clean pytree under `jit` and `vmap`. A transform used to build an ensemble must be given to the adapter too, or the model receives standardised values as if they were physical.

**Time in a step.**
`VardaxForwardModel` passes the same start time to every substep, so it is exact only for an autonomous model. `WindowOperator` lets the model integrate the window with its own clock, which a forcing that depends on time needs.

**Guards.**
A blow-up inside a jitted solve is caught where it happens by `guard_finite`, `guard_positive` and `guard_ceiling` in the tendency. A check between chunks of a run is a monitor of `sesmax-experiment`.

**Optional dependencies.**
`import sesmax_core` loads no assimilation library, no xarray and no flowjax. Each sits behind an extra, and importing its module without the extra raises an `ImportError` that names it.

## What it is built on

| Need | From | Name |
| --- | --- | --- |
| modules as pytrees | equinox | `eqx.Module`, `eqx.field(static=True)` |
| integration, adjoints | diffrax | `diffeqsolve`, `ODETerm`, `MultiTerm`, the solvers |
| the IMEX solver, the stable step, a forward model, sampled paths | integrix | `march.imex_solver`, `march.stable_step`, `march.DiffraxForwardModel`, `SampledPath` |
| constrained parameters | paramax | `Parameterize`, `unwrap` |
| grids, masks, located fields, stencils, ghost fills | gridax | `CartesianCGrid2D`, `Mask2D`, `CField`, `GridOps`, `fill_ghosts` |
| the conjugate-gradient inversions | gaussx | `linear_solve_with_info` |
| interpolation of a record in time | functax | `functax.interp` schemes |
| the dictionaries of the spatial bases | geonnax | `geonnax.basis` |
| constants, the Coriolis parameter | geophysix | `constants.STANDARD`, `rotation` |
| operators and cycles | pipekit | `pipekit_ops.Operator`, `pipekit_cycle.ForwardModel` |
| filters and variational methods, behind `[da]` | filterax, vardax | the adapters' targets |

Four places hold what another library has or will have.

- **The bases** go through `geonnax.basis`; functax has them now, and sesmax#114 moves the core onto it.
- **`SubsampleObs`** is an observation operator of our own, where vardax has `GridPointObs`, `InterpObs` and `MaskedIdentity` (sesmax#27).
- **The toys** move to `sesmax-testing`, with checks built on pipekit-array's `testing` module, which the core does not import today.
- **Drag, wind stress and relaxation** are parameterizations and forcing terms, and move to `sesmax-param`.

## The API

```mermaid
flowchart TD
    types["State, Params, PhysConsts, Diagnostics<br/>positive, interval, frozen"] --> model
    terms["Term, Sum, Scaled, implicit"] --> tm["TermModel"]
    model["Model<br/>vector_field, apply_boundary_conditions"] --> tm
    model --> integ["integrate, step, as_forward_model"]
    integ --> train["train<br/>rollout, rollout_loss"]
    integ --> da["da<br/>FilteraxDynamics, VardaxForwardModel, WindowOperator"]
    scales["scales<br/>Scales, StateAffine, ScaledModel"] --> model
    forcing["forcing<br/>ForcingProtocol, BasisForcing, ForcingTerm"] --> terms
    gfd["gfd, vertical<br/>inversions, modes"] --> model
    comp["component<br/>Component, BuiltRun"] --> model
```

| Rung | Name | Job |
| --- | --- | --- |
| structure | `State`, `Params`, `PhysConsts`, `Diagnostics`, `Domain`, `Scales` | what a model carries |
| primitive | `Term`, the inversions, the vertical modes, the guards | one operation each |
| strategy | `positive`, `interval`, `frozen`, `explicit`, `implicit`, a solver | how a parameter or a term is treated |
| model | `Model`, `TermModel`, `ScaledModel` | a tendency with boundaries |
| step | `integrate`, `step`, `rollout` | the flow |
| recipe | `Component`, the adapters | a model offered to a run or to a method |

### The model contract

```python
# Shapes: the state is a pytree with D_x values

# today (sesmax-core)
class Model(eqx.Module):
    # f(t, x; θ), a tendency with the structure of the state
    @abc.abstractmethod
    def vector_field(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree: ...

    # B[x]
    @abc.abstractmethod
    def apply_boundary_conditions(self, state: PyTree) -> PyTree: ...

    # optional: energy, invariants, derived fields
    def diagnose(self, state: PyTree) -> Diagnostics: ...

    # x(t₁) from x(t₀), a diffrax Solution whose saved states have B applied
    def integrate(self, state0: PyTree, t0: float, t1: float, dt: float, **kw) -> dfx.Solution: ...

    # x_{k+1} = ℳ(x_k), a bare state
    def step(self, state: PyTree, dt: float, *, t0: float = 0.0, **kw) -> PyTree: ...

    # what a pipekit cycle or a filter drives
    def as_forward_model(self, *, dt: float, dt0: float, **settings) -> DiffraxForwardModel: ...

    # override for an SDE or an IMEX split; never override step
    def build_terms(self) -> dfx.AbstractTerm: ...
```

Two methods are the whole obligation of an author. The rest is derived, and is the same for every model.

### Parameters

```python
# today (sesmax-core)
# θ = softplus(ρ) > 0
def positive(value: ArrayLike) -> Array: ...
# θ = a + (b − a) σ(ρ)
def interval(value: ArrayLike, lower: float, upper: float) -> Array: ...
# a leaf no gradient reaches
def frozen(value: ArrayLike) -> Array: ...
# True at the leaves an optimiser may move
def trainable_mask(tree: PyTree) -> PyTree: ...
```

`Model` unwraps a constrained parameter on the way into `vector_field` and `diagnose`, so a model reads it as a plain array and a gradient lands on the stored value.

### Terms

```python
# today (sesmax-core)
class Term(eqx.Module):
    # fᵢ(t, x), zero in the fields it does not touch
    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree: ...

# f = f₁ + c f₂ − f₃, and f_I marked for the implicit stage
tree = advection + 2.0 * drag - damping + implicit(diffusion)
# (f_E, f_I)
explicit_part, implicit_part = partition(tree)

# today (sesmax-core): a model that is its term tree
model = TermModel(terms=tree)
```

`TermFn` wraps a plain function, and `OperatorTerm` lifts a pipekit operator.

### Scales, forcing, shared physics

```python
# today (sesmax-core)
scales = Scales.advective(L=1e6, U=0.1, f0=1e-4, H=1000.0)   # T = L/U
transform = StateAffine.from_scales(StateClass, scales)      # x' = (x − μ)/σ
scaled = ScaledModel.from_scales(model, scales, StateClass)   # the model in dimensionless units

# today (sesmax-core): u(s, t) = ∑ₐ wₐ bₐ(t) φₐ(s), lifted onto one field
forcing = BasisForcing(coeffs=w, spatial=spatial_from_fourier(...), temporal=HatInTime(knots), grid_shape=shape)
term = ForcingTerm(forcing, place=add_to("q"), grid=grid)

# today (sesmax-core): (∇² − λ) ψ = q
psi = streamfunction_from_vorticity(zeta, dx, dy, bc="dst")
psi = pv_inversion(pv, dx, dy, lambda_=1.0 / radii**2)
# today (sesmax-core): the modes of a layered fluid
modes = decompose_vertical_modes(build_coupling_matrix(H, g_prime), f0)
```

### Adapters and components

```python
# today (sesmax-core), behind [da]
dynamics = FilteraxDynamics(model=model, template=state)              # for filterax
forward = VardaxForwardModel(model=model, template=state, dt=0.05)    # for vardax, autonomous models
window = WindowOperator(build_model=..., ...)                         # G(θ), for a forcing or a parameter

# today (sesmax-core): what a package of models registers,
# under the entry-point group "sesmax.components"
COMPONENT = Component(name="ocean", models={...}, scenarios={...}, checks={...}, evaluate=...)
```

### Proposed

```python
# proposed (sesmax-core): the known fields from outside the model, by name
# c(s, t), in their own units, outside the trainable leaves
covariates = Covariates({"tau_x": InterpolatedField.from_data(ts, tau_x), "depth": depth})


# proposed (sesmax-core): one more field of Component
# the priors of a model's state and of its own parameters, by path
priors: Callable[[Model, PyTree], Mapping[str, Any]] | None = None


# proposed (sesmax-core): rollout takes the adjoint of the solve
def rollout(model, state0, *, dt, n_steps, adjoint=None, ...) -> PyTree: ...


# proposed (sesmax-core): what a model can show, and where it lives
class Observable(eqx.Module):
    """A field of the model on its grid: the map o, with what a reading needs."""

    # o(x), a located field, so its point is known
    read: Callable[[PyTree], CField]
    # coords(axis, at=), today (gridax)
    grid: Any
    # wet(at=), today (gridax); None for no land
    mask: Any | None = None


# proposed (sesmax-core): one more field of Component
# the observables of a model, by name: "ssh", "u", "v"
observables: Callable[[Model], Mapping[str, Observable]] | None = None


# proposed (sesmax-core): a model seen through a pair of maps
class TransformedModel(Model):
    """dz/dt = J_enc(x) f(x, t), with x = g_dec(z)."""

    inner: Model
    encode: Callable[[PyTree], PyTree]
    decode: Callable[[PyTree], PyTree]

    def vector_field(self, t, state, args=None):
        # x = g_dec(z)
        inner_state = self.decode(state)
        # f(x, t)
        tendency = self.inner.vector_field(t, inner_state, args)
        # J_enc(x) f, one forward-mode product
        return jax.jvp(self.encode, (inner_state,), (tendency,))[1]

    def apply_boundary_conditions(self, state):
        # the boundaries are those of the inner model
        return self.encode(self.inner.apply_boundary_conditions(self.decode(state)))
```

- **Covariates.** [The param design](sesmax_param.md) separates a known field from a forcing term. The reader of records (`InterpolatedForcing` today) stays in the core under that name, since a component, a term and an experiment all read it.
- **Priors.** [The experiment design](sesmax_experiment.md) reads a component's priors through this field, so it names no component.
- **Adjoints.** Recommended: a spec that integrix resolves to a diffrax adjoint. Alternative: a diffrax object, which ties every caller to diffrax.
- **Observables.** A component names the fields its models show, and `sesmax-experiment` builds the reading from the grid and the mask of each. Recommended: a located field, so that the point comes with it. Alternative: a bare array and a point name, which is one more thing to get wrong.
- **A model in other coordinates.** `ScaledModel` stays as the affine case, which is exact, needs no Jacobian product and also scales time. `TransformedModel` is the general case, for a basis or a learned pair. Its maps are any two callables, so the `.encode` and `.decode` of a filterax codec fit, and filterax's `EncodedDynamics` is the same idea for one step of a filter. Diagnostics are those of the inner model on the decoded state. A model learned in the latent space needs nothing new: it is a `Model`, and its decoder is its `Observable`.
- **What stays out.** Fitting a basis is vardax's `eof_basis`, and training an encoder, a decoder or a latent model is pipekit-train's with the networks of geonnax. The core holds the slot they fill.
- **What leaves.** The toys (75 names) to `sesmax-testing`; drag, wind, relaxation, the bases and `BasisForcing` (most of the 48 names of `forcing`) to `sesmax-param`; `SubsampleObs` to vardax's operators. Recommended: move with the spelling kept, so a caller changes one import.

## Examples

Each block ran on 26 November 2026 at sesmax `a34e46c`, in float64; the printed values are in the comments.

### A model in ten lines

```python
import diffrax as dfx
import jax
import jax.numpy as jnp
from sesmax_core import Model, positive

# Shapes: D_x = 3


# dx/dt = −r x, with r > 0
class Decay(Model):
    rate: jax.Array

    def vector_field(self, t, state, args=None):
        return -self.rate * state

    def apply_boundary_conditions(self, state):
        return state


# today (sesmax-core): r is stored in softplus space
model = Decay(rate=positive(0.5))

# x₁ = ℳ(x₀)
model.step(jnp.ones(3), 0.1)                       # [0.9512, 0.9512, 0.9512], and exp(−0.05) = 0.9512

# x(t) at three times
solution = model.integrate(
    jnp.ones(3), 0.0, 1.0, 0.1, saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 1.0, 3))
)
solution.ys.shape                                  # (3, 3), and ys[-1, 0] = 0.60653 = exp(−0.5)
```

### A model as a sum of terms

```python
from sesmax_core import TermModel
from sesmax_core.terms import Term, implicit, partition


# −r x
class Relax(Term):
    rate: jax.Array

    def __call__(self, t, state, args=None):
        return -self.rate * state


# a constant push
class Push(Term):
    amplitude: jax.Array

    def __call__(self, t, state, args=None):
        return self.amplitude * jnp.ones_like(state)


# today (sesmax-core): f = 1 − 2 · 0.5 x
model = TermModel(terms=Push(jnp.asarray(1.0)) + 2.0 * Relax(jnp.asarray(0.5)))
model.vector_field(0.0, jnp.ones(2))               # [0., 0.]

# a stiff relaxation goes to the implicit stage of an IMEX solver
explicit_part, implicit_part = partition(Push(jnp.asarray(1.0)) + implicit(Relax(jnp.asarray(50.0))))
```

### The gradient of a trajectory with respect to a parameter

```python
import equinox as eqx
from sesmax_core.testing import L96State, Lorenz96
from sesmax_core.train import rollout, rollout_loss

# Shapes: D_x = 40; T = 20 steps of 0.01

# today (sesmax-core)
# dx_k/dt = (x_{k+1} − x_{k−2}) x_{k−1} − x_k + F
truth = Lorenz96.create(F=8.0)
state0 = L96State.init_state(ndim=40, key=jax.random.key(0))
# → (T, D_x)
reference = rollout(truth, state0, dt=0.01, n_steps=20)

# ℒ(F) = (1/T) ∑ₜ ‖xₜ(F) − xₜʳᵉᶠ‖², at F = 7
gradient = eqx.filter_grad(rollout_loss)(Lorenz96.create(F=7.0), state0, reference, dt=0.01)
# dℒ/dF = −0.0247: the loss falls as F rises towards 8
```

### A model in other coordinates

The slot is five lines, and the run below is a scratch one from 27 November 2026, on Lorenz-96 with 40 variables.

```python
import jax
import sesmax_core as sx

# proposed (sesmax-core), as a function: dz/dt = J_enc(x) f(x), with x = g_dec(z)
def latent_field(model, encode, decode):
    def field(t, z, args=None):
        x = decode(z)
        return jax.jvp(encode, (x,), (model.vector_field(t, x, args),))[1]

    return field


# today (sesmax-core): the affine case
affine = sx.StateAffine(loc=L96State(x=jnp.asarray(2.0)), scale=L96State(x=jnp.asarray(4.0)))
scaled = sx.ScaledModel(inner=truth, transform=affine)
# the two tendencies differ by 0.0
latent_field(truth, affine.forward, affine.inverse)(0.0, affine.forward(state0))

# a basis from a long run: z = Φᵀ (x − μ), x = μ + Φ z
encode = lambda x: Phi.T @ (x - mean)
decode = lambda z: mean + Phi @ z
```

| Modes kept | Variance held | Error of the start, in the basis | Error after 0.2 time units |
| --- | --- | --- | --- |
| 40 | 100 % | 7e-15 | 1e-14 |
| 30 | 97.8 % | 1.05 | 2.43 |
| 20 | 90.2 % | 1.51 | 3.13 |
| 10 | 70.5 % | 2.29 | 3.01 |

The errors are root mean squares over the 40 variables, against the full model from the same state. With every mode the reduced model is the full one to rounding, which is the check that the slot is right. With fewer it is poor, because the spectrum of Lorenz-96 is nearly flat. So Lorenz-96 tests the slot and not the idea, which needs a model with a red spectrum.

### One model, two assimilation libraries

```python
# today (sesmax-core), behind [da]; not run for this page
from sesmax_core.da import FilteraxDynamics, VardaxForwardModel, make_ensemble

# N_E = 32 members around the background
ensemble = make_ensemble(state0, jax.random.key(1), size=32, std=0.1)
# for a filterax filter
dynamics = FilteraxDynamics(model=truth, template=state0)
# for a vardax 4D-Var
forward = VardaxForwardModel(model=truth, template=state0, dt=0.05)
```

The twin experiments in `packages/sesmax-core/tests/da/` run both, and the notebooks `data_assimilation_etkf` and `data_assimilation_4dvar` show them.

## What is available

Counted at sesmax `a34e46c`: 15,612 lines of source, 1,026 test functions in 60 files, 89 names at the top level, and 30 executed notebooks in the docs.

| Module | Holds | State |
| --- | --- | --- |
| top level | `Model`, `TermModel`, `State`, `Params`, `PhysConsts`, `Diagnostics`, `positive`, `interval`, `frozen`, `trainable_mask` | released |
| `terms` | `Term`, `TermFn`, `Sum`, `Scaled`, `Compose`, `explicit`, `implicit`, `partition`, `build_diffrax_terms`, `OperatorTerm` | released |
| `scales` | `Scales`, `StateAffine`, `ScaledModel` | released |
| `domain` | `Domain`, `TimeDomain` | released |
| `fields` | `each`, `map_fields`, `placed`, `values`, `is_field` | released |
| `solvers` | `imex_solver`, `imex_stepsize_controller`, `stable_step`, from integrix | released |
| `train` | `rollout`, `rollout_loss`, `state_mse`, `ModelStep` | released |
| `guards` | `guard_finite`, `guard_positive`, `guard_ceiling` | released |
| `validation` | `require_positive`, `require_non_negative`, `reject_derived_kwargs` | released |
| `gfd` | `streamfunction_from_vorticity`, `pressure_from_divergence`, `pv_inversion`, `ModalTransform`, `StratificationProfile`, the Burger-number helpers | released |
| `vertical` | `multilayer`, `build_coupling_matrix`, `decompose_vertical_modes`, `layer_to_mode`, `mode_to_layer`, `barotropic_filter` | released |
| `operators` | `ModelOp`, `OperatorTerm` | released |
| `component` | `Component`, `Builder`, `BuiltRun`, `Check` | merged, after 0.3.1 |
| `da`, behind `[da]` | `FilteraxDynamics`, `VardaxForwardModel`, `WindowOperator`, `SubsampleObs`, `state_to_vector`, `make_ensemble` | released |
| `io`, behind `[xarray]` | states to and from xarray and Zarr, with scale metadata | released |
| `flows`, behind `[flows]` | `StateAffineBijection`, `to_flowjax`, `learned_prior` | released |
| `forcing` | 48 names: fields in time, `BasisForcing`, `ForcingTerm`, eight spatial and four temporal bases, drag, wind stress, relaxation | released; most of it moves to `sesmax-param` |
| `testing` | 75 names: Lorenz-63, Lorenz-96 and its two-scale form, the 1-D and 2-D PDE problems, linear shallow water | released; moves to `sesmax-testing` |
| monitors, checkpoints, the run loop | — | moved to `sesmax-experiment` (sesmax#197) |
| `Covariates`, `Component.priors`, an adjoint in `rollout` | — | proposed here |
| `Observable`, `Component.observables` | — | proposed here, for [sesmax-experiment](sesmax_experiment.md), E2a |
| `TransformedModel` | the affine case is `ScaledModel` | proposed here |
| gradient checks against finite differences | — | open, sesmax#105 |
| the levels L1 to L3 (`_src/array.py`, `Step`, `simulate`) | — | open, sesmax#29 |
| the walls of the linear shallow-water toys | defect | sesmax#202 |

The tutorial series, steps 1 to 16, follows the twelve steps to Navier–Stokes of [Barba and Forsyth (2018)](https://doi.org/10.21105/jose.00021) and continues to gravity waves and geostrophic adjustment.

## Where it came from

Decided on 30 October 2026: sesmax is a uv workspace in its own repository, and `sesmax-core` began as a copy of somax's model-agnostic machinery at somax `9cb19b3`, with mechanical renames and no change in behaviour. The alternative, rejected, was to rename somax in place and split it there, which keeps the git history and drops the template scaffold. This section keeps the record of that move: what went where, what was renamed, and what was decided. A file moved from somax is still checked against its source as `AGENTS.md` says.

### Where each somax module went

Every module in `somax/_src/` at `9cb19b3` has one destination; the imports were read from the source (`git show 9cb19b3:<path>`), not run. The table is the plan, with the rows that ended somewhere else corrected on 26 November.

| somax module (`somax/_src/…`) | Holds | Goes to | Change |
| --- | --- | --- | --- |
| `core/types.py` | `State`, `Params`, `PhysConsts`, `Diagnostics`, `positive`, `interval`, `frozen`, `as_parameter`, `trainable_mask` | `sesmax_core` (top level) | none |
| `core/model.py` | `SomaxModel`, `TermModel` | `sesmax_core` (top level) | `SomaxModel` → `Model` |
| `core/terms.py`, `operator_term.py` | `Term`, `TermFn`, `Sum`, `Scaled`, `Compose`, `explicit`, `implicit`, `partition`, `build_diffrax_terms`; `OperatorTerm` | `sesmax_core.terms` | `OperatorTerm` joins the term algebra |
| `core/forcing.py`, `core/basis.py` | `ForcingProtocol` and the constant, seasonal and interpolated forcings; `SpatialBasis`, `TemporalBasis`, `BasisForcing`, `ForcingTerm`, `add_to`, `control_filter` and the vector versions | `sesmax_core.forcing` | none |
| `core/forcing_bank.py` (generic half) | `spatial_from_*` (Gabor, RBF, Fourier, graph Laplacian, EOF, wavelet, spherical RBF, divergence-free), `tile_in_time`, `GaussianWindowsInTime`, `matern_spectral_density` | `sesmax_core.forcing` | all eight stay over geonnax (SX-5) |
| `core/forcing_bank.py` (ocean presets) | `ssh_geostrophic`, `sss_coastal`, `sst_frontal`, `geostrophic_currents` | `sesmax-ocean` (C2) | named after ocean fields |
| `core/scales.py`, `core/scaled.py` | `Scales` (Rossby, Burger, Froude numbers, `dt_from_cfl`), `ScaledModel` | `sesmax_core.scales` | none |
| `core/transforms.py` | `StateAffine`; `StratificationProfile`, `ModalTransform` | `StateAffine` to `sesmax_core.scales`; the layered pieces to `sesmax_core.gfd` | split by concept |
| `core/helmholtz.py` | the periodic, Dirichlet, Neumann and multimodal Helmholtz caches over spectraldiffx | moved to `sesmax_core.gfd`, then removed on 5 Nov 2026 because no model used them | none |
| `models/_nondim.py` | `require_positive`, `require_non_negative`, `reject_derived_kwargs`; `resolve_burger`, `burger_to_g_prime`, `resolve_burger_spherical` | validators to `sesmax_core.validation`, public since sesmax#163; Burger helpers to `sesmax_core.gfd` | split by concept |
| `constants/` | physical constants | not moved: nothing imported them, and geophysix owns them (`geophysix.constants.STANDARD`, `geophysix.earth.SPHERE`) | none |
| `domain/` | `Domain`, `TimeDomain` | `sesmax_core.domain` | none |
| `solvers.py` | `imex_solver`, `imex_stepsize_controller` (Krylov Newton) | `sesmax_core.solvers` | since moved to integrix and re-exported |
| `guards.py` | `guard_finite`, `guard_positive`, `guard_ceiling` | `sesmax_core.guards` | none |
| `monitor/` | `Monitor`, `BaseMonitor`, the six built-in monitors, `default_monitors` | `sesmax_core.monitor`, then `sesmax_experiment.monitor` | moved again in sesmax#197 |
| `train/` | `rollout`, `rollout_loss`, `state_mse`, `ModelStep` | `sesmax_core.train` | none |
| `operators.py` | `SomaxModelOp`; `Burgers2DOp` | `ModelOp` to `sesmax_core.operators`; `Burgers2DOp` to `sesmax_core.testing` | `SomaxModelOp` → `ModelOp` |
| `da/` | `SomaxDynamics`, `SomaxForwardModel`, `SubsampleObs`, `state_to_vector`, `make_ensemble` | `sesmax_core.da`, behind `[da]` | `SomaxDynamics` → `FilteraxDynamics`, `SomaxForwardModel` → `VardaxForwardModel`; `SubsampleObs` stays until vardax#81 lands |
| `core/flows.py` | `StateAffineBijection`, `to_flowjax`, `learned_prior` | `sesmax_core.flows`, behind `[flows]` | none |
| `io/` | `state_to_dataset`, `dataset_to_state`, Zarr save and load, scale metadata | `sesmax_core.io`, behind `[xarray]` | none |
| `core/checkpoint.py` | `SimulationCheckpointer` (orbax) | `sesmax_experiment.checkpoint`, behind its `[checkpoint]` | moved in sesmax#197; versioned and resumable since sesmax#207 |
| `models/lorenz63.py`, `lorenz96.py`, `lorenz96t.py` | the Lorenz systems on `Model` | `sesmax_core.testing` | none |
| `models/pde1d/`, `models/pde2d/` | Burgers, diffusion, linear and nonlinear convection, Navier–Stokes, Poisson | `sesmax_core.testing` | none (SX-3) |
| `models/base.py` | `DynamicalSystem`, marked for removal | dropped | — |
| `core/resolution.py` | Munk, Stommel and deformation-radius resolution checks | `sesmax-ocean` (C2) | ocean boundary layers |
| `eval/` | RMS divergence, enstrophy, KE, geostrophic imbalance, QG balance residual | `sesmax-ocean` (C2) | see the table above |
| `models/swm/linear_1d.py`, `nonlinear_1d.py`, `linear_2d.py`, `linear_swm_terms.py`, `gfd_testcases.py` | 1D linear and nonlinear shallow water, 2D linear shallow water (monolithic and term-based), gravity-wave, inertial-oscillation and geostrophic-adjustment cases | `sesmax_core.testing` | none (SX-3) |
| `models/_diagnostic_ops.py` | finitevolX pipekit diagnostics shared by the C-grid models | `sesmax_core` (private until C2 needs a public name) | none |
| `models/qg/`, `swm/nonlinear_2d.py`, `swm/multilayer.py`, `spherical/` | the ocean models | `sesmax-ocean` (C2) | — |
| `cli/` runner (`_run`, `_progress`, `spec`, `_units`, `_data`, `_assertions`) | the `somax-sim` run loop, run specs, units, progress logs | `sesmax_experiment.run` and `.sim`, behind its `[sim]` | the runner learns models through `Component`, which each package registers |
| `cli/models_registry/`, `cli/scenarios/`, `configs/`, DVC | the registered ocean models, scenarios (double gyre, Gulf Stream, …), run configs | `sesmax-ocean` (C2) | — |

`sesmax_core.testing` holds about 3,960 lines of toy models (counted with `wc -l` at `9cb19b3`): the Lorenz systems, the PDE test problems, and the linear shallow-water models that steps 14 to 16 of somax's tutorial series use.
They are standard test problems (numerical analysis, chaos, linear geophysical waves) rather than Earth-system components, they import only the core and finitevolX, and the core's own tests for solvers, scaling, rollouts and operators run on them.

### Renames

The renames are the ones that carry the old package name; everything else keeps its name, so a somax user's code changes only its imports.

| somax | sesmax-core |
| --- | --- |
| `somax.SomaxModel` | `sesmax_core.Model` |
| `somax.operators.SomaxModelOp` | `sesmax_core.operators.ModelOp` |
| `somax.da.SomaxDynamics` | `sesmax_core.da.FilteraxDynamics` |
| `somax.da.SomaxForwardModel` | `sesmax_core.da.VardaxForwardModel` |
| `somax.core.*` | `sesmax_core.*`, `sesmax_core.terms`, `sesmax_core.forcing`, `sesmax_core.scales`, `sesmax_core.gfd` |

No deprecated aliases: nothing outside somax imports these names (a grep of `geo_models`, `geo_stack` and `geo_ml` finds no `import somax`; plumax only lists it in a dependency-rules test), and somax keeps its own names until it is archived.

### What changed since the plan of 30 October

| Plan of 30 October | Today | Change |
| --- | --- | --- |
| finitevolX for grids and stencils | gridax: one `GridOps` per model, and the functions | sesmax#135, sesmax#146 |
| states of arrays, with `mask_locations` | states of arrays or `gridax.CField`s, ghosts from `gridax.fill_ghosts` | sesmax#138 |
| `sesmax_core.monitor` | `sesmax_experiment.monitor` | sesmax#197 |
| `sesmax_core.checkpoint`, behind `[checkpoint]` | `sesmax_experiment.checkpoint`, versioned since sesmax#207 | sesmax#197 |
| `sesmax_core.sim`, behind `[sim]` | `sesmax_experiment.run` and `sesmax_experiment.sim` | sesmax#198, sesmax#203 |
| a registry each component fills | `Component` in the core, found through the entry point `sesmax.components` | sesmax#197, sesmax#203 |
| the Helmholtz caches in `gfd` | removed on 5 November, as no model used them | — |
| constants copied or left in somax | geophysix | sesmax#124 |
| `imex_solver` in the core | integrix, re-exported, with `stable_step` | sesmax#118 |
| the inversions in finitevolX | `sesmax_core.gfd` | sesmax#129 |
| the vertical modes and the barotropic filter in finitevolX | `sesmax_core.vertical` | sesmax#122 |
| a forcing sampled by diffrax's interpolations | any functax 1-D scheme, which reports its step times | sesmax#141 |
| validators and field helpers private | `sesmax_core.validation`, `sesmax_core.fields` | sesmax#163 |
| two tiers, core and components | `sesmax-experiment` beside the ocean at tier 1; [the workspace overview](sesmax_workspace.md) proposes `sesmax-param` and `sesmax-testing` | sesmax#197 |

Two fixes after the move changed what every model returns: `Model.integrate` compiles once across calls (sesmax#196), and the states a run returns have their boundary conditions (sesmax#201).

### Phases and gates

Each phase ships as its own PR (stacked, in this repository) and merges only when its gate passes.

| Phase | Scope | Gate |
| --- | --- | --- |
| C0 Scaffold (done) | adopt the template as the workspace above; `sesmax-core` and the umbrella `sesmax` as empty packages with `__version__`; CI, release-please, docs and agent files for a workspace | `uv sync --all-packages --all-groups`, ruff, ty, the public-API and capability tests and `make docs` all pass; `git grep -e mypackage -e pypackage_template` finds nothing |
| C1 Core (done, 0.1.0 on 10 November) | copy the modules above from somax `9cb19b3`; the renames; the 22 whole and 7 split test files; the four extras; API pages and the capability index | every moved somax test passes with only import and rename changes; a bare `pip install ./packages/sesmax-core` imports `sesmax_core` with none of xarray, orbax, filterax, vardax or flowjax installed; an import-boundary test fails if `sesmax_core` imports `sesmax_ocean` or `somax`; coverage of `sesmax_core` passes the low starting gate of 30 % (SX-6) |
| C2 Ocean (done, with a different gate) | `sesmax-ocean` from the ocean models, presets, `resolution`, `eval`, the runner and configs; its own design page first | every remaining somax test passes under `sesmax-ocean` with only import changes; the DUACS sea-surface-height benchmark from the esmax plan runs as a baseline |
| C3 Levels (open, sesmax#29) | L1 in `_src/array.py`, `Step` and `simulate` behind `sesmax-core[geo]`, as geostack's SL-O2 decides | primitive modules import no carrier; `simulate` exposes `model.l1` |

The gate of C2 was replaced by equality with somax's numbers, because the core had moved to gridax; [the ocean design](sesmax_ocean.md) has it, and the DUACS baseline is sesmax#236. The runner went to `sesmax-experiment` and not to `sesmax_core.sim`.
C0 and C1 change no science.
Releases follow release-please: `sesmax-core` starts at 0.1.0, and the umbrella's first release waits for `sesmax-ocean`.

### Decisions of the move

All six were settled on 30 Oct 2026.

| ID | Question | Decision | Consequence |
| --- | --- | --- | --- |
| SX-1 | What `sesmax` installs | a code-free umbrella; `sesmax-core` is the core | the umbrella's first release waits for `sesmax-ocean` |
| SX-2 | How the code arrives | a copy at somax `9cb19b3`, with the source commit named in each C1 commit message and in the CHANGELOG | history and `git blame` stay in the archived somax repository |
| SX-3 | Where the toy models live | `sesmax_core.testing`: Lorenz, the PDE test problems and, decided later the same day, the 1D shallow-water and 2D linear shallow-water models (the models of steps 1 to 16 of somax's tutorial series) | 27 somax test files move to the core whole; the step tutorials move with the models |
| SX-4 | Where the `somax-sim` runner lives | first leaning towards `sesmax_core.sim`; changed on 25 November to its own package, `sesmax-experiment`, with each component registering through `Component` | the runner names no ocean model, and the core carries no run loop |
| SX-5 | Spatial bases | stay on `geonnax.basis`, changed on 4 Nov 2026 from taking vardax's, whose constructors delegate to it | the core's base install keeps geonnax and needs no vardax. functax has the bases since, and sesmax#114 moves the core onto it |
| SX-6 | Coverage gate | very low to start: `fail_under = 30` per package, raised in follow-ups | the template's 90 % is a target, not a C1 gate |

### A core for other repositories

On 5 November 2026 we asked whether the core could sit below every repository of the stack that steps in time, with plumax as its second consumer. Two things came of it, and one decision replaced its plan.

**A model that is not an ocean fits the contract.** plumax's transport model, `EulerianDispersionRHS`, was rebuilt as a `TermModel` of advection, diffusion and a source on plumax's own stencils, on a 40 × 16 × 8 grid with a source that varies in time.

| Check | Result |
| --- | --- |
| largest difference in the tendency | 0, against a largest tendency of $3.8 \times 10^{-4}$ |
| largest difference after a Heun run to 200 s | $1.8 \times 10^{-12}$, against a largest concentration of $7.8 \times 10^{-6}$ |
| the gradient of the mass with respect to the source, through the solver | $1.054564 \times 10^{-3}$, against a central difference of $1.054560 \times 10^{-3}$ |
| tagging diffusion `implicit` | `build_terms` returns a `MultiTerm`, with no other change |

One thing did not map: advection and diffusion need different ghost values at a Dirichlet boundary, so each term fills its own and the model's `apply_boundary_conditions` is a pass-through. The contract allows that since.

**Six pieces were added, each where it belongs.** An audit of every definition in the stack found little copied and several things written twice under other names.

| ID | Piece | Where it is today |
| --- | --- | --- |
| SC-1 | `Model.as_forward_model`, over `DiffraxForwardModel` | the core, sesmax#85; the class is integrix's |
| SC-2 | `HatInTime`, a piecewise-linear basis in time | the core, sesmax#86; functax#179 takes the windows upstream |
| SC-3 | tracer transport terms in the core | withdrawn: they are plumax's physics |
| SC-4 | `stable_step` | integrix, re-exported by the core, sesmax#87 |
| SC-5 | `WindowOperator`, a control through a whole run to the observations | `sesmax_core.da`, sesmax#88 |
| SC-6 | a periodic seam that conserves | gridax#403 |
| SC-7 | vertical advection and diffusion of a 3-D field | gridax#404 |

**plumax and radtrax did not join the workspace.** Decided by the owner on 26 November: they stay in their own repositories and meet sesmax at three contracts, the model (`TermModel`), the known field in time (`Covariates`) and the estimation `Problem` with its observation operator. geostack#328 holds that work, and geostack#326 is the later audit of what each library owns.

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| SC-1 | What stays in the core after the split? | the contract, terms, scales, fields, the shared physics, the adapters, `Component`, `Covariates` | the core as it is |
| SC-2 | Do `gfd` and `vertical` belong here? | yes, while two components need them | gridax for the inversions, as they are built from the grid |
| SC-3 | The adjoint of `rollout` | an integrix spec | a diffrax adjoint |
| SC-4 | The adapters, here or in `sesmax-experiment`? | here: they are the contract seen from a method | there, with the problem that uses them |
| SC-5 | The moved names | keep the spelling, no alias left behind | deprecated aliases for one release |
| SC-6 | A model in other coordinates, here? | yes, `TransformedModel` beside `ScaledModel` | leave it to filterax's `EncodedDynamics`, which covers a filter and not an integration or a gradient |
| SC-7 | What does an observable return? | a located field, with its grid and mask beside it | a bare array and the name of its point |

## References

- Ascher, U. M., Ruuth, S. J. and Spiteri, R. J. (1997). [Implicit-explicit Runge–Kutta methods for time-dependent partial differential equations](https://hdl.handle.net/10.1016/S0168-9274(97)00056-1). *Applied Numerical Mathematics* 25, 151–167.
- Barba, L. A. and Forsyth, G. F. (2018). [CFD Python: the 12 steps to Navier–Stokes equations](https://doi.org/10.21105/jose.00021). *Journal of Open Source Education* 1, 21.
- Benner, P., Gugercin, S. and Willcox, K. (2015). [A survey of projection-based model reduction methods for parametric dynamical systems](https://doi.org/10.1137/130932715). *SIAM Review* 57, 483–531.
- Griewank, A. and Walther, A. (2008). [Evaluating Derivatives](https://doi.org/10.1137/1.9780898717761). SIAM.
- Kidger, P. (2022). [On neural differential equations](https://arxiv.org/abs/2202.02435). DPhil thesis, University of Oxford.
- Le Dimet, F.-X. and Talagrand, O. (1986). [Variational algorithms for analysis and assimilation of meteorological observations: theoretical aspects](https://doi.org/10.3402/tellusa.v38i2.11706). *Tellus A* 38, 97–110.
- Lorenz, E. N. (1963). [Deterministic nonperiodic flow](https://hdl.handle.net/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2). *Journal of the Atmospheric Sciences* 20, 130–141.
- Tsitouras, C. (2011). [Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption](https://doi.org/10.1016/j.camwa.2011.06.002). *Computers & Mathematics with Applications* 62, 770–775.
- Vallis, G. K. (2017). [Atmospheric and Oceanic Fluid Dynamics](https://doi.org/10.1017/9781107588417), 2nd ed. Cambridge University Press.

## Relationships

- Part of: [the workspace packages](sesmax_workspace.md), tier 0
- Builds on: equinox, diffrax, integrix, paramax, gridax, gaussx, functax, geonnax, geophysix, pipekit
- Used by: [sesmax-ocean](sesmax_ocean.md), [sesmax-testing](sesmax_testing.md), [sesmax-param](sesmax_param.md), [sesmax-experiment](sesmax_experiment.md)
- Issues: sesmax#27, sesmax#29, sesmax#105, sesmax#114, sesmax#202
