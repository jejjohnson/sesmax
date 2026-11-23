<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/packages/sesmax-core/logo-dark.svg">
  <img alt="sesmax-core" src="../../docs/assets/packages/sesmax-core/logo-light.svg" width="430">
</picture>

**The shared core of simple Earth-system models for JAX: states, terms, forcing, rescaling, rollouts and data-assimilation adapters, on Equinox, diffrax, gridax and pipekit.**

[![Tests](https://github.com/jejjohnson/sesmax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/sesmax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/sesmax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/sesmax/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/sesmax/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

📖 [Documentation](https://jejjohnson.github.io/sesmax/) ·
📓 [Tutorials](https://jejjohnson.github.io/sesmax/steps-to-navier-stokes/) ·
📚 [API reference](https://jejjohnson.github.io/sesmax/reference/) ·
🗂️ [Capability index](https://jejjohnson.github.io/sesmax/reference/capabilities/) ·
🧭 [Design](https://jejjohnson.github.io/sesmax/workspace-and-core/)

</div>

---

An assimilation window is hours to weeks long, and the model inside it runs many times: once per ensemble member, or once per iteration of a minimiser which also needs the gradient through the whole window. So the model has to be cheap, it has to be differentiable, and we have to be able to swap a piece of it (a drag, a closure, a forcing with a prior) without rewriting the rest. sesmax-core treats the right-hand side of the model as the one object, written as a sum of terms,

```math
\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \sum_{j=1}^{J} c_j f_j(\mathbf{x}, t; \boldsymbol{\theta})
```

where $`\mathbf{x} \in \mathbb{R}^{D_x}`$ is the state (a pytree of fields, flattened), $`f_j: \mathbb{R}^{D_x} \times \mathbb{R} \to \mathbb{R}^{D_x}`$ is one term (advection, Coriolis, a forcing), $`c_j \in \mathbb{R}`$ is its coefficient, $`J`$ is the number of terms and $`\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}`$ holds the parameters. The terms know nothing about which part of the Earth system they describe, which is what lets the ocean component and the ones after it share this package.

```python
# Shapes: Ny = Nx = 18 cells with the ghost ring; D_w = 4 sources; T = 36 steps
# (base, forcing and rest as in the quickstart below)

# today (sesmax-core)
# dx/dt = ∑ⱼ cⱼ fⱼ(x, t; θ), one more term
terms = base.terms + ForcingTerm(forcing, place=add_to("h"))
# today (sesmax-core)
model = LinearSWM2DTermModel(terms=terms, grid=base.grid)
# today (sesmax-core)
# x_t = step(x_{t−1}), t = 1 … T
# h: → (T, Ny, Nx)
trajectory = rollout(model, rest, dt=600.0, n_steps=36)
```

Most of the numerics belongs to other libraries, and this package only composes them. A model is an [`equinox`](https://github.com/patrick-kidger/equinox) module and time stepping is [`diffrax`](https://github.com/patrick-kidger/diffrax), so any diffrax solver goes into `Model.integrate`. The C-grid stencils and the elliptic solves are [`gridax`](https://github.com/jejjohnson/gridax), the spatial bases of a forcing are [`geonnax`](https://github.com/jejjohnson/geonnax), constrained parameters are [`paramax`](https://github.com/danielward27/paramax) wrappers, a rollout is a [`pipekit`](https://github.com/jejjohnson/pipekit) scan, and the adapters hand a model to [`filterax`](https://github.com/jejjohnson/filterax) and [`vardax`](https://github.com/jejjohnson/vardax).

## 🌟 Highlights

- ➕ **Terms that add up**: `Term`s compose with `+`, `-`, `*` and `@`, and every coefficient is a JAX leaf, so `jax.grad` reaches a viscosity or a drag like any other parameter.
- ⏱️ **IMEX from a label**: mark a term with `implicit`, and `build_diffrax_terms` gives diffrax the explicit and implicit parts for an IMEX solver.
- 🌬️ **Forcing as a control with a prior**: `BasisForcing` is a fixed space-time dictionary times a few coefficients, each with a prior standard deviation, with eight `spatial_from_*` constructors for the dictionary.
- 🔒 **Parameters that stay physical**: `positive`, `interval` and `frozen` keep a parameter in bounds or out of the gradient, and the model unwraps them before its tendency sees them.
- 📏 **Rescaling without touching the physics**: `ScaledModel` integrates any model in nondimensional or standardised coordinates, through an affine map which is exact.
- 🛰️ **One model, several assimilation methods**: the same model drives filterax's ensemble filters and vardax's variational methods, or a forcing inversion through `WindowOperator`.
- 🧪 **Test problems included**: Lorenz-63 and Lorenz-96, the PDEs on the way to Navier–Stokes, and linear and nonlinear shallow water are in `sesmax_core.testing`.

## 📐 Five uses of one right-hand side

Write $`f(\mathbf{x}, t; \boldsymbol{\theta})`$ for the whole sum above. Everything in the package either builds $`f`$ or uses it.

| Use | Equation | Names |
|---|---|---|
| Integrate it | $`\mathbf{x}(t_1) = \mathbf{x}(t_0) + \int_{t_0}^{t_1} f(\mathbf{x}, t; \boldsymbol{\theta}) \, \mathrm{d}t`$ | `Model.integrate`, `Model.step`, `solve_pde` |
| Force it | $`u(\mathbf{s}, t) = \sum_{m} w_m \, b_m(t) \, \phi_m(\mathbf{s})`$ | `BasisForcing`, `ForcingTerm`, `spatial_from_*` |
| Rescale it | $`\frac{\mathrm{d}\tilde{\mathbf{x}}}{\mathrm{d}\tilde{t}} = \tau \, \boldsymbol{\sigma}^{-1} \odot f(\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \tilde{\mathbf{x}}, \tau \tilde{t}; \boldsymbol{\theta})`$ | `Scales`, `StateAffine`, `ScaledModel` |
| Fit it | $`\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{T} \sum_{t=1}^{T} \ell(\mathbf{x}_t(\boldsymbol{\theta}), \mathbf{x}_t^\text{ref})`$ | `rollout`, `rollout_loss` |
| Assimilate with it | $`\mathcal{J}(\mathbf{v}) = \frac{1}{2} \Vert \mathbf{v} \Vert^2 + \frac{1}{2} \sum_{t=1}^{T} \Vert \mathbf{y}_t - \mathbf{H} \mathbf{x}_t(\mathbf{v}) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}}`$ | `sesmax_core.da` |

**Forcing.** The forcing field $`u`$ at the point $`\mathbf{s}`$ is a sum over $`D_w`$ atoms, each with a spatial shape $`\phi_m`$, a gate in time $`b_m`$ and a coefficient $`w_m \in \mathbb{R}`$. The coefficients are the only control, and the rest of the dictionary is fixed. `ForcingTerm` writes the field onto one component of the tendency, so a forcing is a term like any other.

**Rescaling.** The state is $`\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \tilde{\mathbf{x}}`$ with $`\boldsymbol{\mu}, \boldsymbol{\sigma} \in \mathbb{R}^{D_x}`$, and the time is $`t = \tau \tilde{t}`$ with $`\tau > 0`$. The pair $`(\boldsymbol{\mu}, \boldsymbol{\sigma})`$ comes from physical scales (`StateAffine.from_scales`) or from samples (`StateAffine.from_samples`).

**Fitting and assimilation.** Here $`T`$ is the number of snapshots, $`\mathbf{x}_t^\text{ref} \in \mathbb{R}^{D_x}`$ a reference trajectory, $`\ell`$ a loss per snapshot, $`\mathbf{y}_t \in \mathbb{R}^{D_y}`$ the observations with error covariance $`\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}`$, $`\mathbf{H} \in \mathbb{R}^{D_y \times D_x}`$ the observation operator and $`\mathbf{v}`$ a whitened control (an initial state, a forcing or a parameter, divided by its prior standard deviation about a zero prior mean). The methods that minimise $`\mathcal{J}`$ and the ensemble filters live in vardax and filterax. This package holds only the adapters that let them call a model.

## 📥 Install

sesmax-core is not on PyPI, and its git dependencies are pinned, so install it from the repository. It needs Python 3.12 or newer and brings in JAX, Equinox, diffrax, lineax, optimistix, einops, paramax, gridax, spectraldiffx, geonnax, geophysix, integrix and pipekit.

```bash
pip install "sesmax-core @ git+https://github.com/jejjohnson/sesmax.git#subdirectory=packages/sesmax-core"
```

Four extras add the optional parts: `da` (filterax and vardax, for `sesmax_core.da`), `flows` (flowjax, for `sesmax_core.flows`), `xarray` (xarray and zarr, for `sesmax_core.io`) and `checkpoint` (orbax, for `SimulationCheckpointer`). An optional module imported without its extra raises an `ImportError` that names it.

```bash
pip install "sesmax-core[da] @ git+https://github.com/jejjohnson/sesmax.git#subdirectory=packages/sesmax-core"
```

The packages of the stack pin each other at different tags. When uv reports a conflict, copy the `override-dependencies` of the [workspace `pyproject.toml`](../../pyproject.toml) into your own project.

## 🌊 Quickstart: which sources raised the sea surface?

A layer of water at rest on a rotating plane is raised by four sources of mass, and we only see its surface height at a few cells. The state is $`\mathbf{x} = (h, u, v)`$, the height anomaly and the two velocities of linear shallow water. The source field is $`\boldsymbol{\Phi} \mathbf{w}`$, where $`\boldsymbol{\Phi} \in \mathbb{R}^{N_y N_x \times D_w}`$ holds one Gaussian bump of width $`\ell`$ per column (centred at $`\mathbf{s}_j`$, evaluated at the cell centres $`\mathbf{s}_i`$) and $`\mathbf{w} \in \mathbb{R}^{D_w}`$ the strengths, with prior $`\mathcal{N}(\mathbf{0}, \sigma_w^2 \mathbf{I})`$. The observations are $`\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \boldsymbol{\varepsilon}_t`$, where $`\mathbf{H}`$ picks the observed cells and $`\boldsymbol{\varepsilon}_t`$ has standard deviation $`\sigma_\varepsilon`$. The viscosity is $`\nu`$ (zero here), the bottom drag is $`\kappa`$, and we minimise the last row of the table over the whitened strengths $`\mathbf{v} = \mathbf{w} / \sigma_w`$, which gives $`\hat{\mathbf{v}}`$ and $`\hat{\mathbf{w}} = \sigma_w \hat{\mathbf{v}}`$.

```python
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from sesmax_core import BasisForcing, ConstantInTime, ForcingTerm, SpatialBasis, add_to
from sesmax_core.testing import LinearSW2DState, LinearSWM2DTermModel
from sesmax_core.train import rollout


jax.config.update("jax_enable_x64", True)

# Shapes: Ny = Nx = 18 cells with the ghost ring; D_w = 4 sources; T = 36 steps;
#         4 × 4 observed cells

# today (sesmax-core)
# ∂ₜ(h, u, v) = gravity wave + Coriolis + ν diffusion − κ drag
base = LinearSWM2DTermModel.create(nx=16, ny=16, Lx=1e6, Ly=1e6, bottom_drag=1e-5)

# the cell centres along each side, in metres
# → (Nx,)
axis = (jnp.arange(18) - 0.5) * base.grid.dx
# → (Ny, Nx) each
y_cell, x_cell = jnp.meshgrid(axis, axis, indexing="ij")
# s_j, the source centres in metres
# → (D_w, 2)
centres = 1e6 * jnp.array([[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]])
# s_i, the cell centres
# → (Ny Nx, 2)
cells = jnp.stack([x_cell.ravel(), y_cell.ravel()], axis=-1)
# ‖s_i − s_j‖²
# → (Ny Nx, D_w)
sq_dist = jnp.sum((cells[:, None] - centres) ** 2, axis=-1)
# Φ_ij = exp(−½ ‖s_i − s_j‖² / ℓ²), with ℓ = 10⁵ m
# → (Ny Nx, D_w)
dictionary = jnp.exp(-0.5 * sq_dist / 1e5**2)
# today (sesmax-core)
# w ~ 𝒩(0, σ_w² I), with σ_w = 10⁻⁵ m s⁻¹
spatial = SpatialBasis.from_array(dictionary, std=1e-5 * jnp.ones(4))

# today (sesmax-core)
# x_0 = 0, a layer at rest
zeros = jnp.zeros((18, 18))
rest = LinearSW2DState(h=zeros, u=zeros, v=zeros)


# user
# f(x; w) = base + (Φ w, 0, 0), a source of mass in h
def build_model(coeffs: Float[Array, " source"]) -> LinearSWM2DTermModel:
    # today (sesmax-core)
    forcing = BasisForcing(coeffs, spatial, ConstantInTime(m=4), grid_shape=(18, 18))
    # today (sesmax-core)
    terms = base.terms + ForcingTerm(forcing, place=add_to("h"))
    return LinearSWM2DTermModel(terms=terms, grid=base.grid)


# user
# H x_t, the surface height at every fourth interior cell
def observe(coeffs: Float[Array, " source"]) -> Float[Array, "time y x"]:
    # today (sesmax-core)
    trajectory = rollout(build_model(coeffs), rest, dt=600.0, n_steps=36)
    return trajectory.h[:, 1:-1:4, 1:-1:4]


# w_true, in m s⁻¹
# → (D_w,)
true_coeffs = 1e-5 * jnp.array([1.0, -0.5, 0.3, 0.8])
# ε_t ~ 𝒩(0, σ_ε² I), with σ_ε = 3 mm
# → (T, 4, 4)
noise = 3e-3 * jax.random.normal(jax.random.key(0), (36, 4, 4))
# y_t = H x_t + ε_t
# → (T, 4, 4)
obs = observe(true_coeffs) + noise


# user
# 𝒥(v) = ½ ‖v‖² + ½ ∑ₜ ‖y_t − H x_t(σ_w v)‖² / σ_ε²
def cost(white: Float[Array, " source"], args: None) -> Float[Array, ""]:
    # today (sesmax-core)
    # w = σ_w v
    coeffs = spatial.prior_std() * white
    misfit = (obs - observe(coeffs)) / 3e-3
    return 0.5 * jnp.sum(white**2) + 0.5 * jnp.sum(misfit**2)


# today (optimistix)
# v̂ = argmin 𝒥(v), by gradients through the whole window
solution = optx.minimise(cost, optx.BFGS(rtol=1e-8, atol=1e-8), jnp.zeros(4))
# today (sesmax-core)
# ŵ = σ_w v̂
# → (D_w,)
estimate = spatial.prior_std() * solution.value

print((1e5 * estimate).round(2))
```

This prints `[ 0.99 -0.48  0.34  0.73]`: the recovered strengths in $`10^{-5}`$ m s⁻¹, where the truth is `1.0, -0.5, 0.3, 0.8`, from 16 noisy cells over a six-hour window. The minimiser saw only $`\mathcal{J}`$ and its gradient through the 36 steps. Nothing here is specific to shallow water, so swapping `base` for another model, or `SpatialBasis.from_array` for one of the `spatial_from_*` bases, leaves the rest as it is.

The [data-assimilation notebooks](https://jejjohnson.github.io/sesmax/data-assimilation-4dvar/) take this further with vardax and filterax instead of a hand-written cost, and the [composable models notebook](https://jejjohnson.github.io/sesmax/composable-models/) builds a model from its terms.

## 🔬 What else is in the box

Every name below is public in the module named beside it. The [capability index](https://jejjohnson.github.io/sesmax/reference/capabilities/) lists all of them, together with the ones in the libraries underneath.

| Module | Symbols | Notes |
|---|---|---|
| `sesmax_core` | `Model`, `TermModel`, `State`, `Params`, `PhysConsts`, `Diagnostics`, `positive`, `interval`, `frozen`, `trainable_mask`, `solve_pde` | The model contract, the types and the constraints |
| `sesmax_core.terms` | `Term`, `TermFn`, `Sum`, `Scaled`, `Compose`, `OperatorTerm`, `explicit`, `implicit`, `partition`, `build_diffrax_terms` | The term algebra and the IMEX split |
| `sesmax_core.forcing` | `BasisForcing`, `ForcingTerm`, `SpatialBasis`, `FourierInTime`, `HatInTime`, `InterpolatedForcing`, `WindStress2D`, `QuadraticDrag2D`, `RayleighDamping2D` and the rest | Reduced-order forcing, tabulated forcing, and wind, drag and relaxation operators |
| `sesmax_core.scales` | `Scales`, `StateAffine`, `ScaledModel` | Nondimensional and standardised models |
| `sesmax_core.solvers` | `imex_solver`, `imex_stepsize_controller`, `stable_step` | Stiff stepping and the explicit step bound, from integrix |
| `sesmax_core.train` | `rollout`, `rollout_loss`, `state_mse`, `ModelStep` | One scan over a window, with checkpointing |
| `sesmax_core.da` | `FilteraxDynamics`, `VardaxForwardModel`, `WindowOperator`, `make_ensemble`, `state_to_vector`, `SubsampleObs` | Needs `[da]`; `VardaxForwardModel` is exact only for autonomous models |
| `sesmax_core.gfd`, `sesmax_core.vertical` | `ModalTransform`, `StratificationProfile`, `pv_inversion`, `streamfunction_from_vorticity`, `pressure_from_divergence`, `barotropic_filter` | Layered fluids: vertical modes and the elliptic inversions |
| `sesmax_core.guards`, `sesmax_core.monitor` | `guard_finite`, `guard_positive`, `guard_ceiling`; `BaseMonitor`, `default_monitors` and the built-in monitors | Guards stop inside a step; the run loop that calls the monitors is not in the package yet |
| `sesmax_core.operators` | `ModelOp`, `OperatorTerm` | pipekit operators in and out of a model |
| `sesmax_core.domain` | `Domain`, `TimeDomain` | Spatial and time domains |
| `sesmax_core.io`, `sesmax_core.checkpoint`, `sesmax_core.flows` | `state_to_dataset`, `save_dataset`; `SimulationCheckpointer`; `to_flowjax`, `learned_prior` | Need `[xarray]`, `[checkpoint]` and `[flows]` |
| `sesmax_core.testing` | `Lorenz63`, `Lorenz96`, `Burgers2D`, `IncompressibleNS2D`, `LinearSWM2DTermModel`, `NonlinearShallowWater1D` and the rest | Toy models for tests, tutorials and examples |

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Steps to Navier–Stokes](https://jejjohnson.github.io/sesmax/steps-to-navier-stokes/) | Thirteen notebooks, from linear convection to channel flow |
| 2 | [Chaos and predictability](https://jejjohnson.github.io/sesmax/chaos-and-predictability/) | The Lorenz systems |
| 3 | [Elliptic inversions](https://jejjohnson.github.io/sesmax/elliptic-inversions/) | Streamfunction, pressure and potential-vorticity inversions |
| 4 | [Geostrophic adjustment](https://jejjohnson.github.io/sesmax/step16-geostrophic-adjustment/) | Rotating shallow water, the last of three notebooks |
| 5 | [Composable models](https://jejjohnson.github.io/sesmax/composable-models/) | A model assembled from its terms |
| 6 | [ETKF](https://jejjohnson.github.io/sesmax/data-assimilation-etkf/) and [4D-Var](https://jejjohnson.github.io/sesmax/data-assimilation-4dvar/) | Data assimilation with filterax and vardax |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/sesmax/)**: the home page, the guides and the executed notebooks.
- **[API reference](https://jejjohnson.github.io/sesmax/reference/)**: one page per public module, generated from the docstrings.
- **[Capability index](https://jejjohnson.github.io/sesmax/reference/capabilities/)**: every public name once, with a one-line summary.
- **[Workspace and core design](https://jejjohnson.github.io/sesmax/workspace-and-core/)**: what moved from somax, the phases and the gates.

## 🧩 Ecosystem

sesmax-core is tier 0 of the [sesmax workspace](https://github.com/jejjohnson/sesmax), in the GeoModels layer of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the state types, the term algebra, the forcing and the adapters, and composes the rest in:

| Concern | Owner |
|---|---|
| Modules and filtered transforms | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Time integration | [`diffrax`](https://github.com/patrick-kidger/diffrax) |
| IMEX solver, step control and the step bound | [`integrix`](https://github.com/jejjohnson/integrix) |
| Grids, masks, C-grid operators and elliptic solves | [`gridax`](https://github.com/jejjohnson/gridax) |
| Spectral solves (the Navier–Stokes test problem) | [`spectraldiffx`](https://github.com/jejjohnson/spectraldiffx) |
| Spatial bases of a forcing | [`geonnax`](https://github.com/jejjohnson/geonnax) |
| Constrained parameters | [`paramax`](https://github.com/danielward27/paramax) |
| Physical constants | [`geophysix`](https://github.com/jejjohnson/geophysix) |
| Operators, rollouts and cycles | [`pipekit`](https://github.com/jejjohnson/pipekit) |
| Ensemble filters (the `da` extra) | [`filterax`](https://github.com/jejjohnson/filterax) |
| Variational assimilation (the `da` extra) | [`vardax`](https://github.com/jejjohnson/vardax) |
| Normalising flows (the `flows` extra) | [`flowjax`](https://github.com/danielward27/flowjax) |

Imports point down the tiers: a component such as `sesmax-ocean` (planned) will import the core, and the core imports no component. No other project imports it yet.

## 🌳 Repository layout

```
packages/sesmax-core/
├── src/sesmax_core/
│   ├── __init__.py    ← the contract, the types and the most-used names
│   ├── terms.py       ← public modules, each re-exporting from _src/
│   ├── forcing.py
│   ├── …
│   └── _src/          ← the implementation
├── tests/             ← one file or folder per module
└── CHANGELOG.md
```

## ⚙️ Development

The commands run from the workspace root:

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
make install    # every package and dependency group, plus the pre-commit hooks
make test       # every test and doctest
make lint       # ruff on the whole repository
make typecheck  # ty on packages/*/src and scripts
```

[`AGENTS.md`](../../AGENTS.md) holds the rules for contributors and coding agents: the tiers, the contracts of a model, a term and a forcing, and the tests that enforce them.

## 🎓 References

- Lorenz (1963). [Deterministic nonperiodic flow](https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2). *Journal of the Atmospheric Sciences*.
- Lorenz (1996). Predictability: a problem partly solved. *Proceedings of the Seminar on Predictability*, ECMWF.

## 📄 License

MIT. See [LICENSE](LICENSE).
