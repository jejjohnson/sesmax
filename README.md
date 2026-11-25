<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="sesmax" src="docs/assets/logo-light.svg" width="340">
</picture>

**Simple Earth-system models for JAX: cheap, differentiable priors for short-window data assimilation, on Equinox, diffrax, gridax and pipekit.**

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

Observations of the Earth are sparse: a satellite sees a strip of the ocean, then comes back days later. What fills the gaps is a model, which carries information from one observation to the next. Inside an assimilation window of hours to weeks that model is a prior, so we judge it as one: it has to be cheap enough to run hundreds of times, differentiable through the whole window, and open to correction by the data. sesmax is a workspace of such models, with a shared core, one package per component of the Earth system, and later a coupler on top. They all go into the same cost,

```math
\mathcal{J}(\mathbf{x}_0) = \frac{1}{2} \Vert \mathbf{x}_0 - \boldsymbol{\mu}_x \Vert^2_{\boldsymbol{\Sigma}_x^{-1}} + \frac{1}{2} \sum_{t \leq T} \Vert \mathbf{y}_t - \mathbf{H}_t \mathbf{x}_t(\mathbf{x}_0) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}}
```

where $`\mathbf{x}_0 \in \mathbb{R}^{D_x}`$ is the state at the start of the window, with prior mean $`\boldsymbol{\mu}_x \in \mathbb{R}^{D_x}`$ and covariance $`\boldsymbol{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}`$, $`\mathbf{x}_t(\mathbf{x}_0) \in \mathbb{R}^{D_x}`$ is the state after $`t`$ of the $`T`$ model steps, $`\mathbf{y}_t \in \mathbb{R}^{D_y}`$ are the observations with error covariance $`\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}`$, and $`\mathbf{H}_t \in \mathbb{R}^{D_y \times D_x}`$ picks what the instrument saw. A sesmax model supplies $`\mathbf{x}_t(\mathbf{x}_0)`$ and its gradient; the methods which minimise $`\mathcal{J}`$, and the ensemble filters which avoid the gradient, live in other libraries.

```python
# Shapes: D_x = 40 variables; T = 10 steps
# (model, truth, truth_state and background as in the quickstart below)

# today (sesmax-core)
# x_t = step(x_{t−1}; θ), for 4D-Var and for an ensemble filter
forward = VardaxForwardModel(model=model, template=truth_state, dt=0.05)
dynamics = FilteraxDynamics(model=model, template=truth_state)
# today (sesmax-core)
# ∇_θ ℒ(θ), through the whole window, in the tree of the model
reference = L96State(x=truth[1:])
state = L96State(x=background)
gradient = eqx.filter_grad(rollout_loss)(model, state, reference, dt=0.05)
```

Most of the numerics belongs to other libraries, and the workspace composes them. A model is an [`equinox`](https://github.com/patrick-kidger/equinox) module, time stepping is [`diffrax`](https://github.com/patrick-kidger/diffrax), the C-grid stencils are [`gridax`](https://github.com/jejjohnson/gridax) and the spectral solves [`spectraldiffx`](https://github.com/jejjohnson/spectraldiffx), a rollout is a [`pipekit`](https://github.com/jejjohnson/pipekit) scan, and the assimilation is [`vardax`](https://github.com/jejjohnson/vardax) (variational) or [`filterax`](https://github.com/jejjohnson/filterax) (ensemble). The workspace grows out of [somax](https://github.com/jejjohnson/somax), a simple ocean model in JAX.

## 🌟 Highlights

- 🛰️ **One model, several methods**: the same model goes to vardax's 4D-Var through `VardaxForwardModel` and to filterax's ensemble filters through `FilteraxDynamics`.
- ➕ **Models from terms**: a right-hand side is a sum of `Term`s, so we can swap a drag or a closure without rewriting the model, and `jax.grad` reaches every coefficient.
- 🌬️ **Forcings as controls**: `BasisForcing` writes a forcing as a few coefficients, each with a prior standard deviation, and `WindowOperator` inverts for them.
- 🔁 **Gradients through a window**: `rollout` is one scan over the window, and `rollout_loss` checkpoints it, so `jax.grad` reaches every parameter of the model.
- 🔒 **Parameters that stay physical**: `positive`, `interval` and `frozen` keep a parameter in bounds or out of the gradient.
- 🧪 **Test problems included**: the Lorenz systems, the PDEs on the way to Navier–Stokes, and linear and nonlinear shallow water are in `sesmax_core.testing`.
- 🧱 **Tiers that hold**: a package imports only from the tiers below it, and a test in the workspace fails when one does not.

## 📐 The model is the prior

We ask four things of every model here. They decide what goes in, so anything that makes the prior cheaper or better calibrated comes before anything that makes a long run more faithful.

| Requirement | Why | Today |
|---|---|---|
| Cheap | ensembles and repeated gradients | `rollout`, `make_ensemble`, `imex_solver` |
| Good where the observations look | it carries the information | a benchmark per component, with the ocean |
| Honest about its error | an overconfident prior ignores data | `BasisForcing`, `WindowOperator` |
| Learnable | data corrects the model | `rollout_loss`, `trainable_mask` |

**Cheap.** A minimiser of $`\mathcal{J}`$ runs the model and its adjoint at every iteration, and a filter runs the model once per member. Simple dynamics (shallow water, quasi-geostrophy, slab and bucket models) are enough over a window inside the predictability time of the scales they resolve, so we leave out full physics suites and multi-decade runs.

**Good where the observations look.** A model earns its place when it beats the best statistical prior (optimal interpolation, a Gaussian process) given the same observations. That test is a gate of each component; the first is the ocean's, on sea-surface height.

**Honest about its error.** A cheap model is wrong, so its error has to be part of the problem. Today we do it with an unknown forcing, whose coefficients have a prior and are estimated with the window.

**Learnable.** Parameters and forcing coefficients are leaves of the model, so a rollout loss or a cost like $`\mathcal{J}`$ can correct them instead of a person retuning them.

## 📥 Install

Nothing is on PyPI yet; the packages are installed from this repository with [uv](https://docs.astral.sh/uv/), and they need Python 3.12 or newer. From a clone, one command installs every package with its extras and the development tools, which is what the quickstart needs:

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
uv sync
```

In your own project, depend on `sesmax-core`. The libraries of the stack pin each other at different git tags, so a plain `pip install` of the git URL stops at a conflict; name the source, and copy the `override-dependencies` list of this repository's [`pyproject.toml`](pyproject.toml), which holds one tag for each:

```toml
[project]
dependencies = ["sesmax-core[da]"]

[tool.uv]
override-dependencies = [
    # copy the list from sesmax's root pyproject.toml
]

[tool.uv.sources]
sesmax-core = { git = "https://github.com/jejjohnson/sesmax.git", subdirectory = "packages/sesmax-core" }
```

The `da` extra adds filterax and vardax, which the quickstart uses. The umbrella `sesmax` has no release yet; it waits for the ocean. The other extras are in the [core's README](packages/sesmax-core/README.md#-install).

## 🌪️ Quickstart: one window of Lorenz-96, two methods

Lorenz-96 stands in for any model: a ring of $`D_x = 40`$ variables driven by a constant forcing $`F`$, chaotic at $`F = 8`$. We observe every second variable for $`T = 10`$ steps, as $`\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \boldsymbol{\varepsilon}_t`$ with noise of standard deviation $`\sigma_\varepsilon = 0.2`$, and we start from a background $`\boldsymbol{\mu}_x`$ whose error has standard deviation $`\sigma_x = 0.5`$. The truth starts on the attractor. Then the same model and the same observations go to 4D-Var, which finds the $`\hat{\mathbf{x}}_0`$ that minimises $`\mathcal{J}`$, and to an ensemble transform Kalman filter (ETKF), which moves $`N_E = 100`$ members toward each observation in turn.

```python
import einx
import jax
import jax.numpy as jnp
import lineax as lx
from filterax import ETKF
from jaxtyping import Array, Float
from vardax import Batch1D, MaskedIdentity, StrongFourDVar

from sesmax_core.da import (
    FilteraxDynamics,
    SubsampleObs,
    VardaxForwardModel,
    make_ensemble,
    state_to_vector,
)
from sesmax_core.testing import L96State, Lorenz96
from sesmax_core.train import rollout


jax.config.update("jax_enable_x64", True)

# Shapes: D_x = 40 variables; D_y = 20 observed; T = 10 steps; N_E = 100 members

# today (sesmax-core)
# dx_i/dt = (x_{i+1} − x_{i−2}) x_{i−1} − x_i + F
model = Lorenz96.create(F=8.0)
# today (sesmax-core)
# x_0, on the attractor after 20 time units
# x: → (1, D_x)
start = L96State.init_state(ndim=40, key=jax.random.key(0))
spin_up = model.integrate(start, t0=0.0, t1=20.0, dt=0.01).ys
truth_state = L96State(x=spin_up.x[-1])
# → (D_x,)
truth_start, _ = state_to_vector(truth_state)


# user
# x_t = step(x_{t−1}; θ), t = 0 … T
def forecast(vector: Float[Array, " D_x"]) -> Float[Array, "time D_x"]:
    # today (sesmax-core)
    state = L96State(x=vector)
    return rollout(model, state, dt=0.05, n_steps=10, include_initial=True).x


# → (T+1, D_x)
truth = forecast(truth_start)
# μ_x = x_0 + σ_x ξ, with σ_x = 0.5
# → (D_x,)
background = truth_start + 0.5 * jax.random.normal(jax.random.key(1), (40,))
# y_t = H x_t + ε_t, with ε_t ~ 𝒩(0, σ_ε² I), σ_ε = 0.2, every second variable
# → (D_y,)
observed = jnp.arange(0, 40, 2)
# → (T+1, D_x)
noisy = truth + 0.2 * jax.random.normal(jax.random.key(2), (11, 40))
# 1 where y_t is seen, from t = 1
# → (T+1, D_x)
mask = jnp.zeros((11, 40)).at[1:, observed].set(1.0)

# today (vardax, sesmax-core)
# x̂_0 = argmin 𝒥(x_0)
# → (1, D_x)
psd = lx.positive_semidefinite_tag
four_dvar = StrongFourDVar(
    forward=VardaxForwardModel(model=model, template=truth_state, dt=0.05),
    obs_op=MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=lx.MatrixLinearOperator(0.25 * jnp.eye(40), psd),
    obs_cov_op=lx.MatrixLinearOperator(0.04 * jnp.eye(40), psd),
    max_steps=500,
)
window = Batch1D(input=(mask * noisy)[None], mask=mask[None], target=truth[None])
variational = four_dvar(window)

# today (sesmax-core)
# x_0⁽ᵉ⁾ ~ 𝒩(μ_x, σ_x² I)
# → (N_E, D_x)
ensemble = make_ensemble(L96State(x=background), jax.random.key(3), size=100, std=0.5)
# today (filterax, sesmax-core)
# p(x_t | y_1, …, y_t), one analysis per step
# analysis_history: → (T, N_E, D_x)
etkf = ETKF(
    dynamics=FilteraxDynamics(model=model, template=truth_state),
    obs_op=SubsampleObs(indices=observed),
)
obs = [(noisy[t, observed], 0.05 * t) for t in range(1, 11)]
obs_cov = lx.DiagonalLinearOperator(0.04 * jnp.ones(20))
filtered = etkf.assimilate(ensemble, obs, obs_cov, t0=0.0)

# x_T from the background, from 4D-Var and from the filter
# → (3, D_x)
ends = jnp.stack(
    [
        forecast(background)[-1],
        forecast(variational[0])[-1],
        einx.mean("[member] k -> k", filtered.analysis_history[-1]),
    ]
)
# RMSE against the truth at the end of the window
# → (3,)
rmse = jnp.sqrt(einx.mean("run [k] -> run", (ends - truth[-1]) ** 2))
print(rmse.round(2))
```

This prints `[1.6  0.11 0.14]`, the root-mean-square error of the forty variables at the end of the window. The background forecast has drifted to `1.6`. 4D-Var brings it down to `0.11` and the filter to `0.14`, both below the observation noise of `0.2`, although half of the variables were never seen. Neither method knows that the model is Lorenz-96. To use another one, change `model`, its state and the sizes, and keep the two assimilation blocks.

The [core's README](packages/sesmax-core/README.md) inverts for an unknown source in shallow water, and the [4D-Var](https://jejjohnson.github.io/sesmax/data-assimilation-4dvar/) and [ETKF](https://jejjohnson.github.io/sesmax/data-assimilation-etkf/) notebooks go further into each method.

## 🔬 The packages

| Tier | Package | Holds | Status |
|---|---|---|---|
| – | [`sesmax`](packages/sesmax/README.md) | no code: it installs the others | scaffolded |
| 2 | `sesmax-couple` | flux exchange, regridding, coupled runs | later |
| 1 | [`sesmax-ocean`](packages/sesmax-ocean/README.md) | shallow-water, QG and spherical ocean models | moving from somax (#152) |
| 1 | [`sesmax-experiment`](packages/sesmax-experiment/README.md) | runs in chunks, monitors and checkpoints | the run loop is moving from somax (#186) |
| 1 | `sesmax-atmos`, `sesmax-land`, `sesmax-ice` | one component each | later |
| 0 | [`sesmax-core`](packages/sesmax-core/README.md) | states, terms, forcing, rescaling, rollouts, assimilation adapters | released (0.1.0) |

Imports point down the tiers. Two components never import each other: what they share goes into the core, and what couples them goes into the coupler. The phases and their gates are in the [workspace and core design](https://jejjohnson.github.io/sesmax/workspace-and-core/), and the components after the ocean are in geostack's [plan for the Earth-system workspace](https://github.com/jejjohnson/geostack/blob/main/docs/design/esmax.md) (written when the name was esmax). The land and the ice only start when they have real code.

somax stays the working ocean package until `sesmax-ocean` is in.

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
- **[Capability index](https://jejjohnson.github.io/sesmax/reference/capabilities/)**: every public name once, with a one-line summary, together with the names of the main libraries underneath.
- **[Design](https://jejjohnson.github.io/sesmax/workspace-and-core/)**: the workspace, what moves from somax, the phases and the gates.

## 🧩 Ecosystem

sesmax is in the GeoModels layer of the [GeoStack](https://github.com/jejjohnson/geostack), beside [plumax](https://github.com/jejjohnson/plumax) (plumes and their sources) and [radtrax](https://github.com/jejjohnson/radtrax) (radiative transfer). It owns the models and their contracts, and composes the rest:

| Concern | Owner |
|---|---|
| Modules and filtered transforms | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Time integration | [`diffrax`](https://github.com/patrick-kidger/diffrax), [`integrix`](https://github.com/jejjohnson/integrix) |
| Grids, masks and C-grid operators | [`gridax`](https://github.com/jejjohnson/gridax) |
| Elliptic and spectral solves | [`gaussx`](https://github.com/jejjohnson/gaussx), [`spectraldiffx`](https://github.com/jejjohnson/spectraldiffx) |
| Spatial bases of a forcing | [`geonnax`](https://github.com/jejjohnson/geonnax) |
| Constrained parameters | [`paramax`](https://github.com/danielward27/paramax) |
| Physical constants | [`geophysix`](https://github.com/jejjohnson/geophysix) |
| Operators, rollouts and cycles | [`pipekit`](https://github.com/jejjohnson/pipekit) |
| Variational assimilation | [`vardax`](https://github.com/jejjohnson/vardax) |
| Ensemble filters | [`filterax`](https://github.com/jejjohnson/filterax) |

The packages import these libraries, and none of these libraries imports sesmax. No other project imports it yet; building plumax's transport model on `sesmax-core` is [an open issue](https://github.com/jejjohnson/plumax/issues/232).

## 🌳 Repository layout

```
sesmax/
├── pyproject.toml     ← the workspace root, never published
├── packages/
│   ├── sesmax-core/   ← tier 0: src/sesmax_core and its tests
│   └── sesmax/        ← the umbrella: dependencies only
├── tests/             ← workspace tests: tiers, docs build, capability index
├── docs/              ← one site: MyST prose and the MkDocs API reference
└── scripts/           ← build_docs.py, capabilities.py
```

## ⚙️ Development

With [uv](https://docs.astral.sh/uv/) installed:

```bash
make install    # every package and dependency group, plus the pre-commit hooks
make test       # every test and doctest
make lint       # ruff on the whole repository
make typecheck  # ty on packages/*/src and scripts
make docs       # build the documentation site into public/
```

`make help` lists every target. The rules for contributors and coding agents are in [AGENTS.md](AGENTS.md), and the review checklist is in [CODE_REVIEW.md](CODE_REVIEW.md).

## 🎓 References

- Lorenz (1996). Predictability: a problem partly solved. *Proceedings of the Seminar on Predictability*, ECMWF.

## 📄 License

MIT. See [LICENSE](LICENSE).
