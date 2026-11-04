# sesmax

[![Tests](https://github.com/jejjohnson/sesmax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/sesmax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/sesmax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/sesmax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple Earth-system models in JAX.

Author: J. Emmanuel Johnson

sesmax is a workspace of packages whose models are cheap, differentiable forward models to use as priors in short-window data assimilation: a window of hours to weeks, inside the predictability time of the scales the model resolves. A model here is judged as a prior: cheap enough to run ensembles and take gradients through a whole window, good where the observations look, honest about its error, and learnable. It grows out of [somax](https://github.com/jejjohnson/somax), a simple ocean model in JAX.

The core has moved from somax, and the ocean component is next. The plan, its phases and its gates are in the [workspace and core design](docs/design/workspace_and_core.md).

## Packages

| Tier | Distribution | Import | Holds | Status |
| --- | --- | --- | --- | --- |
| — | `sesmax` | `sesmax` | code-free umbrella that installs the packages | scaffolded |
| 1 | `sesmax-ocean` | `sesmax_ocean` | shallow-water, QG and spherical ocean models, ocean diagnostics, scenarios | planned (C2) |
| 0 | `sesmax-core` | `sesmax_core` | state and parameter types, term algebra, steppers, forcing and bases, scales, DA adapters, monitors, I/O, test problems | moved from somax; first release 0.1.0 |

A package imports only packages in a lower tier.

## The core in use

A Lorenz-96 system stands in for any model: integrate one window, differentiate a rollout loss with respect to the forcing $F$, and hand the model to a filter and to 4D-Var.

```python
# Shapes: D_x = 40 state variables; N_E = 32 ensemble members; T = 100 snapshots
import equinox as eqx
import jax

from sesmax_core.da import FilteraxDynamics, make_ensemble
from sesmax_core.testing import L96State, Lorenz96
from sesmax_core.train import rollout, rollout_loss

# dx_i/dt = (x_{i+1} − x_{i−2}) x_{i−1} − x_i + F
model = Lorenz96.create(F=8.0)
state0 = L96State.init_state(ndim=40, key=jax.random.key(0))  # x: (D_x,)

# diffrax underneath
solution = model.integrate(state0, t0=0.0, t1=5.0, dt=0.01)  # final state x: (1, D_x)

# L(F) = (1/T) Σₜ ‖x_t(F) − x_tʳᵉᶠ‖²
ref_traj = rollout(model, state0, dt=0.01, n_steps=100)      # x: (T, D_x)
grad_F = eqx.filter_grad(rollout_loss)(                      # same tree as model
    model, state0, ref_traj, dt=0.01
)

# filterax drives the model on flat vectors, with the [da] extra
ensemble = make_ensemble(state0, jax.random.key(1), size=32, std=0.1)  # (N_E, D_x)
dynamics = FilteraxDynamics(model=model, template=state0)
```

The same model goes to vardax for 4D-Var. Here the background's RMSE of 0.89 falls to 0.08.

```python
import jax.numpy as jnp
import lineax as lx
from vardax import Batch1D, MaskedIdentity, StrongFourDVar

from sesmax_core.da import VardaxForwardModel, state_to_vector

# Without advection the window stays free of chaos, which the 4D-Var minimiser needs
linear_model = Lorenz96.create(F=8.0, advection=False)
truth = rollout(linear_model, state0, dt=0.05, n_steps=10, include_initial=True)
noise = 0.2 * jax.random.normal(jax.random.key(2), truth.x.shape)        # (11, D_x)
batch = Batch1D(
    input=(truth.x + noise)[None],
    mask=jnp.ones_like(truth.x)[None],
    target=truth.x[None],
)
flat_state0, _ = state_to_vector(state0)                                 # (D_x,)
background = flat_state0 + jax.random.normal(jax.random.key(3), (40,))   # (D_x,)

# x̂₀ = argmin ½‖x₀ − μ_x‖²(Σ_x⁻¹) + ½ ∑ₜ ‖yₜ − h(xₜ)‖²(Σ_ε⁻¹)
psd = lx.positive_semidefinite_tag
strong = StrongFourDVar(
    forward=VardaxForwardModel(model=linear_model, template=state0, dt=0.05),
    obs_op=MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=lx.MatrixLinearOperator(jnp.eye(40), psd),
    obs_cov_op=lx.MatrixLinearOperator(0.04 * jnp.eye(40), psd),
)
analysis = strong(batch)[0]                                              # (D_x,)
```

The [capability index](docs/api/capabilities.md) lists every public name in `sesmax_core` and in the libraries it builds on.

## Layout

```text
sesmax/
├── pyproject.toml          # the workspace root (never published)
├── packages/
│   ├── sesmax-core/        # tier 0: src/sesmax_core, tests/
│   └── sesmax/             # the umbrella: dependencies only
├── tests/                  # workspace tests: tiers, docs build, capability index
├── docs/                   # one site: MyST prose + MkDocs API reference
└── scripts/                # build_docs.py, capabilities.py
```

## Development

With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
make install    # every package and dependency group, plus pre-commit hooks
make test       # every test and doctest
make lint       # ruff on the whole repo
make typecheck  # ty on packages/*/src and scripts
make docs       # build the documentation site into public/
```

`make help` lists every target. The rules for contributors and coding agents are in [AGENTS.md](AGENTS.md), and the review checklist is in [CODE_REVIEW.md](CODE_REVIEW.md).

## License

MIT; see [LICENSE](LICENSE).
