# sesmax

> Simple Earth-system models in JAX: cheap, differentiable forward models to use as priors in short-window data assimilation.

sesmax is a workspace of packages: a shared core, one package per Earth-system component, and later a coupler on top. Its models are the forward models inside an assimilation window of hours to weeks, so they are built to be cheap enough for ensembles, differentiable through a whole window, honest about their error, and learnable.

:::{note} Status
`sesmax-core` holds the model-agnostic machinery moved from [somax](https://github.com/jejjohnson/somax), and the ocean component follows. The plan, its phases and its gates are in the [workspace and core design](design/workspace_and_core.md).
:::

## Packages

| Distribution | Import | Holds | Status |
| --- | --- | --- | --- |
| `sesmax-core` | `sesmax_core` | state and parameter types, term algebra, steppers, forcing and bases, scales, DA adapters, monitors, I/O, test problems | moved from somax; first release 0.1.0 |
| `sesmax-ocean` | `sesmax_ocean` | shallow-water, QG and spherical ocean models, ocean diagnostics, the double-gyre scenarios | planned (C2) |
| `sesmax` | — | code-free umbrella that installs the packages | scaffolded (C0) |

## The core in use

A Lorenz-96 system stands in for any model: integrate one window, differentiate a rollout loss with respect to the forcing $F$, and hand the model to a filter.

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

The [test problems](notebooks/step01_linear_convection_1d.ipynb) work through the models one equation at a time.

## Installation

From source, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
make install
```
