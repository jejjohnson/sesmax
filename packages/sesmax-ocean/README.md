# sesmax-ocean

**Simple ocean models in JAX: quasi-geostrophic, shallow-water and spherical models on [`sesmax-core`](../sesmax-core/README.md) and [`gridax`](https://github.com/jejjohnson/gridax).**

The models and their run configs come from [somax](https://github.com/jejjohnson/somax) (commit `9cb19b3`), and each one reproduces somax's numbers to round-off, up to the fixes since, which the API pages list. The plan and its gates are in [the ocean design page](https://jejjohnson.github.io/sesmax/ocean/).

## 🧱 What is here

| Module | Names | What it does |
|---|---|---|
| `sesmax_ocean.qg` | `BarotropicQG`, `BaroclinicQG`, `ReparameterizedQG` | Quasi-geostrophic models, one layer or several; the last is shallow water projected onto geostrophic balance |
| `sesmax_ocean.swm` | `NonlinearShallowWater2D`, `MultilayerShallowWater2D` | Shallow-water models, one layer or several |
| `sesmax_ocean.spherical` | `SphericalSWM`, `SphericalQG` | Shallow water and QG on a latitude band of the sphere |
| `sesmax_ocean.scenarios` | `SCENARIOS`, `MODELS`, `build_ocean_run`, `check_compatible` | What to simulate, and which models go with it |
| `sesmax_ocean.resolution` | `check_munk_width`, `check_stommel_width`, `check_deformation_radius`, `check_equatorial_deformation_radius` | Does the grid span the boundary layers and the deformation radius? |
| `sesmax_ocean.checks` | `check_pv_inversion`, `check_static_stability` | Is the inversion sound, and are the layers stable? |
| `sesmax_ocean.eval` | `compute_eval_metrics` and the metrics it picks from | Numbers about one state, with no truth to compare with |
| `sesmax_ocean.forcing` | `ssh_geostrophic`, `sss_coastal`, `sst_frontal`, `geostrophic_currents` | Reduced-order forcings for one ocean field each |

## 📥 Install

The package is not on PyPI. Install it from a clone, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
uv sync --package sesmax-ocean
```

## 🚀 Quickstart

```python
import jax.numpy as jnp
from sesmax_ocean.eval import compute_eval_metrics
from sesmax_ocean.qg import BarotropicQG, BarotropicQGState

# a wind-driven double gyre, from rest
model = BarotropicQG.create(
    nx=64, ny=64, lateral_viscosity=500.0, bottom_drag=1e-7, wind_amplitude=1e-11
)
rest = BarotropicQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))

# one day of hourly steps
solution = model.integrate(rest, t0=0.0, t1=86400.0, dt=3600.0)
state = BarotropicQGState(q=solution.ys.q[-1])
metrics = compute_eval_metrics(model, state)
```

## 🏃 Run a config

The package registers its models as a component, so [`sesmax-experiment`](../sesmax-experiment/README.md) runs them from a file. Eight are in [`configs/`](configs), each with a `debug` block that makes it small and short:

| Config | Model | Cells | Time |
|---|---|---|---|
| `doublegyre_bt_qg` | `barotropic_qg` | 64² | 1 year |
| `doublegyre_bt_qg_nondim` | `barotropic_qg`, from dimensionless numbers | 64² | 20 crossings |
| `doublegyre_bt_qg_hires` | `barotropic_qg` | 256² | 5 years |
| `spinup_bc_qg` | `multilayer_qg`, 3 layers | 128² | 3 years |
| `doublegyre_bc_qg` | `multilayer_qg`, 3 layers, from the spinup | 128² | 1 year |
| `doublegyre_bc_qg_hires` | `multilayer_qg`, 3 layers | 256² | 5 years |
| `doublegyre_swm_hires` | `multilayer_nonlinear_swm`, 3 layers | 256² | 5 years |
| `swm_jet` | `multilayer_nonlinear_swm`, 2 layers | 64² | 30 days |

```bash
uv sync
uv run sesmax-experiment list-scenarios
uv run sesmax-experiment simulate --debug --config packages/sesmax-ocean/configs/swm_jet.yaml --output-dir runs/swm_jet
uv run sesmax-experiment spinup --config packages/sesmax-ocean/configs/spinup_bc_qg.yaml --output-dir runs/spinup
uv run sesmax-experiment restart --config packages/sesmax-ocean/configs/doublegyre_bc_qg.yaml --from runs/spinup/final_state.zarr --output-dir runs/bc_qg
```

## 📄 License

MIT. See [LICENSE](LICENSE).
