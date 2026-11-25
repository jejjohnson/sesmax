# sesmax-ocean

**Simple ocean models in JAX: quasi-geostrophic, shallow-water and spherical models on [`sesmax-core`](../sesmax-core/README.md) and [`gridax`](https://github.com/jejjohnson/gridax).**

The models are moving here from [somax](https://github.com/jejjohnson/somax) (commit `9cb19b3`), and each one has to reproduce somax's numbers to round-off before it lands. The plan and its gates are in [the ocean design page](https://jejjohnson.github.io/sesmax/ocean/); the progress is in [sesmax#152](https://github.com/jejjohnson/sesmax/issues/152).

## 🧱 What is here

| Module | Names | What it does |
|---|---|---|
| `sesmax_ocean.resolution` | `check_munk_width`, `check_stommel_width`, `check_deformation_radius`, `check_equatorial_deformation_radius` | Does the grid span the boundary layers and the deformation radius? |
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
from sesmax_core.testing import LinearShallowWater2D, LinearSW2DState
from sesmax_ocean.eval import compute_eval_metrics

# an f-plane box with a uniform flow
model = LinearShallowWater2D.create(nx=32, ny=32, Lx=1e6, Ly=1e6)
flow = jnp.ones((model.grid.Ny, model.grid.Nx))
state = LinearSW2DState(h=0.0 * flow, u=0.5 * flow, v=-0.3 * flow)

# with no pressure gradient, the imbalance is the whole Coriolis term: 1
metrics = compute_eval_metrics(model, state)
```

## 📄 License

MIT. See [LICENSE](LICENSE).
