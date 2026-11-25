# sesmax-experiment

**What is done with a [sesmax](https://github.com/jejjohnson/sesmax) model: runs in chunks, with monitors and checkpoints.**

The run loop is moving here from [somax](https://github.com/jejjohnson/somax) (commit `9cb19b3`), as a pipeline on [pipekit](https://github.com/jejjohnson/pipekit). The design is in [the run-loop page](https://jejjohnson.github.io/sesmax/run-loop/); the progress is in [sesmax#186](https://github.com/jejjohnson/sesmax/issues/186). The package names no component of the Earth system: a component registers its models through `sesmax_core.component.Component`.

## 🧱 What is here

| Module | Names | What it does |
|---|---|---|
| `sesmax_experiment.monitor` | `BaseMonitor`, `MonitorVerdict`, `default_monitors` and the built-in monitors | Watch a run between its chunks, and stop it |
| `sesmax_experiment.checkpoint` (`[checkpoint]`) | `SimulationCheckpointer` | Periodic checkpoints with orbax |

## 📥 Install

The package is not on PyPI. Install it from a clone, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
uv sync --package sesmax-experiment
```

## 🚀 Quickstart

A monitor is asked after each chunk, and may stop the run:

```python
import jax.numpy as jnp
from sesmax_core.testing import L96State, Lorenz96
from sesmax_experiment.monitor import ChunkInfo, NonFiniteMonitor

model = Lorenz96.create(F=8.0)
state = L96State(x=jnp.array([1.0, jnp.nan, 2.0, 3.0]))
info = ChunkInfo(
    index=0, n_chunks=1, t0=0.0, t1=0.05, wall_seconds=0.1, is_snapshot=True
)
verdict = NonFiniteMonitor().on_chunk_end(model, state, info)
assert verdict.terminate
```

## 📄 License

MIT. See [LICENSE](LICENSE).
