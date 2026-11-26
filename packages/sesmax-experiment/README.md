# sesmax-experiment

**What is done with a [sesmax](https://github.com/jejjohnson/sesmax) model: runs in chunks, with monitors and checkpoints.**

The run loop is moving here from [somax](https://github.com/jejjohnson/somax) (commit `9cb19b3`), as a pipeline on [pipekit](https://github.com/jejjohnson/pipekit). The design is in [the run-loop page](https://jejjohnson.github.io/sesmax/run-loop/); the progress is in [sesmax#186](https://github.com/jejjohnson/sesmax/issues/186). The package names no component of the Earth system: a component registers its models through `sesmax_core.component.Component`.

## 🧱 What is here

| Module | Names | What it does |
|---|---|---|
| `sesmax_experiment.run` | `run_chunks`, `Timeline`, the schedules | Run a model in chunks, with stages between them |
| `sesmax_experiment.monitor` | `BaseMonitor`, `MonitorVerdict`, `default_monitors` and the built-in monitors | Watch a run between its chunks, and stop it |
| `sesmax_experiment.checkpoint` (`[checkpoint]`) | `SimulationCheckpointer` | Periodic checkpoints with orbax |
| `sesmax_experiment.sim` (`[sim]`) | `RunSpec`, `simulate`, `spinup`, `restart` | Run a model described by a YAML file, and write its artifacts |
| `sesmax_experiment.components` | `components`, `find_model` | The installed components and the models they offer |

## 📥 Install

The package is not on PyPI. Install it from a clone, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
uv sync --package sesmax-experiment
```

## 🚀 Quickstart

Two time units of Lorenz-96 in chunks, with a snapshot every 0.4 and a monitor that stops a run that blows up:

```python
import jax.numpy as jnp
from sesmax_core.testing import L96State, Lorenz96
from sesmax_experiment.monitor import NonFiniteMonitor
from sesmax_experiment.run import Timeline, run_chunks

model = Lorenz96.create(F=8.0)
state = L96State(x=jnp.linspace(-2.0, 2.0, 40))

run = run_chunks(
    model,
    state,
    Timeline.build(0.0, 2.0, chunk=0.1, snapshot=0.4),
    dt=0.01,
    monitors=[NonFiniteMonitor()],
)
print([round(frame.info.t1, 1) for frame in run.snapshots])
```

A stage is any pipekit operator. This one writes a checkpoint every five chunks, tries again once if the write fails, and lets the run go on if it fails twice:

```python
from pipekit_ops import Branch, Identity, Retry, Sink, Try
from sesmax_experiment.run import every_chunks

write = Sink(lambda frame: print("checkpoint at", frame.info.t1))
sturdy = Try(
    Retry(write, attempts=2, base_delay=1.0, on=(OSError,)), Identity(), on=(OSError,)
)
run = run_chunks(
    model,
    state,
    Timeline.build(0.0, 2.0, chunk=0.1),
    dt=0.01,
    stages=[Branch(every_chunks(5), sturdy)],
)
```

A run can also be described by a file. This one names a model of the testing component that `sesmax-core` registers; the component of a model is found by its name, so nothing here imports it:

```yaml
scenario:
  name: adjustment
  grid: {nx: 32, ny: 32, Lx: 1.0e6, Ly: 1.0e6}
  consts: {f0: 1.0e-4}
  initial_condition: {type: step, params: {amplitude: 1.0}}
model:
  name: linear_swm
  params: {H0: 100.0}
timestepping: {t0: 0.0, t1: 86400.0, dt: 100.0, save_interval: 21600.0}
```

```bash
uv sync --package sesmax-experiment --extra sim
uv run sesmax-experiment list-models
uv run sesmax-experiment list-scenarios
uv run sesmax-experiment simulate --config adjustment.yaml --output-dir runs/adjustment
uv run sesmax-experiment restart --config adjustment.yaml --from runs/adjustment/final_state.zarr --output-dir runs/more
```

A run that succeeds leaves `snapshots.zarr`, `final_state.zarr`, `metrics.json`, `resolved.yaml`, `manifest.json`, `run.log` and `metrics.jsonl`, a line of numbers per chunk. One that fails leaves its two logs and the snapshots it had taken, as `snapshots.partial.zarr`.

`assertions` in the file asks for checks, which fail the run: `cfl: {wave_speed_m_per_s: 31.3, max_cfl: 0.5}` before it starts, `bounded_metric: {name: energy, min: 0.0}` before its metrics are written, and those of the model's component.

A long run writes checkpoints, each a store of its own that is finished by a rename, so a crash leaves the older ones. A run whose wall time is up stops with a checkpoint and exit code 75, and the rest of it takes the steps the whole run would have taken:

```bash
uv run sesmax-experiment simulate --config adjustment.yaml --output-dir runs/long --checkpoint-every-n-chunks 1 --keep-last 2 --max-wall-seconds 3600
uv run sesmax-experiment restart --config adjustment.yaml --from runs/long --output-dir runs/long-2
```

## 📄 License

MIT. See [LICENSE](LICENSE).
