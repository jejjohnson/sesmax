---
title: "The run loop in sesmax-core"
subtitle: "Design · moving somax's runner into sesmax_core.sim, built on pipekit"
short_title: Run loop
description: "Places the generic part of somax-sim in sesmax-core as a chunk pipeline of pipekit operators, with components registering their models and checks, and reports a prototype and two defects it found in how the core integrates."
date: 2026-11-05
keywords:
  - sesmax
  - pipekit
  - runner
  - checkpointing
numbering:
  heading_1: true
  heading_2: true
  table: true
  code: true
---

5 Nov 2026 · J. Emmanuel Johnson

## Decision

Proposed: put the run loop in `sesmax_core.sim`, split into two levels, with the ocean's models, scenarios and ocean-specific checks registered from `sesmax-ocean`.
The lower level, `run_chunks`, is a pipeline of pipekit operators that needs nothing beyond the base install.
The upper level, with run specs, artifacts and the `sesmax-sim` command, sits behind a `[sim]` extra.
Alternative: keep the runner in `sesmax-ocean`, which is simpler for C2, but a second component would then have to copy the loop, because the tiers forbid it importing the ocean.
Status: proposed, for review. The chunk pipeline is prototyped, and nothing else here is implemented. This page answers SX-4 and #28.

## What this page settles

| Question | Answer |
| --- | --- |
| Where does the loop live? | In `sesmax_core.sim`; the ocean keeps its models, scenarios, metrics and DVC commands |
| How much can pipekit carry? | The loop, the cadence, the sinks and the retry; the advance, the monitors and the run spec stay ours, for the reasons in [the table](#pipekit-map) |
| Does the design work? | The prototype passes on Lorenz-96, and [the results](#prototype) include two defects it found |
| What does it need first? | One fix to `Model.integrate`, and `pipekit-cycle` 0.0.4 |
| What changes for a user? | No compile per chunk and no snapshots held in memory, with the same YAML files and the same artifacts |

## What the runner is today

The runner is `somax/_src/cli/`, and #28 counts 5,662 lines at `9cb19b3`.
Very little of the loop knows about the ocean: in `_run.py`, a name search finds one hard dependency, `compute_eval_metrics`, and models arrive through the `_factories.build` import.
The rest of the ocean knowledge is in the registries, the checks and a table of units.

| File | Lines | Holds | Home |
| --- | --- | --- | --- |
| `_run.py` | 1,448 | the chunk loop, artifacts, manifest, `simulate`, `spinup`, `restart` | `sesmax_core.sim` |
| `spec.py` | 458 | `RunSpec` and its YAML | `sesmax_core.sim`, with `scenario` and `model` left opaque |
| `_progress.py` | 201 | the `run.log` sink and its heartbeat thread | `sesmax_core.sim` |
| `_units.py` | 138 | time and field formatting; `FIELD_UNITS` | formatting in the core; `FIELD_UNITS` in the ocean |
| `app.py` | 434 | the cyclopts commands | `sesmax_core.sim` |
| `_assertions.py` | 535 | the check runner, `check_cfl`, `check_bounded_metric`; PV, stability and deformation checks | runner and first two in the core; the rest in the ocean |
| `_factories.py`, `_compatibility.py` | 265 | scenario × model dispatch and its rules | `sesmax-ocean` |
| `models_registry/`, `scenarios/` | not counted here | the registered models and scenarios | `sesmax-ocean` |
| `_data.py` | 411 | basin data through the DVC command line | `sesmax-ocean` |

`_ChunkStep._apply` is where the design has to change.
It already runs inside `pipekit_cycle.Cycle`, but one method integrates, logs, runs the monitors, keeps snapshots and writes checkpoints, because its docstring records that pipekit's per-step hook dispatch is a reserved no-op.
That hook is still a no-op in pipekit `main`, but the design doesn't need it, since each of those jobs can be an operator in a pipeline.

## The design

### Two levels

`run_chunks` is for a notebook, a test or another repository: it takes a built model and returns states, and it writes nothing unless given a sink.
`simulate`, `spinup` and `restart` take a `RunSpec` and an output directory, and they add the logging, the artifacts and the checks.

```{mermaid}
flowchart TD
    cli["sesmax-sim<br/>[sim]"]
    spec["simulate, spinup, restart<br/>[sim]"]
    loop["run_chunks<br/>base install"]
    registry["component registry<br/>entry points"]
    ocean["sesmax-ocean<br/>models, scenarios, checks"]
    pipekit["pipekit, pipekit-cycle"]
    cli --> spec
    spec --> loop
    spec --> registry
    ocean -. registers .-> registry
    loop --> pipekit
```

### One chunk as a pipeline

The carrier is a small frame, the state plus the `ChunkInfo` the core's monitors already read, so a stateless operator can see the time and the cadence.
Each job then becomes one stage of a `Sequential`, and `Cycle` repeats it.

```python
# Shapes: the state is the model's pytree; N_chunk = diagnostic chunks in the run
from pipekit import Branch, Identity, Retry, Sequential, Sink, Tap, Try
from pipekit_cycle import Cycle


# proposed (SIM-1)
class ChunkFrame:
    state: PyTree
    info: ChunkInfo  # today (sesmax_core.monitor)


# today (pipekit): retry once, then warn and keep running
sturdy_checkpoint = Try(
    Retry(Sink(write_checkpoint), attempts=2, base_delay=0.0, on=(OSError,)),
    Tap(warn_checkpoint_skipped),
    on=(OSError,),
)
# proposed (SIM-1): AdvanceChunk and RunMonitors
chunk = Sequential(
    [
        AdvanceChunk(model, schedule),  # x(tₖ₊₁) = x(tₖ) + ∫ f(x, t) dt
        RunMonitors(model, monitors),  # raises if a monitor asks to stop
        Branch(is_snapshot, Sink(write_snapshot), Identity()),
        Branch(is_checkpoint, sturdy_checkpoint, Identity()),
    ]
)
# today (pipekit_cycle, from 0.0.4): a stateful Sequential as the step
frame, _ = Cycle(step_op=chunk, n_steps=N_chunk)(first_frame, 0)
```

A monitor runs before the sinks, so a state that fails a check never reaches the disk, as today.
The carry-state shrinks from `_ChunkCarry` to the chunk index, because snapshots leave through a sink.

(pipekit-map)=
### What pipekit carries

| Job | Today | Use | Why |
| --- | --- | --- | --- |
| Repeating chunks | `Cycle` | `pipekit_cycle.Cycle` | already in use |
| Cadence | `if` in `_ChunkStep` | `pipekit.Branch` | a predicate on the frame |
| Snapshots, checkpoints, log lines | inline | `pipekit.Sink`, `pipekit.Tap` | the documented purpose of both |
| Checkpoint retry | a hand-written `for` loop | `pipekit.Retry` inside `pipekit.Try` | same behaviour, measured below |
| Run identity | git commit in the manifest | `pipekit.hashing.stable_json` and `sha256_hex` on the spec | not `pipeline_hash`, which hashes a closure by name: two different writers with one name hashed alike |
| Advancing a chunk | `model.integrate` | our own `AdvanceChunk` | `ModelOp` and `as_forward_model` integrate over $[0, \Delta t]$ and drop the solver statistics; on a forced Lorenz-96 this was wrong by 11.8 |
| Monitors | `Monitor` protocol | our own, adapted by `RunMonitors` | they have start and end hooks and may stop a run; `pipekit.Profile` reports only afterwards, and `ThroughputMonitor` needs each chunk's time during the run |
| Run spec | dataclasses and YAML | our own | `pipekit.dumps` refuses `Sink`, `Tap` and `ModelOp`, which hold closures or models; the spec is the faithful record |
| In-graph stepping | none | not `ScanCycle` or `Rollout` | a single `lax.scan` cannot write or stop between chunks; they stay the tools for short differentiable windows |

`pipekit-experiment` is the one package we don't pin yet that fits, and it can wait.
Its `ExperimentTracker` protocol maps onto a monitor: `start_run` on run start, `log_metrics` on each chunk and `end_run` on run end.
A small `TrackerMonitor` would therefore give MLflow or W&B logging without touching the loop, and pipekit-train's `LogToExperiment` is the precedent.
`Run` has no dedicated field for the git commit, library versions or platform that the manifest records, so `manifest.json` stays.
Its `DVCDatasetVersioning` is worth comparing with `_data.py` in C2, which I haven't done.

Nothing needs asking upstream: `pipekit-cycle` 0.0.4 already has what this page needs.

### The component registry

A component registers through an entry point, as #28 proposed, and core names no component.

```python
# proposed (SIM-2)
@dataclass(frozen=True)
class Component:
    name: str
    models: Mapping[str, Builder]  # model name → build(spec) → BuiltRun
    checks: Mapping[str, Check]  # extra names for spec.assertions
    evaluate: Callable[[Model, State], dict[str, float]] | None = None
    field_units: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


# proposed (SIM-2): replaces every read of spec.scenario.nondim in the loop
@dataclass(frozen=True)
class BuiltRun:
    model: Model
    state0: State
    coordinates: Literal["si", "nondimensional"] = "si"
```

```toml
# proposed (C2): packages/sesmax-ocean/pyproject.toml
[project.entry-points."sesmax.components"]
ocean = "sesmax_ocean.sim:COMPONENT"
```

The loop finds a builder by `spec.model.name` across installed components, and two components claiming one name is an error at load time.
The YAML files therefore don't change, which keeps C2's gate of "only import changes".
`scenario` and `model` reach the builder as plain mappings, so the ocean's scenario × model dispatch and compatibility rules stay private to it.

`sesmax_core.testing` registers itself as a component too, with Lorenz-96 and the linear shallow-water model, so the core can test its runner end to end without the ocean.
`ModelOp.from_registry`, which the core move left out, returns on top of this.

### Dependencies

`run_chunks` needs only `pipekit` and `pipekit-cycle`, which are already in the base install.
`sesmax_core.sim` stays out of the top-level `__init__`, because `import sesmax_core` must stay pipekit-free.
The `[sim]` extra needs `cyclopts`, `loguru`, `pyyaml`, `xarray` and `zarr`.
That is smaller than #28 expected: somax's `sim` group also lists `hydra-core`, `hydra-zen`, `omegaconf`, `orbax-checkpoint` and `dvc`, but nothing under `cli/` imports the first four, and only the ocean's `_data.py` needs `dvc`.

(prototype)=
## The prototype

We built the chunk pipeline in a scratch script, on Lorenz-96 with 40 variables, in double precision, with `pipekit` 0.1.4 and `pipekit-cycle` 0.0.3 and a small wrapper in place of the `Sequential` step.
The prototype's chunks use `integrate` with a plain `dt`, so the fix for the step count below was tested beside it rather than inside it.
`pipekit-cycle` 0.0.4 was then checked separately, on a toy stateful step with a `Branch` and a `Sink`.
The run specs, the registry, the command line and the rename of partial snapshots were not prototyped.

| Check | Result |
| --- | --- |
| The pipeline's 8 snapshots, against one solve, up to $t = 4$ | largest difference $4.1 \times 10^{-4}$, against a largest value of 16, from [the step count](#step-count) |
| A checkpoint write that fails once | 5 write calls for 4 snapshots, no warning, same final state |
| A checkpoint write that always fails | 8 write calls, 4 warnings, the run finishes with the same final state |
| A NaN state with `NonFiniteMonitor` | the run stops with the monitor's reason, and no snapshot is kept |
| The five `default_monitors()` on the frame's `ChunkInfo` | run without change |
| 8 snapshots streamed to Zarr, against the same states held in memory | identical |
| Restart from the streamed snapshot at $t = 2$ | identical to the uninterrupted run at $t = 4$ |
| Cost of the operators over 400 chunks, compiled once | 219 ms against 201 ms for a bare loop, about 45 µs per chunk |

The cost of the operators doesn't matter next to a solve, so the choice between them and a plain method rests on separation alone.

## Two defects the prototype found

Both are in how the core integrates, and somax's runner has them too, because it calls `model.integrate` once per chunk.

### Every chunk recompiles

Today (bug): each `Model.integrate` call compiles again. `build_terms` makes a new closure per call, and Python-float times are static arguments of diffrax's `filter_jit`. Either one defeats the compile cache.

| `build_terms` returns | Times | Lorenz-96, ms per call after the first |
| --- | --- | --- |
| closure (today) | floats | 710 to 760 |
| closure | arrays | 1,020 to 1,110 |
| module | floats, the same times each call | 1 |
| module | arrays, new times each call | 1 |

An earlier run with the module form and new float times stayed at 740 to 770 ms, so the module form alone isn't enough for a run whose times advance.
On a 64 × 64 linear shallow-water model, each call of ten steps took about 1.7 s.

After the fix: `build_terms` returns `dfx.ODETerm(_ModelVectorField(self))`, the module `as_forward_model` already uses, and `integrate` turns its times into arrays.
Repeat calls on Lorenz-96 then take 1 ms.
With float times, the module form gave exactly the closure's answer on both models, and a finite gradient on Lorenz-96.
With array times in single precision, a separate check differed by $9.5 \times 10^{-7}$ after 0.1 time units, which I didn't trace, so the fix's tests should compare with a tolerance.
`TermModel`, `ScaledModel` and the IMEX path have their own `build_terms` and were not measured, so they are part of this fix's gate.

(step-count)=
### A chunk can take a different number of steps

Today (bug): with a fixed `dt`, a chunk takes one extra step when rounding pushes $(t_1 - t_0) / \Delta t$ just above a whole number.
From $t = 0.2$ to $0.30000000000000004$ with $\Delta t = 0.01$, diffrax took 11 steps of about 0.0091 instead of 10 of 0.01, and 24 of 40 chunks did this.
The result is still a valid solution, but its discretisation depends on where the chunks fall.

| Run to $t = 4$ on Lorenz-96 | Largest difference from one solve |
| --- | --- |
| Chunks of 0.1 with `dt = 0.01` (today) | $1.2 \times 10^{-3}$, growing from $4 \times 10^{-15}$ after the first chunk |
| Chunks of $2^{-3}$ with $\Delta t = 2^{-7}$, which divides exactly | 0 |
| Chunks of 0.1 with `dfx.StepTo` and 10 steps each | $3.8 \times 10^{-11}$ |
| Linear shallow water, chunks of 100 s with `dt = 10` s | $1.0 \times 10^{-15}$ |

So this shows up only when the ratio is inexact, which SI runs with whole-second steps avoid and nondimensional runs may not.

After the fix: `AdvanceChunk` states the count itself, $n = \lceil (t_1 - t_0) / \Delta t - 10^{-9} \rceil$, and passes `dfx.StepTo(ts=jnp.linspace(t0, t1, n + 1))`.
A shorter last interval, which `_build_save_times` allows, gets its own smaller $n$.
Over one whole solve, `StepTo` and `dt0` gave identical results.
An explicit `stepsize_controller`, such as `imex_stepsize_controller`, passes through unchanged and sets its own steps.

## What changes and what stays

| | somax today | `sesmax_core.sim` |
| --- | --- | --- |
| Compiles per run | one per chunk | one |
| Steps in a chunk | can vary by one | stated |
| Snapshots during a run | all held in memory until the end | streamed to `snapshots.partial.zarr`, renamed when the run succeeds |
| A run that fails | no artifacts | no `snapshots.zarr`, `final_state.zarr` or `metrics.json`; the partial store stays for inspection |
| Metrics hook | `compute_eval_metrics` | `Component.evaluate` |
| `check_cfl` | its own formula | `stable_step`, as SC-4 planned |

What stays: the YAML schema, the commands, the artifact names, the checks' names, `run.log`, the monitors and the crash-recovery `checkpoint.zarr`.
`SimulationCheckpointer`, the orbax class in the core, stays unused by the runner, as it is in somax.

## Phases

| Phase | Scope | Gate |
| --- | --- | --- |
| SIM-0 | Compile once in `integrate`; bump `pipekit-cycle` to 0.0.4 | Repeat calls hit the cache on `Model`, `TermModel`, `ScaledModel` and IMEX; existing suites pass; `uv lock` resolves |
| SIM-1 | `run_chunks`, `ChunkFrame`, `AdvanceChunk`, `RunMonitors` and the sinks, in the base install | The table above as tests, on Lorenz-96 and linear shallow water |
| SIM-2 | `RunSpec`, `simulate`, `spinup`, `restart`, the registry, `sesmax-sim`, and the testing component, behind `[sim]` | somax's `test_cli_run.py`, `test_cli_spec.py` and `test_cli_progress.py`, ported to the testing component, pass |
| SIM-3 | The generic checks on `stable_step`, and `ModelOp.from_registry` | somax's matching tests pass |
| C2 | `sesmax-ocean` registers its models, scenarios, checks and units | A short double-gyre run from one of somax's configs matches `somax-sim`, up to the two fixes above |

SIM-0 is worth doing whatever happens to the rest, because it speeds up every caller of `integrate`.
`test_cli_run.py` is a fair gate for SIM-2, because it mostly drives the public `simulate`, `spinup` and `restart`, and doesn't name `_ChunkStep`.
SIM-2 ports from `feat/doublegyre-bt-qg-hires`, not from `9cb19b3`: its checkpoint retry, `62ec02c`, isn't on somax's `main`, so the #19 query misses it.

## Risks

- **The prototype ran on small models.** Streaming and the compile cache were not tried on a 256 × 256 multilayer run. C2's gate is the first test at that size.
- **Older runs must still restart.** Checkpoints and final states carry `somax_sim_t`, `somax_coordinates` and related attributes, and runs on the data mount use those names. Fallback: write `sesmax_*`, and read either.
- **Two trees during the move.** somax's runner keeps changing until C2. #19's "core changes land in sesmax" note should cover the runner from SIM-2 onwards.
- **The name search can miss things.** The file table comes from names and a partial read of `_run.py`, so an ocean assumption may still sit inside a helper. The testing component is what will expose one.

## Open questions

| ID | Question | Leaning |
| --- | --- | --- |
| SIM-Q1 | Does #29's `geo.simulate` keep that name, given `sim.simulate` here and the "one object, one name" rule? | Keep `sim.simulate` for the spec-driven run, and settle #29's name when that design is written |
| SIM-Q2 | Stream snapshots by default, or keep them in memory? | Stream: it removes the memory growth, and the partial store keeps the no-artifacts promise |
| SIM-Q3 | Does `spec.timestepping` gain a solver block, since the runner always uses `integrate`'s default today? | Not in this move; `run_chunks` takes `**integrate_kwargs`, as `WindowOperator` does |

## Follow-up issues

- `fix(model): compile Model.integrate once across calls`
- `fix(sim): state the step count of a chunk`, part of SIM-1
- `feat(sim): run_chunks, a chunk pipeline on pipekit`
- `feat(sim): RunSpec, the component registry and sesmax-sim behind [sim]`
- `feat(sim): generic checks on stable_step, and ModelOp.from_registry`

## Relationships

- Depends on: [the workspace and core design](workspace_and_core.md), [the shared-core design](shared_core.md)
- Answers: SX-4, #28
- Related: #29 (levels), #19 (later somax commits), #15 (`ModelOp`)
- Parent: #8
