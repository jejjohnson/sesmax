---
title: "sesmax: the workspace scaffold and sesmax-core"
subtitle: "Design · moving somax's model-agnostic machinery into the first sesmax package"
short_title: Workspace and core
description: "Plans the sesmax uv workspace (adopted from the package template), the contents of sesmax-core taken from somax at a pinned commit, the renames, the tests that move with it, and the phases and gates up to the ocean component."
date: 2026-10-30
keywords:
  - sesmax
  - somax
  - workspace
  - core
  - migration
numbering:
  heading_1: true
  heading_2: true
  table: true
  code: true
---

30 Oct 2026 · J. Emmanuel Johnson (drafted with Claude Code)

## Decision

Decided: build sesmax as a uv workspace in the new `jejjohnson/sesmax` repository, and make its first package, `sesmax-core`, a copy of somax's model-agnostic machinery at somax `9cb19b3` with mechanical renames and no change in behaviour.
The ocean models, the `somax-sim` runner and the ocean diagnostics stay in somax until a second phase moves them into `sesmax-ocean`.
Rejected alternative: rename `jejjohnson/somax` to sesmax in place and split it there, as the geostack plan proposed, which keeps the git history but drops the template scaffold this repository already has.
Status: decided on 30 Oct 2026; the open questions SX-1 to SX-6 are settled in their table below, and C0 (the scaffold) has started.

## What this page settles

This page answers four questions, in order.

1. How the template in this repository becomes a workspace that holds several packages (the scaffolding).
2. Which somax modules form `sesmax-core`, and which wait for `sesmax-ocean`.
3. What gets renamed on the way, and which somax tests prove the copy is faithful.
4. In what phases the work ships, and the gate each phase must pass.

The science case, the component list (ocean, atmosphere, land, ice, coupler) and the levels L1 to L3 are in geostack's `docs/design/esmax.md` and `docs/design/stack_levels.md`.
This page only repeats what the core needs.

## Motivation

sesmax is for short-window data assimilation.
Its models are the forward models inside an assimilation window of hours to weeks, so they are judged as priors: cheap enough to run ensembles and take gradients through a whole window, good where the observations look, honest about their error, and learnable.
That puts differentiability, batching, typed slots for learned closures and learned forcings, and the adapters to filterax and vardax in the core, because every component needs them.

Three facts set the shape of the migration.

- **The name changed.** The geostack plan called the workspace esmax; it is now sesmax, "simple Earth-system models in JAX". `sesmax`, `sesmax-core` and `sesmax-ocean` all returned 404 on PyPI on 30 Oct 2026, so the names are free.
- **The repository is new.** `jejjohnson/sesmax` is an unadopted copy of the package template (`mypackage`, version 0.1.7, one initial commit `7ffcf24`). The workspace is built here, and somax keeps running until the ocean component reaches parity.
- **The core comes first.** Every later package imports it, and it is also the part of somax with the fewest dependencies on finitevolX's C-grid models, so it can be moved and tested alone.

## What changes from the esmax plan

The geostack plan's module table was inferred from somax's capability index; reading the code at `9cb19b3` corrects it in four places.

| Topic | esmax plan (geostack, 22 Oct 2026) | This page, from the code |
| --- | --- | --- |
| Names | `esmax`, `esmax-core`, `esmax_core` | `sesmax`, `sesmax-core`, `sesmax_core` |
| Repository | rename `jejjohnson/somax` | new repository `jejjohnson/sesmax`; somax archived after parity (not deleted) |
| Spatial bases | duplicates of vardax, removed | they are thin adapters over `geonnax.basis` that return somax's `SpatialBasis` with a prior standard deviation, and vardax has only four of the eight (RBF, Fourier, wavelet, EOF); they still move to vardax (SX-5), and the other four (Gabor, graph Laplacian, spherical RBF, divergence-free) go upstream first |
| `eval` | split between core and ocean | moves whole to `sesmax-ocean`: every metric takes a finitevolX C-grid `Difference2D`, and `compute_eval_metrics` dispatches on QG models |
| `models` | all to the ocean | Lorenz-63, Lorenz-96 and the 1D and 2D PDE test problems go to `sesmax_core.testing`; only the shallow-water, QG and spherical models go to the ocean |
| Modules the plan did not list | — | `basis`, `forcing_bank`, `scales`, `scaled`, `transforms`, `resolution`, `checkpoint`, `train`, `guards`, `operator_term`, `cli` (placed below) |

## The workspace

The repository holds one uv workspace, like pyrox: a root project that is never published and one directory per distribution under `packages/`.
Each distribution is imported as `sesmax_<part>`, and imports only point down the tiers.

```{mermaid}
flowchart TD
    couple["sesmax-couple<br/>tier 2 · later"]
    ocean["sesmax-ocean<br/>tier 1 · phase C2"]
    atmos["sesmax-atmos<br/>tier 1 · later"]
    core["sesmax-core<br/>tier 0 · phase C1"]
    stack["finitevolX · spectraldiffx · geonnax · pipekit<br/>filterax · vardax (extras)"]
    umbrella["sesmax<br/>code-free umbrella"]
    couple --> ocean
    couple --> atmos
    ocean --> core
    atmos --> core
    core --> stack
    umbrella -.installs.-> core
    umbrella -.installs.-> ocean
```

The tree after phase C1 (the scaffold plus the core):

```text
sesmax/
├── pyproject.toml              # sesmax-workspace, never published; [tool.uv.workspace]
├── uv.lock
├── release-please-config.json  # one entry per package, component tags
├── .release-please-manifest.json
├── Makefile
├── AGENTS.md  CLAUDE.md  CODE_REVIEW.md  README.md
├── docs/                       # one site for the workspace (MyST + MkDocs API)
│   ├── design/workspace_and_core.md
│   └── api/                    # one page per public sesmax_core module
├── scripts/capabilities.py     # workspace-aware, after pyrox's PACKAGES table
└── packages/
    ├── sesmax-core/
    │   ├── pyproject.toml
    │   ├── README.md  CHANGELOG.md
    │   ├── src/sesmax_core/
    │   └── tests/
    └── sesmax/                 # umbrella: dependencies only, no code
        ├── pyproject.toml
        └── src/sesmax/__init__.py   # __version__ only
```

## Scaffolding: adopting the template as a workspace

The template is written for one package in `src/`, so adoption has two parts: the usual rename and clean-out from the `adopt-template` skill, and the changes a workspace needs on top.

The usual adoption steps apply as written.
The example package (`stats`, `smoothing`, `transforms`, `utils`, the argparse CLI) and its tests, guide pages and notebooks are deleted, not kept as a reference, because nothing in sesmax resembles them.
Every `template-example` passage in `AGENTS.md`, `CODE_REVIEW.md`, `.github/copilot-instructions.md` and the Python coding instructions is rewritten from the core's real code, once it exists in phase C1.
The downstream plugin under `plugins/mypackage/` and `docs/llms.txt` are deleted in C0 and rewritten when sesmax has users outside the stack.

The workspace changes, taken from pyrox where it already solves them:

| Concern | Template today | sesmax |
| --- | --- | --- |
| Root `pyproject.toml` | the published package | `sesmax-workspace`, `Private :: Do Not Upload`, `bypass-selection = true`, `[tool.uv.workspace] members = ["packages/*"]`, members as workspace sources and in the `dev` group |
| Package layout | `src/mypackage/` | `packages/<dist>/src/<import>/`, each with its own `pyproject.toml`, README, CHANGELOG and tests |
| Git-only dependencies | none | finitevolX pinned by tag and geonnax by commit (as somax pins them), plus somax's `gaussx` override for filterax and vardax; `allow-direct-references = true` per package |
| release-please | one package | one entry per package with `include-component-in-tag`, its `__init__.py` and its `uv.lock` version in `extra-files`, as pyrox configures it |
| Tests and coverage | `testpaths = ["tests", "src/mypackage"]`, gate 90 % | one `testpaths` entry per package; coverage per package with its own `fail_under` (SX-6) |
| Capability index | `scripts/capabilities.py` for one package | the workspace version from pyrox (its `PACKAGES` table maps each distribution to its modules) |
| Lint, format, types | ruff on `.`, ty on the package | ruff on `.` unchanged; `ty check packages/*/src scripts` |
| CI | one install | `uv sync --all-packages --all-groups`; a bare install of each package alone in a separate job, which is what proves `sesmax-core` imports no component |
| Pytest policy | `--doctest-modules` | kept, plus somax's `filterwarnings = ["error", …]`, `xfail_strict = true` and its `slow` and `integration` markers |
| CLI | argparse console script | none in C0 and C1; the runner arrives with the ocean in C2 and uses cyclopts and loguru, as `somax-sim` already does |
| Skills | `add-function`, `add-transform`, `add-cli-command` | `add-term`, `add-forcing-basis`, `add-monitor`, `add-da-adapter` (the core's extension points), written in C1 from the moved code |

## sesmax-core

`sesmax-core` holds everything a model needs that does not know which part of the Earth system it describes: the state and parameter types, the term algebra, the model base classes, time stepping, forcing and its bases, non-dimensionalisation, the layered-fluid pieces that both an ocean and an atmosphere use, the DA adapters, training rollouts, monitors, guards and I/O.

### Where each somax module goes

Every module in `somax/_src/` at `9cb19b3` has one destination; the imports were read from the source (`git show 9cb19b3:<path>`), not run.

| somax module (`somax/_src/…`) | Holds | Goes to | Change |
| --- | --- | --- | --- |
| `core/types.py` | `State`, `Params`, `PhysConsts`, `Diagnostics`, `positive`, `interval`, `frozen`, `as_parameter`, `trainable_mask` | `sesmax_core` (top level) | none |
| `core/model.py` | `SomaxModel`, `TermModel` | `sesmax_core` (top level) | `SomaxModel` → `Model` |
| `core/terms.py`, `operator_term.py` | `Term`, `TermFn`, `Sum`, `Scaled`, `Compose`, `explicit`, `implicit`, `partition`, `build_diffrax_terms`; `OperatorTerm` | `sesmax_core.terms` | `OperatorTerm` joins the term algebra |
| `core/forcing.py`, `core/basis.py` | `ForcingProtocol` and the constant, seasonal and interpolated forcings; `SpatialBasis`, `TemporalBasis`, `BasisForcing`, `ForcingTerm`, `add_to`, `control_filter` and the vector versions | `sesmax_core.forcing` | none |
| `core/forcing_bank.py` (generic half) | `spatial_from_*` (Gabor, RBF, Fourier, graph Laplacian, EOF, wavelet, spherical RBF, divergence-free), `tile_in_time`, `GaussianWindowsInTime`, `matern_spectral_density` | `sesmax_core.forcing` | `spatial_from_{rbf,fourier,wavelet,eof}` are replaced by vardax's bases; the other four stay over geonnax until vardax ships them (SX-5) |
| `core/forcing_bank.py` (ocean presets) | `ssh_geostrophic`, `sss_coastal`, `sst_frontal`, `geostrophic_currents` | `sesmax-ocean` (C2) | named after ocean fields |
| `core/scales.py`, `core/scaled.py` | `Scales` (Rossby, Burger, Froude numbers, `dt_from_cfl`), `ScaledModel` | `sesmax_core.scales` | none |
| `core/transforms.py` | `StateAffine`; `StratificationProfile`, `ModalTransform` | `StateAffine` to `sesmax_core.scales`; the layered pieces to `sesmax_core.gfd` | split by concept |
| `core/helmholtz.py` | the periodic, Dirichlet, Neumann and multimodal Helmholtz caches over spectraldiffx | `sesmax_core.gfd` | none |
| `models/_nondim.py` | `require_positive`, `require_non_negative`, `reject_derived_kwargs`; `resolve_burger`, `burger_to_g_prime`, `resolve_burger_spherical` | validators to `sesmax_core._validation`; Burger helpers to `sesmax_core.gfd` | split by concept |
| `constants/` | physical constants | `sesmax_core.constants` | none |
| `domain/` | `Domain`, `TimeDomain` | `sesmax_core.domain` | none |
| `solvers.py` | `imex_solver`, `imex_stepsize_controller` (Krylov Newton) | `sesmax_core.solvers` | none |
| `guards.py` | `guard_finite`, `guard_positive`, `guard_ceiling` | `sesmax_core.guards` | none |
| `monitor/` | `Monitor`, `BaseMonitor`, the six built-in monitors, `default_monitors` | `sesmax_core.monitor` | none |
| `train/` | `rollout`, `rollout_loss`, `state_mse`, `ModelStep` | `sesmax_core.train` | none |
| `operators.py` | `SomaxModelOp`; `Burgers2DOp` | `ModelOp` to `sesmax_core.operators`; `Burgers2DOp` to `sesmax_core.testing` | `SomaxModelOp` → `ModelOp` |
| `da/` | `SomaxDynamics`, `SomaxForwardModel`, `SubsampleObs`, `state_to_vector`, `make_ensemble` | `sesmax_core.da`, behind `[da]` | `SomaxDynamics` → `FilteraxDynamics`, `SomaxForwardModel` → `VardaxForwardModel`; `SubsampleObs` stays until vardax#81 lands |
| `core/flows.py` | `StateAffineBijection`, `to_flowjax`, `learned_prior` | `sesmax_core.flows`, behind `[flows]` | none |
| `io/` | `state_to_dataset`, `dataset_to_state`, Zarr save and load, scale metadata | `sesmax_core.io`, behind `[xarray]` | none |
| `core/checkpoint.py` | `SimulationCheckpointer` (orbax) | `sesmax_core.checkpoint`, behind `[checkpoint]` | none |
| `models/lorenz63.py`, `lorenz96.py`, `lorenz96t.py` | the Lorenz systems on `Model` | `sesmax_core.testing` | none |
| `models/pde1d/`, `models/pde2d/` | Burgers, diffusion, linear and nonlinear convection, Navier–Stokes, Poisson | `sesmax_core.testing` | none (SX-3) |
| `models/base.py` | `DynamicalSystem`, marked for removal | dropped | — |
| `core/resolution.py` | Munk, Stommel and deformation-radius resolution checks | `sesmax-ocean` (C2) | ocean boundary layers |
| `eval/` | RMS divergence, enstrophy, KE, geostrophic imbalance, QG balance residual | `sesmax-ocean` (C2) | see the table above |
| `models/qg/`, `swm/`, `spherical/`, `gfd_testcases.py`, `_diagnostic_ops.py` | the ocean models | `sesmax-ocean` (C2) | — |
| `cli/` runner (`_run`, `_progress`, `spec`, `_units`, `_data`, `_assertions`) | the `somax-sim` run loop, run specs, units, progress logs | `sesmax_core.sim`, behind `[sim]` (C2, SX-4) | the runner learns models through a registry each component fills |
| `cli/models_registry/`, `cli/scenarios/`, `configs/`, DVC | the registered ocean models, scenarios (double gyre, Gulf Stream, …), run configs | `sesmax-ocean` (C2) | — |

`sesmax_core.testing` holds about 2,850 lines of toy models (counted with `wc -l` at `9cb19b3`).
They are the standard test problems of numerical analysis rather than Earth-system components, they import only the core and finitevolX, and the core's own tests for solvers, scaling, rollouts and operators run on them.

### Dependencies

The base install is what somax's base install already pulls for these modules; anything that somax imports lazily becomes an extra.

| Install | Brings | Needed by |
| --- | --- | --- |
| base | jax, jaxtyping, equinox, diffrax, lineax, optimistix, paramax, einops, finitevolX (tag), spectraldiffx, geonnax (commit), vardax (commit, for the spatial bases), pipekit, pipekit-jax | everything not listed below |
| `[da]` | filterax (commit), gaussx through the workspace override | `sesmax_core.da` (vardax is already in the base install) |
| `[sim]` | cyclopts, loguru, hydra-zen, omegaconf, pyyaml (C2) | `sesmax_core.sim` |
| `[flows]` | flowjax | `sesmax_core.flows` |
| `[xarray]` | xarray, zarr | `sesmax_core.io` |
| `[checkpoint]` | orbax-checkpoint | `sesmax_core.checkpoint` |

finitevolX is a base dependency for `ModalTransform` (`build_coupling_matrix`) and for the PDE toys.
Making `[da]` an extra is what somax#212 does for somax; sesmax-core starts that way, although taking vardax's bases (SX-5) puts vardax itself in the base install.
An optional module imported without its extra raises an `ImportError` that names the extra, as `somax.flows` already does.

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

### The core in use

A Lorenz-96 system stands in for any model in this sketch.
The state is $\mathbf{x} \in \mathbb{R}^{D_x}$ with $D_x = 40$ variables on a ring, the forcing is $F = 8$, and the dynamics are $\mathrm{d}x_i/\mathrm{d}t = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F$.
The sketch integrates one window, differentiates a rollout loss with respect to $F$, and hands the model to filterax; only the import lines differ from somax at `9cb19b3`.

```python
# Shapes: D_x = 40 state variables; N_E = 32 ensemble members; T = 100 snapshots
import equinox as eqx
import jax

import sesmax_core as sx
from sesmax_core.da import FilteraxDynamics, make_ensemble
from sesmax_core.testing import L96State, Lorenz96
from sesmax_core.train import rollout, rollout_loss

# today (somax: Lorenz96.create, L96State.init_state) — sesmax_core after C1
# dx_i/dt = (x_{i+1} − x_{i−2}) x_{i−1} − x_i + F
model: sx.Model = Lorenz96.create(F=8.0)
state0 = L96State.init_state(ndim=40, key=jax.random.key(0))  # (D_x,)

# today (somax: SomaxModel.integrate, diffrax underneath)
solution = model.integrate(state0, t0=0.0, t1=5.0, dt=0.01)  # final state (D_x,)

# today (somax: train.rollout, train.rollout_loss)
# L(F) = (1/T) Σₜ ‖x_{tk}(F) − x_tʳᵉᶠ‖²
ref_traj = rollout(model, state0, dt=0.01, n_steps=100)      # (T, D_x)
grad_F = eqx.filter_grad(rollout_loss)(                      # same tree as model
    model, state0, ref_traj, dt=0.01
)

# today* (somax: SomaxDynamics, make_ensemble) — renamed in C1
ensemble = make_ensemble(state0, jax.random.key(1), size=32, std=0.1)  # (N_E, D_x)
dynamics = FilteraxDynamics(model=model, template=state0)
```

## Tests that move with the core

Of somax's 62 files under `tests/`, 22 test only what moves to the core, 7 test the core through an ocean model, and the rest test the ocean models, the runner or the configs (from the imports at `9cb19b3`; not run yet).

| Group | somax test files | In sesmax |
| --- | --- | --- |
| Move whole (22) | `core/test_{checkpoint,forcing,helmholtz,model,terms,transforms,types}.py`, `domain/test_domain.py`, `da/test_{filterax,vardax}.py`, `test_{guards,monitor,scales,solvers,operators}.py`, `models/test_{burgers_terms,lorenz,pde1d,pde2d,navier_stokes,poisson}.py`, `test_pde_nondim.py` | `packages/sesmax-core/tests/` in C1, with only import changes and the four renames |
| Split (7) | `core/test_basis.py`, `core/test_operator_term.py`, `test_scaled_model.py`, `test_state_affine.py`, `test_flows.py`, `test_paramax_integration.py`, `test_train.py` | the cases on toy models move in C1; the cases on shallow-water or QG models move with the ocean in C2 |
| With the ocean (the rest) | QG, shallow-water, spherical, conservation, invariants, masks, eval, I/O, CLI, configs, registry, preflight | `packages/sesmax-ocean/tests/` in C2 |

A split test keeps its assertions; where a core feature was only tested through an ocean model, C1 adds the same assertion on a toy model so the core's coverage does not depend on C2.

## Phases and gates

Each phase ships as its own PR (stacked, in this repository) and merges only when its gate passes.

| Phase | Scope | Gate |
| --- | --- | --- |
| C0 Scaffold | adopt the template as the workspace above; `sesmax-core` and the umbrella `sesmax` as empty packages with `__version__`; CI, release-please, docs and agent files for a workspace | `uv sync --all-packages --all-groups`, ruff, ty, the public-API and capability tests and `make docs` all pass; `git grep -e mypackage -e pypackage_template` finds nothing |
| C1 Core | copy the modules above from somax `9cb19b3`; the renames; the 22 whole and 7 split test files; the four extras; API pages and the capability index | every moved somax test passes with only import and rename changes; a bare `pip install ./packages/sesmax-core` imports `sesmax_core` with none of xarray, orbax, filterax, vardax or flowjax installed; an import-boundary test fails if `sesmax_core` imports `sesmax_ocean` or `somax`; coverage of `sesmax_core` passes the low starting gate of 30 % (SX-6) |
| C2 Ocean | `sesmax-ocean` from the ocean models, presets, `resolution`, `eval`, the runner and configs; its own design page first | every remaining somax test passes under `sesmax-ocean` with only import changes; the DUACS sea-surface-height benchmark from the esmax plan runs as a baseline |
| C3 Levels | L1 in `_src/array.py`, `Step` and `simulate` behind `sesmax-core[geo]`, as geostack's SL-O2 decides | primitive modules import no carrier; `simulate` exposes `model.l1` |

C0 and C1 change no science.
Releases follow release-please: `sesmax-core` starts at 0.1.0, and the umbrella's first release waits for `sesmax-ocean`.

## somax during the migration

somax stays the working ocean package until C2 merges, so the migration must not strand work in flight.

- Once C1 merges, changes to the modules that moved land in sesmax-core first; somax gets a port only when a somax run needs the fix.
- The ocean work in flight (the spectral model zoo somax#232, hybrid terms somax#213, PhysConsts as leaves somax#220, the registry adapters somax#209) continues in somax and moves in C2.
- somax#212 (DA as an extra) can merge in somax as is; sesmax-core adopts the same split from the start.
- After C2, somax's README points to sesmax and the repository is archived, not deleted.
- C1 copies from `9cb19b3`; core commits that land in somax after that pin are listed and ported before C1 merges.

## Risks

- **Type checking is stricter here.** somax pins ty 0.0.49 and ignores nine ty rules, whereas the template runs ty ≥ 0.0.83 with none ignored, so the copied code may fail `make typecheck`. Fallback: copy somax's ignore list into `sesmax-core` and remove rules one at a time in follow-up PRs.
- **Doctests run on the copied code.** The template collects docstring examples with `--doctest-modules`, and somax never did. Fallback: fix the examples that fail, or convert them to plain `Example::` blocks, which doctest does not collect.
- **Git-pinned dependencies.** finitevolX, geonnax, filterax, vardax and gaussx come from git, so release-please builds and a PyPI release of `sesmax-core` cannot publish direct references. Fallback: keep git pins until each releases on PyPI, as the plumax and radtrax stack does with radtrax.
- **vardax's bases may not fit `BasisForcing`.** somax's `SpatialBasis` carries a prior standard deviation per mode and a `synthesize` / `analyze` pair, and vardax's basis API was not checked against it for this page (unverified). Fallback: a small adapter in `sesmax_core.forcing` that wraps a vardax basis as a `SpatialBasis`.
- **The core keeps finitevolX in its base install.** `ModalTransform` and the PDE toys need it, so a user who only wants the term algebra still installs a C-grid library. Fallback: move `gfd` and `testing` behind a `[gfd]` extra if the install size becomes a complaint.
- **The modal eigensolver.** somax fixed the non-symmetric layer matrix in its own `ModalTransform` (a symmetric similarity transform, `D^½ A D^-½`), whereas finitevolX still has the bug. Fallback: C1 keeps somax's version and its test, and does not switch to finitevolX's until finitevolX is fixed.

## Decisions on the open questions

All six were settled on 30 Oct 2026.

| ID | Question | Decision | Consequence |
| --- | --- | --- | --- |
| SX-1 | What `sesmax` installs | a code-free umbrella; `sesmax-core` is the core | the umbrella's first release waits for `sesmax-ocean` |
| SX-2 | How the code arrives | a copy at somax `9cb19b3`, with the source commit named in each C1 commit message and in the CHANGELOG | history and `git blame` stay in the archived somax repository |
| SX-3 | Where the toy models live | `sesmax_core.testing` (Lorenz and the PDE test problems) | 22 somax test files move to the core whole |
| SX-4 | Where the `somax-sim` runner lives | leaning towards the core: the generic run loop in `sesmax_core.sim` behind a `[sim]` extra, with each component registering its models and scenarios; confirmed in the C2 design page | the runner stops hard-coding the ocean registry |
| SX-5 | Spatial bases | take vardax's bases | vardax joins the core's base install; the four bases vardax lacks are proposed upstream (an ask for vardax, recorded here and not filed as a PR there) and stay over geonnax until they ship |
| SX-6 | Coverage gate | very low to start: `fail_under = 30` per package, raised in follow-ups | the template's 90 % is a target, not a C1 gate |

## Follow-up issues

- [ ] `chore(workspace): adopt the template as a uv workspace with sesmax-core and the sesmax umbrella` (C0)
- [ ] `feat(core): move somax's model-agnostic machinery into sesmax-core at 9cb19b3` (C1)
- [ ] `test(core): move somax's core tests, and add toy-model cases for the split ones` (C1)
- [ ] `docs(core): API pages, capability index and agent files from the moved code` (C1)
- [ ] `docs(design): sesmax-ocean, the ocean models, the runner and the DUACS benchmark` (C2)
- [ ] vardax: add Gabor, graph-Laplacian, spherical-RBF and divergence-free bases, so sesmax-core can drop its geonnax adapters (SX-5)

## Relationships

- Parent: jejjohnson/somax#233 ([EPIC] esmax P0–P1: rename, split and levels), whose rename item this page replaces with a new repository.
- Blocked by: nothing for C0 and C1; vardax#81 for retiring `SubsampleObs`; the vardax bases follow-up for the last four geonnax adapters.
- Blocks: `sesmax-ocean` (C2), then the atmosphere and the first coupling (somax#234).
- Related: geostack `docs/design/esmax.md`, `docs/design/stack_levels.md` (SL-O2, SL-O7); somax#212, #213, #220, #232.
