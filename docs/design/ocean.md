---
title: "sesmax-ocean: moving the ocean models from somax"
subtitle: "Design · phase C2, on gridax instead of finitevolX"
short_title: Ocean package
description: "Plans the move of somax's ocean models into sesmax-ocean: what moves, a gate by numbers in place of 'only import changes', a trial port of two models, two defects it found, the order of the pull requests and the open questions."
date: 2026-11-23
keywords:
  - sesmax
  - somax
  - ocean
  - gridax
  - migration
numbering:
  heading_1: true
  heading_2: true
  table: true
  code: true
---

23 Nov 2026 · J. Emmanuel Johnson (drafted with Claude Code)

## Decision

Proposed: move somax's ocean models from commit `9cb19b3` into a new package `sesmax-ocean`, models first and the runner later, and accept each model when it reproduces somax's numbers to round-off in float64.
The gate of phase C2 ("every remaining somax test passes with only import changes") cannot be kept, because the core moved from finitevolX to gridax in the meantime.
Alternative: keep the models on finitevolX until the move is over, which keeps that gate but gives one workspace two C-grid libraries, so an ocean state could not be read by the core's own terms.
Status: proposed, for review. A trial port of two models is done in a scratch folder; nothing is in the repository.

## What this page settles

| Question | Answer |
| --- | --- |
| Can we start now? | Yes: somax's `main` is still the pinned commit, and two ported models agree with it |
| What replaces the gate? | Equality with somax in float64: tendencies, 20 steps, diagnostics |
| What moves first? | The models, with no runner |
| What about defects? | Copied, then fixed in their own pull requests |
| What stays out? | The spectral models and the other open somax features |

## What changed since the plan

[The workspace design](workspace_and_core.md) was written on 30 October, against finitevolX.
Three things have changed since.

**gridax.**
`sesmax-core` is on gridax, pinned at `494d73d`. The ocean code imports 24 public names from finitevolX, and 16 of them are still at the top level of gridax, some with a different call.
The other eight have a home elsewhere, listed in the table below.

**Located fields.**
Since sesmax#138 a state holds arrays or `gridax.CField`s, `mask_locations` and `method="weno5"` are gone, and ghosts come from `gridax.fill_ghosts`.

**The runner.**
[The run-loop design](run_loop.md) puts the loop in `sesmax_core.sim`. It is still a proposal and nothing of it is implemented.

| finitevolX name | Replacement |
| --- | --- |
| `CartesianGrid2D`, `SphericalGrid2D` | `gridax.CartesianCGrid2D`, `gridax.SphericalCGrid2D` |
| `streamfunction_from_vorticity`, `pv_inversion` | `sesmax_core.gfd` |
| `multilayer` | `sesmax_core.vertical` |
| `coriolis_fn` | `geophysix.rotation.beta_plane` |
| `solve_cg` | gaussx's `linear_solve_with_info`, as in `sesmax_core.gfd`; gaussx becomes a declared dependency |
| `spherical_area_weights` | `gridax.area_weights`, to be checked in the spherical port |

`OMEGA` and `R_EARTH`, which the spherical models import from a private finitevolX module, come from geophysix.

(trial-port)=
## The trial port

We ported `BarotropicQG` and `NonlinearShallowWater2D` by text replacement, with array states, and compared them with somax at `9cb19b3` on finitevolX `c6f3d99`.
Each case is a 32 × 24 box with random initial fields, in float64, with 20 Heun steps.
The numbers are the largest difference divided by the largest value of somax's array.

| Model | Cases | Tendency | After 20 steps |
| --- | --- | --- | --- |
| `BarotropicQG` | two wind profiles | 8.5e-16 | 5.5e-17 |
| `NonlinearShallowWater2D` | periodic and wall, `upwind1` and `weno5` | 2.5e-16 | 1.0e-16 |

So the move is mechanical for these two, and equality to round-off is a gate we can ask for.
Not covered: `diagnose`, masks, the layered models, the reparameterised QG model and the sphere.
Its CG solve is the one place where we expect the solver's tolerance rather than round-off to set the difference.

## Two defects the port found

Both are in somax today and the ported code reproduces them.
They were measured on `NonlinearShallowWater2D`; `MultilayerShallowWater2D` calls the same operators, and it is the model of the shallow-water double-gyre configs, but we have not measured it.

**Mass.**
Today (bug): the mass equation uses the advection operator with closed outer faces, which leaves the outermost ring of cells (108 of 768 here) with a wrong tendency. The model does not conserve mass, with walls or in a periodic box, although its diagnostics call the mass conserved to machine precision: on a random state the net mass tendency is 1.3 % (walls) and 0.3 % (periodic) of the sum of the absolute ones.
With gridax's `wall="open"`, the first-order scheme agrees with a hand-written flux form in every cell, and the net tendency falls to 2e-18 and 8e-18.

**Coriolis at the edge of a periodic box.**
Today (bug): the mass and PV fluxes are averaged across the box's edge from ghost cells which no one has filled. For a uniform flow of 0.1 m/s the meridional acceleration in the first interior column is -2.5e-6 m/s², where it is -1.0e-5 everywhere else.

## What moves

The line counts are `wc -l` at `9cb19b3`.

| Group | Lines | Goes to | Phase |
| --- | --- | --- | --- |
| `models/qg/`, `swm/nonlinear_2d.py`, `swm/multilayer.py`, `spherical/` | 3,714 | `sesmax_ocean` (models) | C2a |
| `core/resolution.py` | 401 | `sesmax_ocean.resolution` | C2a |
| `eval/` | 390 | `sesmax_ocean.eval` | C2a |
| ocean presets of `core/forcing_bank.py` | 172 | `sesmax_ocean.forcing` | C2a |
| runner, run specs, units, progress, checks | 3,214 | `sesmax_core.sim`, and the ocean's checks in `sesmax_ocean.sim` | C2b |
| model registry, scenarios, dispatch | 2,030 | `sesmax_ocean.sim` | C2c |
| basin data, configs, DVC | 1,083 | `sesmax_ocean.sim` and `configs/` | C2c |

The core keeps private some helpers that the ocean models need: the positivity checks (`require_positive`, `reject_derived_kwargs`), the Burger-number helpers, the diagnostics bridge (`fvx_ops`, `masked`) and the element-wise helpers of located fields (`each`, `values`, `placed`).
`require_non_negative`, which the ocean factories call, is not in the core.
A second package must not import `_src`, so the core makes them public first.

## Phases and gates

Each item is one pull request, stacked in this order.

| Phase | Pull requests | Gate |
| --- | --- | --- |
| C2a Models | scaffold; public helpers in the core; barotropic QG; layered and reparameterised QG; shallow water; sphere; resolution, eval and presets; the two fixes; notebooks | equality with somax, and its ocean tests with their assertions unchanged |
| C2b Run loop | `sesmax_core.sim` (its own design's SIM-1, then SIM-2) | the run-loop design's checks, and an exact restart for each stepper |
| C2c Runs | registry, scenarios, configs and data in `sesmax_ocean.sim` | somax's eight configs build and give somax's `t = 0` state and first chunk |
| C2d Benchmark | the DUACS sea-surface-height baseline | the QG model's score is reported against DUACS |

**The equality gate.**
A script runs somax at `9cb19b3` and writes the initial states, tendencies, 20-step end states and diagnostics of each model to one `.npz` file, which the tests read.
The trial port's file for six cases is 309 kB.
A case passes when the relative difference is below 1e-13 for tendencies and diagnostics and below 1e-12 after 20 steps; the spherical QG model is compared at its CG tolerance.

**Tests.**
somax's tests for these models (356 functions by name, before parametrisation) move with their assertions.
Only imports, grids, `method=` and ghost handling change in them.

**Package mechanics.**
A workspace member with a `[tool.uv.sources]` entry, a `dev` group entry and a release-please component; the umbrella depends on it; `tests/test_workspace.py` gains the tier; the capability index and `docs/api/` gain its pages; its readme and icon follow the family of the other packages.

## somax after the move

somax is archived after C2c and not before, because its runner and configs are the only way to make a reference run until then.
Its open features (the spectral model zoo somax#232, hybrid terms somax#213, the SSH-QG and internal-tide models somax#141, the registry adapters somax#209, self-describing output somax#206) are not part of the move.
Each gets a new issue in sesmax, or is closed, when somax is archived.

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| OC-1 | The C2 gate | equality by numbers | stay on finitevolX and keep the old gate |
| OC-2 | Located fields in the ocean states | from the first port: arrays and `CField`s, as in the core | arrays now, `CField`s later |
| OC-3 | The two defects | copy, then fix in the same stack with the numbers before and after | fix while porting |
| OC-4 | Runner | after the models (C2b, C2c) | together, as the plan had it |
| OC-5 | DUACS benchmark | its own phase, as it needs data and compute that CI does not have | inside C2a |

## Relationships

- Parent: jejjohnson/somax#233.
- Follows: [the workspace and core design](workspace_and_core.md), whose C2 row this page replaces, and [the run-loop design](run_loop.md), which C2b implements.
- Related: sesmax#90 (periodic helpers), sesmax#104 and sesmax#109 (the ocean's energy budget and Jacobian record), sesmax#115 (real basins), jejjohnson/somax#141, #206, #209, #213, #232.
