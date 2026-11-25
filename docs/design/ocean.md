---
title: "sesmax-ocean: moving the ocean models from somax"
subtitle: "Design · phase C2, on gridax instead of finitevolX"
short_title: Ocean package
description: "Plans the move of somax's ocean models into sesmax-ocean: what moves, a gate by numbers in place of 'only import changes', a trial port of all seven models, two defects it found, the order of the pull requests and the decisions. Builds on the workspace packages page."
date: 2026-11-25
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

23 Nov 2026, revised 25 Nov 2026 · J. Emmanuel Johnson (drafted with Claude Code)

## Decision

Decided on 25 November 2026: move somax's ocean models from commit `9cb19b3` into a new package `sesmax-ocean`, and accept each model when it reproduces somax's numbers to round-off in float64.
The package is a component in the sense of [the workspace packages](workspace_packages.md): it holds models and scenarios, its equality test lives in its own tests until `sesmax-testing` exists, and its runs go to `sesmax-experiment`.
The gate of phase C2 in the workspace design ("every remaining somax test passes with only import changes") cannot be kept, because the core moved from finitevolX to gridax in the meantime.
Alternative: keep the models on finitevolX until the move is over, which keeps that gate but gives one workspace two C-grid libraries, so an ocean state could not be read by the core's own terms.
Status: decided. A trial port of all seven models is done in a scratch folder; the pull requests follow the steps below.

## What this page settles

| Question | Answer |
| --- | --- |
| Can we start now? | Yes: somax's `main` is still the pinned commit, and all seven ported models agree with it, the two spherical ones after a half-cell shift |
| What replaces the gate? | Equality with somax in float64: tendencies, 20 steps, diagnostics |
| What moves first? | The models; `sesmax-testing` comes later |
| What about defects? | Copied, then fixed in their own pull requests |
| What stays out? | The spectral models and the other open somax features |

## What changed since the plan

[The workspace design](workspace_and_core.md) was written on 30 October, against finitevolX.
Three things have changed since.

**gridax.**
`sesmax-core` is on gridax v0.3.1. The operator classes the ocean models hold (`Difference2D`, `Interpolation2D`, `Vorticity2D`, `Advection2D`, `Diffusion2D` and their spherical twins) no longer exist: a model holds one `GridOps`, the functions of gridax on its grid and mask, as the core's models do.
The other names have a home elsewhere, listed in the table below.

**Located fields.**
Since sesmax#138 a state holds arrays or `gridax.CField`s, `mask_locations` and `method="weno5"` are gone, and ghosts come from `gridax.fill_ghosts`.

**Spherical latitudes.**
gridax places the T rows of a spherical grid at cell centres (gridax#433). finitevolX at `c6f3d99` places them half a cell further south, and its cell areas are approximate where gridax's are exact.

**The runner.**
[The run-loop design](run_loop.md) put the loop in `sesmax_core.sim`, and nothing of it is implemented. It becomes the simulation part of `sesmax-experiment`.

| finitevolX name | Replacement |
| --- | --- |
| `CartesianGrid2D`, `SphericalGrid2D` | `gridax.CartesianCGrid2D`, `gridax.SphericalCGrid2D` |
| the operator classes | one `gridax.GridOps` per model |
| `enforce_periodic`, `zero_boundaries` | `gridax.fill_ghosts` with a `BC` |
| `streamfunction_from_vorticity`, `pv_inversion` | `sesmax_core.gfd` |
| `multilayer` | `sesmax_core.vertical` |
| `coriolis_fn` | `geophysix.rotation.beta_plane` |
| `solve_cg` | gaussx's `linear_solve_with_info`, as in `sesmax_core.gfd` |
| `spherical_area_weights` | `gridax.area_weights`, exact where finitevolX's is approximate |

`OMEGA` and `R_EARTH`, which the spherical models import from a private finitevolX module, come from geophysix.

(trial-port)=
## The trial port

We ran all seven models of somax `9cb19b3` through a thin adapter from the old class names onto gridax v0.3.1 and sesmax-core `566b3c7`, with array states, and compared them with somax on finitevolX `c6f3d99`.
Each case is a 32 × 24 grid with random initial fields, in float64, with and without a mask, and 20 Heun steps (5 for the spherical QG model).
The masks were all water by a mistake in the script; the equality tests of the package have land.
The numbers are the largest difference over the interior cells divided by the largest value of somax's array, over every case of a model.

| Model | Tendency | End state | Diagnostics |
| --- | --- | --- | --- |
| Barotropic QG | 6.5e-16 | 8.5e-17 | 1.6e-15 |
| Baroclinic QG | 6.5e-15 | 4.4e-16 | 9.0e-15 |
| Reparameterized QG | 2.5e-14 | 1.3e-14 | 2.4e-14 |
| Nonlinear shallow water | 2.5e-16 | 9.8e-17 | 9.8e-16 |
| Multilayer shallow water | 1.8e-16 | 2.8e-15 | 1.8e-15 |
| Spherical shallow water, shifted | 8.3e-16 | 1.9e-16 | 5.6e-4 |
| Spherical QG, shifted | 6.5e-12 | 1.3e-13 | 5.6e-4 |

The shallow-water cases cover a periodic box and walls, with `upwind1` and `weno5`.
"Shifted" means that somax runs with its latitude range moved north by half a cell, which puts its rows where gridax has them; without it the spherical tendencies differ by about 0.2.
The 5.6e-4 is in the area-weighted diagnostics only, and it is the difference between approximate and exact cell areas. The fields themselves agree, at the tolerance of the CG solve (1e-10) for the QG model.

So the move is mechanical, and equality to round-off is a gate we can ask for.
Not covered: gradients, located fields, float32, and a state with velocities on land.

## Two defects the port found

Both are in somax today and the ported code reproduces them.
They were measured on `NonlinearShallowWater2D`, and the mass defect on `MultilayerShallowWater2D` too, which is the model of the shallow-water double-gyre configs.

**Mass.**
Today (bug): the mass equation uses the advection operator with closed outer faces, which leaves the outermost ring of cells (108 of 768 here) with a wrong tendency. The model does not conserve mass, with walls or in a periodic box, although its diagnostics call the mass conserved to machine precision: on a random state the net mass tendency is 1.3 % (walls) and 0.3 % (periodic) of the sum of the absolute ones.
With gridax's `wall="open"`, the first-order scheme agrees with a hand-written flux form in every cell, and the net tendency falls to 2e-18 and 8e-18. The three-layer model has the same defect: its net mass tendency is 0.08 % (walls) and 0.8 % (periodic) of the sum of the absolute ones, and 4.5e-18 with `wall="open"`.

**Coriolis at the edge of a periodic box.**
Today (bug): the mass and PV fluxes are averaged across the box's edge from ghost cells which no one has filled. For a uniform flow of 0.1 m/s the meridional acceleration in the first interior column is -2.5e-6 m/s², where it is -1.0e-5 everywhere else.

## What moves

The line counts are `wc -l` at `9cb19b3`.

| Group | Lines | Goes to | Step |
| --- | --- | --- | --- |
| `models/qg/`, `swm/nonlinear_2d.py`, `swm/multilayer.py`, `spherical/` | 3,714 | `sesmax_ocean` (models) | W1 |
| `core/resolution.py` | 401 | `sesmax_ocean.resolution` | W1 |
| `check_equatorial_deformation_radius` of `cli/_assertions.py`, with its helper | 122 | `sesmax_ocean.resolution` | W1 |
| `eval/` | 390 | `sesmax_ocean.eval` | W1 |
| ocean presets of `core/forcing_bank.py` | 172 | `sesmax_ocean.forcing`; their imports change again in W5 | W1 |
| runner, run specs, units, progress, checks | 3,214 | `sesmax_experiment`; the ocean's own checks in `sesmax_ocean` | W3 |
| model registry, scenarios, dispatch | 2,030 | `sesmax_ocean.scenarios` | W4 |
| basin data, configs, DVC | 1,083 | `sesmax_ocean.scenarios` and `configs/` | W4 |

The core keeps private some helpers that the ocean models call (the positivity checks, the Burger-number helpers, the diagnostics bridge and the element-wise helpers of located fields), and `require_non_negative`, which the ocean factories call, is not in it.
The first pull request of W1 makes them public.

Three imports inside somax set the order. The `from_nondimensional` factories of the QG models call the resolution checks, `ReparameterizedQG` is built on `MultilayerShallowWater2D`, and `SphericalSWM` calls `check_equatorial_deformation_radius`, which lives in the runner today.

## Steps and gates

The steps are those of [the workspace packages](workspace_packages.md). Each item is one pull request, stacked in this order.

| Step | Pull requests | Gate |
| --- | --- | --- |
| W1 Models | public helpers of the core; scaffold with resolution, eval and presets; barotropic and baroclinic QG; shallow water and reparameterized QG; sphere; the two fixes; notebooks | equality with somax, and its ocean tests with their assertions unchanged |
| W3 Run loop | in `sesmax-experiment` | the run-loop design's checks, and an exact restart for each stepper |
| W4 Scenarios | registry, scenarios, configs and data | somax's eight configs build and give somax's `t = 0` state and first chunk |

The DUACS sea-surface-height baseline leaves this plan: it is a benchmark, so it goes to geobench, which reports the QG model's score.

**The equality gate.**
A script runs somax at `9cb19b3` and writes the initial states, tendencies, 20-step end states and diagnostics of each model to one `.npz` file per model family, which the package's own tests read; the test moves to the reference check of `sesmax-testing` when that exists.
The two fixes are each checked by a conservation test and a translation test, which fail before them.
A case passes when the relative difference over the interior cells is below 1e-12 for tendencies, end states and diagnostics.
The spherical models are compared with somax on the shifted latitude range, the QG model at 1e-9, and their area-weighted diagnostics are recomputed with gridax's areas.

**Tests.**
somax's tests for these models (356 functions by name, before parametrisation) move with their assertions.
Only imports, grids, `method=` and ghost handling change in them.

**Package mechanics.**
A workspace member with a `[tool.uv.sources]` entry, a `dev` group entry and a release-please component; the umbrella depends on it; `tests/test_workspace.py` gains the tier; the capability index and `docs/api/` gain its pages; its readme and icon follow the family of the other packages.

## Wind and drag after the move

The ported models keep wind, drag and viscosity inside `vector_field`, as somax has them.
In W5 they become sums of terms, with the wind and the drag from [`sesmax-param`](workspace_packages.md), and the reference check holds the rewrite to the numbers of the port.

## somax after the move

somax is archived after W4 and not before, because its runner and configs are the only way to make a reference run until then.
Its open features (the spectral model zoo somax#232, hybrid terms somax#213, the SSH-QG and internal-tide models somax#141, the registry adapters somax#209, self-describing output somax#206) are not part of the move.
Each gets a new issue in sesmax, or is closed, when somax is archived.

## Decisions

Decided on 25 November 2026.

| ID | Question | Decision | Alternative |
| --- | --- | --- | --- |
| OC-1 | The C2 gate | equality by numbers | stay on finitevolX and keep the old gate |
| OC-2 | Located fields in the ocean states | from the first port: arrays and `CField`s, as in the core | arrays now, `CField`s later |
| OC-3 | The two defects | copy, then fix in the same stack with the numbers before and after | fix while porting |
| OC-4 | Runner | after the models (W3, W4) | together, as the plan had it |
| OC-5 | `sesmax-testing` | after the models; the equality test starts in the package's own tests | before them |

## Relationships

- Parent: sesmax#152 (the epic of W1); jejjohnson/somax#233 before it.
- Follows: [the workspace and core design](workspace_and_core.md), whose C2 row this page replaces, [the workspace packages](workspace_packages.md), and [the run-loop design](run_loop.md), which W3 implements.
- Related: sesmax#90 (periodic helpers), sesmax#104 and sesmax#109 (the ocean's energy budget and Jacobian record), sesmax#115 (real basins), jejjohnson/somax#141, #206, #209, #213, #232.
