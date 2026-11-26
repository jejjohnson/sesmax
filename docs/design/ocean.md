---
title: "sesmax-ocean: the ocean models, their scenarios and their checks"
subtitle: "Design · moved from somax onto gridax, and what the package is today"
short_title: Ocean package
description: "What sesmax-ocean holds at 26 November 2026: seven models, the scenarios and configs, the basin bundles, the checks and metrics, the stack it is built on, the defects the port found and fixed, the equality gate that holds it to somax's numbers, and what is left."
date: 2026-11-26
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

23 Nov 2026, revised 25 and 26 Nov 2026 · J. Emmanuel Johnson (drafted with Claude Code)

## Decision

Decided on 25 November 2026: move somax's ocean models from commit `9cb19b3` into a new package `sesmax-ocean`, and accept each model when it reproduces somax's numbers to round-off in float64.
The package is a component in the sense of [the workspace packages](workspace_packages.md): it holds models and scenarios, and its runs go to `sesmax-experiment`.
Alternative, rejected: keep the models on finitevolX until the move was over, which would have given one workspace two C-grid libraries.
Status on 26 November 2026, at sesmax `a34e46c`: built. The models (W1), the scenarios, the configs and the basin bundles (W4) are merged, and the run loop they use (W3) is in `sesmax-experiment`. The package is at version 0.0.0 and has not been released. What is left is at the end of this page.

## What this page settles

| Question | Answer |
| --- | --- |
| What is in the package? | seven models, one scenario, eight configs, four placeholder basins, checks and metrics |
| How do we know the move is right? | each model equals somax to 1e-12 in float64, with the fixes since put back |
| What did the move find? | nine defects of somax, each fixed in its own pull request |
| What is it built on? | `sesmax-core`, gridax, gaussx and geophysix |
| What stays out? | the spectral models, real basins, a benchmark, and wind and drag as terms |

## The package today

`sesmax-ocean` is 6,992 lines of source in nine public modules, with 497 test functions in 37 files (counted at `a34e46c`).
It imports only `sesmax-core` from the workspace, and `tests/test_workspace.py` holds it to that.

| Module | Holds |
| --- | --- |
| `sesmax_ocean.qg` | `BarotropicQG`, `BaroclinicQG`, `ReparameterizedQG`, with their states, parameters and diagnostics |
| `sesmax_ocean.swm` | `NonlinearShallowWater2D`, `MultilayerShallowWater2D` |
| `sesmax_ocean.spherical` | `SphericalSWM`, `SphericalQG`, on a latitude band |
| `sesmax_ocean.scenarios` | `Scenario`, `Geometry`, `ForcingFields`, the registries of scenarios and models, `build_ocean_run`, `check_compatible` |
| `sesmax_ocean.basins` | `Basin`, `placeholder_bundle`, `write_bundle`, behind `[data]` |
| `sesmax_ocean.resolution` | `check_munk_width`, `check_stommel_width`, `check_deformation_radius`, `check_equatorial_deformation_radius` |
| `sesmax_ocean.checks` | `check_pv_inversion`, `check_static_stability` |
| `sesmax_ocean.eval` | `compute_eval_metrics`, and the metrics it picks from |
| `sesmax_ocean.forcing` | `ssh_geostrophic`, `sss_coastal`, `sst_frontal`, `geostrophic_currents` |

**The models.**
Each is a `Model` of the core with a state of arrays or of `gridax.CField`s, one `gridax.GridOps` for its stencils, and `gridax.fill_ghosts` with a `BC` per field for its boundaries.
Wind, drag and viscosity are still written inside `vector_field`, as somax had them.

**The component.**
The package registers itself through the entry point `sesmax.components`, as a `Component` of the core.
`sesmax-experiment` finds the models by name from a run spec, and names no ocean model in its own code.

**The scenarios and the configs.**
A run spec names a scenario and a model, and `build_ocean_run` builds the two.
Of somax's six scenarios only `double_gyre` can be built; the others wait for the data of their basins (sesmax#115).
somax's eight run configs are in `packages/sesmax-ocean/configs/`, each with a `debug` block that makes it small and short.

**The basins.**
`sesmax-ocean data` builds and fetches the bundles of four basins (Gulf Stream, Mediterranean, North Atlantic, Southern Ocean) with DVC.
Each bundle is a placeholder with the layout of the real one, and no scenario reads one yet.

## What it is built on

The stencils, solvers and constants are those of the stack, and the package writes none.

| Need | From | How it is used |
| --- | --- | --- |
| the model contract, terms, scales, located fields | `sesmax-core` | `Model`, `State`, `each`, the validators, the Burger-number helpers |
| the streamfunction and PV inversions | `sesmax_core.gfd` | the QG models |
| vertical modes and layer coupling | `sesmax_core.vertical` | the multilayer and baroclinic models |
| grids, masks, stencils, advection, ghost fills | gridax | one `GridOps` per model, `fill_ghosts`, `area_weights` |
| the conjugate-gradient solve on the sphere | gaussx | `SphericalQG` |
| the Earth's radius and rotation rate, the β-plane | geophysix | the spherical models and the factories |

Two things are still written here that belong to another layer, and both have a plan.

- **Wind, drag and viscosity** sit inside each `vector_field`. They become terms of [`sesmax-param`](sesmax_param.md), and the equality gate holds the rewrite to today's numbers.
- **The metrics** of `sesmax_ocean.eval` (kinetic energy, enstrophy, geostrophic imbalance, RMS divergence) overlap with gridax's diagnostics and with `diagnostix.skill`. sesmax#166 takes the spectra from diagnostix; the rest should follow it.

## The equality gate

The gate of phase C2 in [the workspace design](workspace_and_core.md), "every remaining somax test passes with only import changes", could not be kept, because the core moved from finitevolX to gridax in the meantime.
What replaced it is equality by numbers.

`packages/sesmax-ocean/tests/reference/make_reference.py` runs somax at `9cb19b3` and writes the initial states, tendencies, end states and diagnostics of each model family to an `.npz` file.
The `test_equal_to_somax.py` files compare the ported models with them, on arrays and on located fields, with land in the masked cases.
A case passes at a relative 1e-12 in float64, and `SphericalQG` at 1e-9, since its streamfunction is solved to a tolerance.
The configs have the same gate: each starts where somax's does, and its first 20 steps agree to 1e-12.

Every fix below changed a number, so each is its own pull request after the port.
That pull request puts somax's numerics back in a test subclass (`packages/sesmax-ocean/tests/support/somax_numerics.py`), so the rest of the model stays under the gate.

Three differences from somax are by design and not defects.
gridax reads no dry cell, so beside land the Laplacian takes no flux through a coast.
The T rows of a spherical grid are at cell centres and the cell areas are exact, where somax's rows were half a cell too far south and its areas approximate.
The default radius and rotation rate are geophysix's.

## What the move found

The port copied somax's numerics first and fixed them afterwards, each fix with a test that fails before it.
All of these were in somax at `9cb19b3`.

| Defect | Models | Fixed by |
| --- | --- | --- |
| mass is not conserved: no flux through the edge faces | shallow water, spherical shallow water | sesmax#171 |
| the momentum tendency reads unfilled ghosts at a periodic edge | shallow water | sesmax#174 |
| the viscous term diffuses a velocity as a tracer | shallow water | sesmax#175 |
| the Coriolis parameter is half a cell too far north | `SphericalSWM` | sesmax#176 |
| the velocity and the fluxes at the edges of the band | `SphericalQG` | sesmax#177 |
| the Coriolis term is halved beside walls and coasts | shallow water | sesmax#182 |
| a tendency on two of the walls adds up there | shallow water | sesmax#184 |
| the states a run returns have unfilled ghosts | every model, in the core | sesmax#201 |
| the potential vorticity is not held on all four walls | QG | sesmax#213 |

The first two were found by the trial port below; the others by the checks written since.
The mass defect measured 1.3 % (walls) and 0.3 % (periodic) of the sum of the absolute tendencies on a random state, and 2e-18 and 8e-18 after the fix.
The same wall defect is still open in the core's linear toys (sesmax#202).
The conservation, translation and boundary tests that found them live in this package's tests today, written for three models by name; [`sesmax-testing`](sesmax_testing.md) makes them one check for any model.

One test of somax passed for the wrong reason: `pv_inversion` inverted the initial state of a run, which is zero for every barotropic model. Here it inverts a random field.

(trial-port)=
## The trial port, before the move

This section is the evidence the decision rested on, kept as it was written on 25 November.

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

## Steps, as they went

| Step | Scope | Pull requests | State |
| --- | --- | --- | --- |
| W1 Models | public helpers of the core; scaffold with resolution, metrics and presets; the seven models; the fixes; tutorials | sesmax#163 to sesmax#184, sesmax#195, sesmax#199, sesmax#205, sesmax#213 | merged; the epic sesmax#152 is open |
| W3 Run loop | chunks, run specs, checkpoints, checks, a command | sesmax#197, sesmax#198, sesmax#203, sesmax#207, sesmax#208 | merged, in `sesmax-experiment` |
| W4 Scenarios | registry, scenarios, configs; basin bundles and their commands | sesmax#210, sesmax#211 | merged; the epic sesmax#186 is open |

The tutorial series gained step 17 (the wind-driven double gyre in shallow water) and step 18 (the barotropic QG double gyre).

## What is left

| Item | Where |
| --- | --- |
| the double-gyre notebook, eight years of spin-up | sesmax#214, open pull request |
| real basins: bathymetry and forcing regridded at build time, and the five scenarios that need them | sesmax#115, sesmax#225, sesmax#226, sesmax#227 |
| wind, drag and viscosity as terms | [`sesmax-param`](sesmax_param.md), step P6 |
| the conservation, translation and boundary tests as generic checks | [`sesmax-testing`](sesmax_testing.md), step T2 |
| spectra and spectral fluxes of a validation run, from diagnostix | sesmax#166 |
| a spectral energy budget, and a record of the Jacobian scheme | sesmax#104, sesmax#109 |
| a QG prior against DUACS, as a benchmark | sesmax#236, and geobench |
| the spectral model zoo | sesmax#215 to sesmax#221 |
| a first release of the package | release-please, with the umbrella |
| archiving somax | sesmax#194 |

## Decisions

Decided on 25 November 2026, and kept.

| ID | Question | Decision | What happened |
| --- | --- | --- | --- |
| OC-1 | The C2 gate | equality by numbers | the `test_equal_to_somax.py` files |
| OC-2 | Located fields in the ocean states | from the first port: arrays and `CField`s | both are tested |
| OC-3 | The defects | copy, then fix in the same stack, with the numbers before and after | nine fixes, each its own pull request |
| OC-4 | Runner | after the models | W3 and W4 followed W1 |
| OC-5 | `sesmax-testing` | after the models | not started; its design is [here](sesmax_testing.md) |

## Relationships

- Parent: sesmax#152 (the epic of W1) and sesmax#186 (W3 and W4); jejjohnson/somax#233 before them.
- Follows: [the workspace and core design](workspace_and_core.md), whose C2 row this page replaces, [the workspace packages](workspace_packages.md), and [the run-loop design](run_loop.md), which W3 implemented.
- Feeds: [sesmax-testing](sesmax_testing.md), [sesmax-param](sesmax_param.md) and [sesmax-experiment](sesmax_experiment.md).
- Related: sesmax#104, sesmax#109, sesmax#115, sesmax#166, sesmax#194, sesmax#202, sesmax#214, sesmax#236.
