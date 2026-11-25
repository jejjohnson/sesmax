---
title: "The packages of the sesmax workspace"
subtitle: "Design · core, param, components, testing and experiment, with benchmarks in geobench"
short_title: Workspace packages
description: "Rethinks which packages the sesmax workspace holds: a core of contracts, a package for the parametric part of a model, one package per component, a testing package with the toy models and the checks, an experiment package for simulation, estimation and monitoring, and benchmarks outside the repository."
date: 2026-11-25
keywords:
  - sesmax
  - workspace
  - parameterization
  - forcing
  - testing
  - experiment
  - geobench
numbering:
  heading_1: true
  heading_2: true
  table: true
  code: true
---

23 Nov 2026, revised 25 Nov 2026 · J. Emmanuel Johnson (drafted with Claude Code)

## Decision

Proposed: give the workspace five kinds of package. `sesmax-core` holds the contracts, `sesmax-param` the parametric part of a model (parameterizations, forcing and model error), each component of the Earth system is a package of models, `sesmax-testing` says whether a model is right, and `sesmax-experiment` is where models are used: simulation, parameter estimation, state estimation, bilevel problems and monitoring.
Benchmarks, which say whether a model is good, live in the planned geobench repository.
Alternative: keep one core and add modules to it as needed (`sesmax_core.testing`, `sesmax_core.sim`), which is fewer releases, but the core then carries every toy and every workflow to anyone who only wants the model contract.
Status: proposed, for review. The ocean models of W1 are merged (sesmax#152); the rest is not started. The owner has settled five points: the toys and the checks are one package, the experiment package covers more than simulation, there is a package for the parametric part, model error belongs in it, and the ocean models come before the split (25 November).

Each new package has its own design page, with the motivation, the maths, the numerics, the API, examples and a prototype: [sesmax-testing](sesmax_testing.md), [sesmax-param](sesmax_param.md) and [sesmax-experiment](sesmax_experiment.md). This page says what the packages are and how they depend on each other.

## What this page settles

| Question | Answer |
| --- | --- |
| Where do the toy models live? | `sesmax-testing`, with the checks |
| Where do drag, closures, forcing and model error live? | `sesmax-param` |
| Where do runs and estimation live? | `sesmax-experiment` |
| Who owns the methods? | vardax, filterax, argminax, as today |
| Where do benchmarks live? | geobench |
| When do we split? | After the ocean models, before the run loop |

## The packages

| Tier | Package | Holds | Imports from the workspace |
| --- | --- | --- | --- |
| 0 | `sesmax-core` | contracts, terms, forcing, scales, adapters | nothing |
| 1 | `sesmax-param` | parameterizations, forcing, model error | core |
| 2 | `sesmax-ocean`, later `-atmos`, `-land`, `-ice` | models and their scenarios | core, param |
| 2 | `sesmax-testing` | toy models and checks | core |
| 2 | `sesmax-experiment` | problems, runs, estimation, monitoring | core, param |
| 3 | `sesmax-couple` | coupled models | core, param, components |
| – | `sesmax` | no code, installs the set | – |

Three rules go with the tiers.

- **Two packages of one tier never import each other.** A component does not know that experiments exist, and the experiment package names no component.
- **No package imports `sesmax-testing` at run time.** It is a development dependency, of the core as well.
- **Only public names cross packages.** An import of another package's `_src` fails `tests/test_workspace.py`.

`tests/test_workspace.py` reads the tiers today from a list of forbidden names for the core alone; it becomes a table of tiers that covers every package.

## The core

The core keeps what a model is made of and what a model is handed to: the types and constraints, `Model`, the terms, the scales, the layered-fluid code, stepping, `rollout`, the assimilation adapters, and I/O.
Of the forcing it keeps the slot: `ForcingProtocol`, and `ForcingTerm` with `add_to`, which lift a field into a tendency.

Four things leave it. The lines are counted at `222d092`, where the whole core has 16,047, and for the forcing at `3c64811`.

| Leaves | Lines | Goes to |
| --- | --- | --- |
| `sesmax_core.testing` | 4,572 | `sesmax_testing` |
| `sesmax_core.forcing`, but for the slot | 2,847 with it | `sesmax_param` |
| `sesmax_core.monitor` | 490 | `sesmax_experiment.monitor` |
| `sesmax_core.checkpoint` | 119 | `sesmax_experiment`, behind `[checkpoint]` |

Nothing in the core calls a monitor, and the run loop which will call them is an experiment.
The guards stay, because a guard is part of a tendency.
The forcing is the one part whose move lightens the install, as nothing else in the core imports geonnax. gridax stays, because other modules of the core import it.

The core gains two things the other packages need.

- **Public helpers.** The positivity checks, the Burger-number helpers, the diagnostics bridge, and `each`, `values` and `placed` for located fields are private today, and every component needs them.
- **`Component`.** The small type in [the run-loop design](run_loop.md) through which a component registers its models, checks and units. It sits in the core so that a component can register without importing `sesmax-experiment`.

(param)=
## sesmax-param: what do the resolved dynamics not supply?

Split the right-hand side of a model into the resolved dynamics and the part whose form we choose or learn,

```math
\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = f(\mathbf{x}) + p(\mathbf{x}; \boldsymbol{\theta}) + u(t; \mathbf{w}) + \boldsymbol{\eta}(t)
```

where $\mathbf{x} \in \mathbb{R}^{D_ x}$ is the state, $f: \mathbb{R}^{D_ x} \to \mathbb{R}^{D_ x}$ the resolved dynamics (advection, pressure, Coriolis), $p$ a parameterization with parameters $\boldsymbol{\theta} \in \mathbb{R}^{D_ \theta}$, $u(t; \mathbf{w}) \in \mathbb{R}^{D_ x}$ a forcing with coefficients $\mathbf{w} \in \mathbb{R}^{D_ w}$, and $\boldsymbol{\eta}(t) \in \mathbb{R}^{D_ x}$ the model error.

The last three terms are the package. Each is parametric: a formula, a basis expansion or a network, whose parameters carry a prior.
"Can be learned" is not the test, because every parameter of $f$ is a leaf too, gravity and layer depth included. The test is whether the term's form is ours to choose.

| Part | Reads the state? | Prior | Today |
| --- | --- | --- | --- |
| Parameterization $p$ | yes | on $\boldsymbol{\theta}$ | drag and Rayleigh damping; `OperatorTerm` takes a network |
| Forcing $u$, known | no | none | `WindStress2D`, `InterpolatedForcing` |
| Forcing $u$, estimated | no | on $\mathbf{w}$ | `BasisForcing` |
| Model error $\boldsymbol{\eta}$ | no | on $\boldsymbol{\eta}$ | only in vardax and filterax |

**Parameterization or forcing.**
The second column decides. Drag and Rayleigh damping are filed under `forcing` today and move to the parameterizations, because they read the state.

**Model error is a forcing we do not know.**
vardax writes it as $\boldsymbol{\eta} = \boldsymbol{\chi} \mathbf{W} \boldsymbol{\Phi}^\top$ (`ModelErrorBasis`), a space-time basis times coefficients with a prior, which is the form of `BasisForcing`. One basis can therefore serve both.
They differ in units. vardax adds its increment to the state after each step, so it is in state units, and a forcing is a tendency. The package has to say which one a prior describes.
An ensemble draws $\boldsymbol{\eta}$ instead of estimating it, and the core has only hooks for that: a model may override `build_terms` for an SDE, and `as_forward_model` takes a `diffusion`.
What the estimated error shows to be systematic is what a parameterization should absorb.

**One hierarchy.**
The core has two that do not meet. `ForcingProtocol` maps a time and a grid to a field and has a wrapper into the term algebra; `AbstractForcing` leaves the call to each subclass, has its own `ForcingSum` and `ForcingProduct`, and has no wrapper.
In the package everything is a `Term`, and the sums are the term algebra's.

**What exists.**
The package starts with the forcing, the drag and the damping. There is no Smagorinsky, Leith, Gent–McWilliams or backscatter scheme in sesmax, somax or gridax, so the sub-grid half starts empty.

| Concern | Owner |
| --- | --- |
| diffusion operators and stencils | gridax |
| `Term`, `OperatorTerm`, the forcing slot | `sesmax-core` |
| bulk fluxes between components | `sesmax-couple` |
| turbulence closures of plumes | plumax |
| fitting any of it | `sesmax-experiment` |

**The models.**
All seven ocean models write wind and drag into their `vector_field`. They become sums of terms once they have been ported, so that the port can still be compared with somax, and the comparison then holds the rewrite to the same numbers.

**The readme and the figures.**
Forcing is one of the five uses in the core's readme, its quickstart inverts for a `BasisForcing`, and geostack#299 draws it. All three change in W5.

**The name.**
"Closure" names the sub-grid part only, and "physics" does not fit a network. "Param" is short for what the parts share. It is close to `Params`, the base class of a model's parameters, and to paramax, so prose says "parameterization" in full and keeps "parameter" for a leaf.

(testing)=
## sesmax-testing: is the model right?

The toy models are for testing, so they live with the checks they are tested by.

| Part | Holds | Today |
| --- | --- | --- |
| `sesmax_testing` | Lorenz, the PDEs up to Navier–Stokes, simple shallow water | `sesmax_core.testing`, 75 names |
| `sesmax_testing.checks` | functions that take any `Model` | scattered in tests, or missing |

The 75 names keep their spelling, so a caller changes one import. No alias stays behind in the core, as no other repository imports it.

A check is a function of a model and a state that raises with the numbers when the model fails.

| Check | Question |
| --- | --- |
| conservation | does the net tendency of an invariant vanish? |
| translation | does a shifted state give a shifted tendency, in a periodic box? |
| reference | are the numbers those of a stored run? |
| gradient | does `jax.grad` agree with finite differences (sesmax#105)? |
| adjoint | is $\langle \mathbf{M} \mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{v}, \mathbf{M}^\top \mathbf{w} \rangle$? |
| convergence | does the error fall at the order of the scheme? |
| contract | `jit`, `vmap`, and the same numbers with arrays and located fields |

Here $\mathbf{M} \in \mathbb{R}^{D_ x \times D_ x}$ is the tangent-linear model over a window, and $\mathbf{v}, \mathbf{w} \in \mathbb{R}^{D_ x}$ are random directions.
The first two are the checks which found the defects in [the ocean plan](ocean.md). somax's own conservation, invariant and multilayer tests (30 in three files) pass at `9cb19b3` all the same; the mass test starts from a nearly uniform state and allows a drift of 1e-6.

vardax ships its own Lorenz-63 and Lorenz-96. It is below sesmax in the stack and cannot import ours, so the two stay separate.

The core's tests depend on the toys: 16 of its 46 test files outside `tests/testing` import them, as do two doctests.
They stay in the core's suite and take `sesmax-testing` from the development group.
The twelve files of `tests/testing` move with the toys.
22 of the 28 notebooks and both README quickstarts change one import.

(experiment)=
## sesmax-experiment: what do we do with the model?

Everything we do with a model asks for a different part of the same control. Collect the initial state, the parameters and the forcing coefficients in one vector $\mathbf{z} \in \mathbb{R}^{D_ z}$, with prior mean $\boldsymbol{\mu}_ z \in \mathbb{R}^{D_ z}$ and covariance $\boldsymbol{\Sigma}_ z \in \mathbb{R}^{D_ z \times D_ z}$, and write $\mathbf{x}_ t(\mathbf{z}) \in \mathbb{R}^{D_ x}$ for the state after $t$ of $T$ steps. The cost of a window is

```math
\mathcal{J}(\mathbf{z}; \boldsymbol{\nu}) = \frac{1}{2} \Vert \mathbf{z} - \boldsymbol{\mu}_z \Vert^2_{\boldsymbol{\Sigma}_z^{-1}} + \frac{1}{2} \sum_{t \leq T} \Vert \mathbf{y}_t - h_t(\mathbf{x}_t(\mathbf{z})) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}}
```

where $\mathbf{y}_ t \in \mathbb{R}^{D_ y}$ are the observations, $h_ t: \mathbb{R}^{D_ x} \to \mathbb{R}^{D_ y}$ the observation operators, $\boldsymbol{\Sigma}_ \varepsilon \in \mathbb{R}^{D_ y \times D_ y}$ the error covariance, and $\boldsymbol{\nu} \in \mathbb{R}^{D_ \nu}$ holds what the cost itself depends on (a prior scale, a closure, a covariance).

| Work | Unknown | Answer | Method from |
| --- | --- | --- | --- |
| Simulation | none | $\mathbf{x}_ t(\mathbf{z})$ for a given $\mathbf{z}$ | diffrax, `pipekit_cycle.Cycle` |
| State estimation | $\mathbf{x}_ 0$, or $\mathbf{x}_ t$ in cycles | $\mathrm{argmin}_ {\mathbf{x}_ 0} \mathcal{J}$, or a filter | vardax, filterax, `pipekit_cycle.DACycle` |
| Parameter estimation | $\boldsymbol{\theta}$ | $\mathrm{argmin}_ {\boldsymbol{\theta}} \mathcal{J}$, or an ensemble process | optimistix, argminax, `filterax.EKI` |
| Forcing estimation | forcing coefficients | $\mathrm{argmin}_ {\mathbf{u}} \mathcal{J}$ | `vardax.ThreeDVar` with `WindowOperator` |
| Bilevel | $\boldsymbol{\nu}$ | see below | the adjoints of vardax and argminax |
| Monitoring | none | health of a run, and of an analysis | the monitors, `filterax.utils` |

A bilevel problem learns $\boldsymbol{\nu}$ through the estimate of the inner one, with an outer loss $\mathcal{L}$:

```math
\hat{\boldsymbol{\nu}} = \mathrm{argmin}_{\boldsymbol{\nu}} \ \mathcal{L}(\hat{\mathbf{z}}(\boldsymbol{\nu}))
```

where the inner estimate is

```math
\hat{\mathbf{z}}(\boldsymbol{\nu}) = \mathrm{argmin}_{\mathbf{z}} \ \mathcal{J}(\mathbf{z}; \boldsymbol{\nu})
```

### What it owns

It owns no method. Every algorithm in the table above already has a home, and the package holds what is missing between them.

- **The problem.** One object says which leaves of a model, a state and a forcing are unknown, and reads the priors that `sesmax-param` put on them. Today that is spread over `control_filter`, `trainable_mask`, `SpatialBasis.prior_std` and the `std` of `make_ensemble`, and none of them covers the joint $\mathbf{z}$.
- **The work.** A thin function per row that takes a problem, hands it to a method and returns the estimate with a record.
- **The run.** The chunk loop, checkpoints, restarts, run specs, artifacts and the command line of [the run-loop design](run_loop.md), which become the simulation part of this package.
- **The monitoring.** The monitors of a run, and the statistics of an analysis (innovations, spread against skill) with the same verdict and the same record.
- **Twin experiments.** Truth from a model, observations from the truth, an estimate, and its error against the truth.

### The call

A sketch, with names that are proposals.

```python
# Shapes: D_x = 40 variables; T = 10 steps; D_z = D_x + 1
import sesmax_experiment as sxe

# proposed (sesmax-experiment)
# z = (x_0, F), with priors 𝒩(μ_x, σ_x² I) and 𝒩(8, 1)
problem = sxe.Problem(
    model=model,
    window=sxe.Window(dt=0.05, n_steps=10),
    unknowns={
        "state": sxe.Gaussian(background, 0.5),
        "params.F": sxe.Gaussian(8.0, 1.0),
    },
    observations=observations,
)

# proposed (sesmax-experiment)
# x_t(z) at the prior mean
run = sxe.simulate(problem)
# proposed (sesmax-experiment), the method from vardax
# ẑ = argmin 𝒥(z)
# → (D_z,)
result = sxe.estimate(problem, method=vardax_method)
# proposed (sesmax-experiment), the adjoint from vardax
# ν̂ = argmin ℒ(ẑ(ν)), with ν the prior scale
tuned = sxe.bilevel(problem, outer=outer_loss, over="prior_scale", method=vardax_method)
```

State and parameter estimation are one verb, `estimate`, because the problem says what is unknown.

### What it does not own

| Concern | Owner |
| --- | --- |
| minimisers, curvature, evidence | argminax, optimistix |
| variational costs and their adjoints | vardax |
| filters, smoothers, ensemble processes | filterax |
| cycles and protocols | pipekit-cycle |
| training an emulator or an amortised map | pipekit-train |
| scores and reports | pipekit-evaluate |
| tracking, model registry | pipekit-experiment, then `geobench-experiment` |

**The name.**
geostack's layer map moves pipekit-experiment to `geobench-experiment`, so two packages of the stack end in "experiment". They differ by what they hold: ours has the problems and the work, geobench's has the tracking of any run.
sesmax imports neither geobench nor its tracker. A run writes its own manifest, and a monitor may forward it to any object with the tracker's methods.

**Extras.**
The base install needs only the core. `[da]` brings vardax and filterax, `[checkpoint]` orbax, and `[cli]` what the run-loop design lists for `[sim]`.

This section draws the boundary only. The problem object, the records and the command line get their own design page, with a prototype, before any code.

(geobench)=
## geobench: is the model good?

| | sesmax-testing | geobench |
| --- | --- | --- |
| Question | is the code right? | is the model good? |
| Compares with | equations, invariants, stored numbers | data and other methods |
| Data | toy, in the repository | frozen, versioned |
| Fails | a pull request | nothing, it reports |

The esmax plan makes "beats the best statistical prior, with the same observations" a gate of each component. Nothing below geobench may import it, so the gate becomes a report that geobench publishes, starting with sea-surface height against DUACS.
Timing the steppers (sesmax#107) compares our own code with itself and stays in sesmax.

## What we do not split

| Candidate | Why it stays |
| --- | --- |
| assimilation adapters (444 lines) | an extra does the job, and the README uses them without an experiment |
| flows, I/O | extras do the job |
| layered-fluid code | one user until there is an atmosphere |
| forcing presets of the ocean | named after ocean fields, so `sesmax-ocean` |
| scenarios | a double gyre is ocean science; the run of it is the experiment |

A package costs a release, a CI job, a README and an icon, so a new one needs a second user or a different set of dependencies.

## Order of work

| Step | Pull requests | Gate |
| --- | --- | --- |
| W1 | public helpers of the core; the ocean models ([ocean plan](ocean.md)) | the equality test against somax |
| W2 | `sesmax-testing`: the toys, then the checks | the moved tests and notebooks give the same numbers; the new checks run on every toy and every ocean model |
| W3 | `sesmax-experiment`: monitors, checkpoints, the run loop | the run-loop design's checks |
| W4 | the ocean's scenarios and configs; somax is archived | somax's configs give somax's first chunk |
| W5 | `sesmax-param`: the forcing, drag and damping move; one hierarchy; model error; the ocean models as sums of terms | the moved tests; the reference check of every ocean model, unchanged |
| W6 | `sesmax-experiment`: problems, estimation, bilevel, twins | its own design page |

The ocean is first because its gate needs one equality test, small enough for its own tests, and not the split. That test becomes the reference check of `sesmax-testing` in W2.
W5 comes after the port, which needs models with their physics inside to compare with somax, and before W6, which reads the priors.

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| WP-1 | Toys' import path | `from sesmax_testing import Lorenz96` | a `sesmax_testing.models` submodule |
| WP-2 | Monitors | move to `sesmax-experiment` with the loop | stay in the core |
| WP-3 | `rollout_loss` | stays in the core, next to `rollout` | moves to `sesmax-experiment` |
| WP-4 | `Component` | in the core | an entry point with no shared type |
| WP-5 | Name of the command | `sesmax-experiment simulate …` | `sesmax-sim`, as in the run-loop design |
| WP-6 | Units of a model-error prior | a tendency, converted for vardax by the adapter | state increments, as vardax has them |
| WP-7 | Noise that scales with a parameterization | left out until a user needs it | in the package from the start |
| WP-8 | vardax's `ModelErrorBasis` | built from our basis by the adapter | one basis class, in vardax |

## Relationships

- Changes: [the workspace and core design](workspace_and_core.md) (its tier table and C2 to C3 rows) and [the run-loop design](run_loop.md) (`sesmax_core.sim` becomes `sesmax_experiment`). Both keep their text until this page is accepted.
- Needs in geostack: the layer map and the dependency graph list the new packages.
- Related: sesmax#29 (levels), sesmax#105, sesmax#106 (scale-selective stabilisation, a parameterization), sesmax#107, sesmax#114 (bases from functax), somax#213 (hybrid terms), somax#234 (atmosphere and coupling).
