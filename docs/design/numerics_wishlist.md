---
title: "A numerics and physics wishlist"
subtitle: "Design · what the literature supports adding to sesmax for stability and model health"
short_title: Numerics wishlist
description: "Ranks what a literature review supports adding to sesmax for long-run stability and model health, each with its evidence and a follow-up issue, and records which claims failed verification and what the review could not settle."
date: 2026-11-05
keywords:
  - sesmax
  - numerics
  - stability
  - diagnostics
numbering:
  heading_1: true
  heading_2: true
  table: true
  code: true
---

5 Nov 2026 · J. Emmanuel Johnson

## Decision

Proposed: track eight items, with local blow-up signals, a spectral energy budget and gradient checks first, because each one tells us something about runs we already do.
Nothing here is implemented, and each item has its own issue under #8.
The engineering half of the same review, on checkpoints, schedules and run records, went into [the run-loop design](run_loop.md).

## How to read the evidence

A literature review extracted 118 claims from 24 sources and had the budget to check 25 of them, with three independent votes on each; 14 survived and 11 failed.
The items below rest on the surviving claims, and the order is our judgement, not any source's.
The fit to sesmax is inference throughout, because only pyqg matches our layered QG directly, and the other sources are a hybrid atmosphere, a nonhydrostatic spectral-element model, an isopycnal ocean and inviscid QG.

## The wishlist

| ID | Item | Issue | Priority | Home | We have today |
| --- | --- | --- | --- | --- | --- |
| NW-1 | Largest local tendency and high-wavenumber tail, per chunk | #103 | 1 | core | `NonFiniteMonitor`, `EnergyGrowthMonitor` |
| NW-2 | Layered QG spectral energy budget, with a residual | #104 | 1 | ocean | nothing |
| NW-3 | Finite-difference gradient checks, and a measured gradient horizon | #105 | 1 | core | some gradient tests |
| NW-4 | Scale-selective stabilisation as a configured term | #106 | 2 | core | Laplacian `lateral_viscosity` in the models read |
| NW-5 | Stepper benchmarks for each configuration | #107 | 2 | core | `imex_solver`, with no measurements |
| NW-6 | Stability as a survival fraction over an ensemble | #108 | 2 | core | nothing |
| NW-7 | Jacobian scheme on record, with a statistics test | #109 | 3 | ocean | the three-term Arakawa Jacobian |
| NW-8 | A second literature pass on what this one missed | #110 | 2 | docs | this page |

### NW-1: local signals, not only energy

Energy can stay perfect while a flow fails.
In a [shallow-water test](https://gmd.copernicus.org/articles/13/581/2020/), a scheme held energy at round-off while the flow still collapsed, about a month later than with RK4 in a flow that broke after about 7 years.
That an energy monitor would have missed it is our inference.
[Griffies and Hallberg](https://www.gfdl.noaa.gov/bibliography/related_files/smg0002.pdf) make the same point about viscosity: in a five-layer double gyre, a constant biharmonic coefficient set from an average of the local values went unstable, because it must control the largest deformation rates anywhere.
In [NeuralGCM](https://arxiv.org/pdf/2311.07222) case studies, local signals came first: unrealistic local tendencies about 3 days before an instability had spread, and a high-wavenumber signal about 16 days before.

Those are two case studies in a hybrid model, with no false-alarm rates.
So the monitors start as record-only, and #103 measures lead times on our own forced blow-ups before any threshold exists.

### NW-2: a budget with its residual

[pyqg](https://pyqg.readthedocs.io/en/latest/equations/notation_layered.html) writes $\partial_t E(k, l)$ for energy $E$ at wavenumbers $(k, l)$ as five terms: kinetic and potential energy flux divergence, potential energy generation, bottom drag and small-scale dissipation.
Its docs assume that the last is small and the others dominate at statistical steady state, but this is an assumption with no measured closure or threshold, so sesmax must set its own tolerance.
pyqg is doubly periodic, while our double gyre has walls and wind, so the derivation has to be redone.
The 256 × 256 baroclinic spin-up reached year 5 with kinetic energy still rising, and this is the tool that would say why.

### NW-3: gradients with a known horizon

[FESOM2-JAX](https://arxiv.org/abs/2608.01546) states that a gradient through chaotic dynamics is informative only over finite horizons, citing Lea, Allen and Haine (2000) and Köhl and Willebrand (2002), the second of which includes QG twin experiments.
This fits a short-window prior in direction only, because no source gives a horizon for models like ours, and reports range from hours to about 278 days.
So #105 measures it, and also adds per-term finite-difference checks, which FESOM2-JAX runs away from non-differentiable switches.
That preprint's costs, 4.7 times a forward step and 0.83 GiB per stored step, belong to its mesh and GPU.

### NW-4: stabilisation you can see

[NeuralGCM](https://arxiv.org/pdf/2311.07222) says filter strength sets the true length scale of a simulation.
[In HOMME-NH](https://arxiv.org/pdf/1906.07219), operator-split hyperviscosity limits temporal accuracy to first order, so convergence studies turn it off, although [Vogl et al.](https://arxiv.org/pdf/1904.10115) say the limit usually appears only well below operational steps.
[Griffies and Hallberg](https://www.gfdl.noaa.gov/bibliography/related_files/smg0002.pdf) found 18 times more kinetic energy with biharmonic than Laplacian friction by day 800 in one channel run, and also found Laplacian useful for coarse, weak flows.
finitevolX has `BiharmonicDiffusion2D` and spectraldiffx has hyperviscosity, so this is wiring rather than new numerics.

### NW-5: benchmark steppers, don't adopt a ranking

[Steyer et al.](https://arxiv.org/pdf/1906.07219) show IMEX rankings shifting with planet size in HOMME-NH, and their stronger claim that one family wins everywhere failed verification.
[NeuralGCM](https://arxiv.org/pdf/2311.07222) treats linear gravity waves implicitly to take longer steps, and it still needs filters.
The source measured the largest usable step, so #107 also measures accuracy at a fixed step, across diffrax's solvers, finitevolX's `RK3SSP` and `IMEX_SSP2`, and `imex_solver`.

### NW-6: survival fractions

[NeuralGCM](https://arxiv.org/pdf/2311.07222) reports 35 of 37 two-year runs and 22 of 37 forty-year runs finishing without instability, then analyses only the stable ones.
A claim of stability should carry its denominator, and any later statistic should state that selection.
Our models are cheap enough to run such an ensemble under `vmap`.

### NW-7: a recorded scheme

[Dubinkina and Frank (2007)](https://research.vu.nl/en/publications/statistical-mechanics-of-arakawas-discretizations/) predict and confirm different statistics for Arakawa's three schemes in inviscid flow, with a PV fluctuation at their monitor point of 5.36 for the energy scheme against 0.970 and 1.01 for the combined and enstrophy schemes.
This hasn't been shown for forced-dissipative models, and the broader claim that discretisation is a first-order control on long-run statistics failed verification.
finitevolX's `arakawa_jacobian` is the three-term scheme, so this is a line in the manifest and a test, not a design driver.

### NW-8: the second pass

The first pass covered too little to be the last word, as the next two sections show, so #110 tracks a focused follow-up.

## Claims that failed verification

A failed claim means that its wording didn't survive three votes, not that the opposite is true.
We don't rely on any of these.

| Claim | Votes |
| --- | --- |
| Differentiating through longer rollouts improves free-running stability (NeuralGCM) | 0-3 |
| Discretisation is a first-order control on long-run statistics (Dubinkina and Frank) | 0-3 |
| Round-off energy conservation needs both an antisymmetric operator and a square-conservative integrator | 0-3 |
| RK4 lost about 0.5 % of energy in the final year, against about $10^{-15}$ for NRK4, as reference magnitudes for a drift check | 1-2 |
| IMEX methods designed from H-stability beat literature methods everywhere | 0-3 |
| A parallel enstrophy budget from pyqg as a spin-up check | 0-3 |
| Closed-form viscosity bounds that can be checked at run time (Griffies and Hallberg) | 0-3 |
| Bit-for-bit restart as a scoped CI gate (FESOM2-JAX) | 0-3 |
| Rolling and archival restart streams with one job per invocation (FESOM2-JAX) | 0-3 |
| The one-step Jacobian spectrum as a gradient diagnostic | 0-3 |
| Monte Carlo averaging can't rescue gradients over long unrolls | 0-3 |

## What the review didn't settle

- **Time stepping.** Nothing survived on SSP Runge-Kutta, CFL-adaptive stepping or implicit diffusion.
- **Established models.** No claim from MOM6, NEMO, MITgcm, Veros, SpeedyWeather, jax-cfd or GeoClaw reached the confirmed set. The budget for checks ran out before most of them, so this says nothing against those sources.
- **Physics checks.** Rossby-number and resolution checks weren't covered. `somax` has its own, moving to `sesmax-ocean` in C2.
- **Our own models.** These need our own measurements: the gradient horizon, the lead time of local signals in pure dynamics, and what restart guarantee holds across JAX versions.

## Relationships

- Depends on: [the run-loop design](run_loop.md), for the health record and the manifest
- Tracked in: #8
- Issues: #103, #104, #105, #106, #107, #108, #109, #110
