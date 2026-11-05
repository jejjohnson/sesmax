---
title: "sesmax-core as a shared core"
subtitle: "Design · what the core should own so that sesmax-ocean, plumax and later components stop rebuilding it"
short_title: Shared core
description: "Puts sesmax-core below every GeoModels repository that integrates in time, reports a proof that plumax's transport model fits the core's term algebra, and lists the candidates to add, each with its API and its use in every library that would inherit it."
date: 2026-11-04
keywords:
  - sesmax
  - plumax
  - radtrax
  - core
  - time stepping
numbering:
  heading_1: true
  heading_2: true
  table: true
  code: true
---

4 Nov 2026 · J. Emmanuel Johnson

## Decision

Proposed: design `sesmax-core` as the layer every time-stepping GeoModels repository can build on, with `sesmax-ocean` and plumax as its first two consumers, and add only what no library already owns: a temporal basis for time-resolved controls, a stable-step bound, a whole-window operator for source inversion, and a thin method that hands a model to `pipekit_jax.DiffraxForwardModel`.
plumax designs against this API but does not switch over until the core and its stack are on PyPI.
radtrax does not depend on the core.
Status: proposed, for review. Nothing here is implemented, and the follow-up issues are filed, with #8 tracking them.

## What this page settles

| Question | Answer |
| --- | --- |
| Can the layering work? | Yes: `sesmax-core` sits below both `sesmax-ocean` and plumax, and the import graph stays acyclic |
| Does a non-ocean model fit the core's contracts? | Yes: plumax's transport model, as a `TermModel`, gives the same tendency to the last bit |
| What is repeated across the stack today? | Little is copied, but several things are written again under other names; [the audit](#repetition-audit) gives an owner for each |
| What is worth adding? | Four small candidates and two upstream asks, each with its API and its use in each consumer |
| What should stay out? | radtrax, plumax's physics and met fields, and observation operators |
| What has to happen first? | The pins have to converge and the stack has to reach PyPI |

## The layering

somax and radtrax share a tier in geostack's dependency graph, with plumax one above them.
`sesmax-ocean` takes somax's place, and `sesmax-core` is a new layer below that tier.

```{mermaid}
flowchart TD
    plumax["plumax<br/>plume models"]
    ocean["sesmax-ocean<br/>replaces somax"]
    radtrax["radtrax<br/>radiative transfer"]
    core["sesmax-core<br/>the shared core"]
    stack["finitevolX · spectraldiffx · diffrax · pipekit · vardax · filterax"]
    plumax --> radtrax
    plumax --> core
    ocean --> core
    core --> stack
```

Every arrow points down, and the rule that repositories of the same tier never import each other still holds, because `sesmax-ocean` and radtrax are peers.
plumax needs only the core, never `sesmax-ocean`.

The two repositories number their tiers differently: sesmax's `AGENTS.md` calls the core tier 0 and the ocean tier 1, whereas geostack calls somax tier 0.
This page therefore names the layers rather than numbering them, and geostack's `docs/api/dependency_graph.md` needs the new layer.
`tests/test_workspace.py` already fails if `sesmax_core` imports a component or somax, and it should cover plumax and radtrax too.

## The proof: plumax's transport model as a `TermModel`

`plumax.les_fvm.EulerianDispersionRHS` computes $\partial c / \partial t$ for a tracer on a 3D C-grid.
It is advection, diffusion and a source in one `__call__`, which is what the core's term algebra exists to express.
So we rebuilt it as a `TermModel` on plumax's own stencils, in a scratch directory, and compared the two on a $40 \times 16 \times 8$ grid with a uniform wind, plumax's default boundaries and a source whose rate varies in time.
plumax was at `origin/main` in its own environment, with the core on `PYTHONPATH`.

| Check | Result |
| --- | --- |
| Largest difference in the tendency, at $t = 0$ and $t = 37$ s | 0, against a largest tendency of $3.8 \times 10^{-4}$ |
| Largest difference after a Heun run to $t = 200$ s | $1.8 \times 10^{-12}$, against a largest concentration of $7.8 \times 10^{-6}$ |
| Wall time of that run, direct and as a `TermModel` | 4.7 s and 4.0 s |
| $\partial (\text{mass}) / \partial (\text{source scale})$ through the solver | $1.054564 \times 10^{-3}$, against a central difference of $1.054560 \times 10^{-3}$ |
| Tagging diffusion `implicit` | `build_terms` returns a `MultiTerm` with no other change |

```python
# Shapes: Nz, Ny, Nx with the ghosts = 10, 18, 42; G = Nz × Ny × Nx
import diffrax as dfx
import equinox as eqx
import finitevolx as fvx
import jax.numpy as jnp
from jaxtyping import Array
from plumax.les_fvm import make_grid, uniform_wind_field
from plumax.les_fvm.advection import advection_tendency
from plumax.les_fvm.boundary import (
    apply_boundary_conditions,
    build_default_concentration_bc,
    periodic_axes,
)
from plumax.les_fvm.diffusion import diffusion_tendency, make_eddy_diffusivity
from plumax.les_fvm.source import make_gaussian_source

import sesmax_core as sx


# today (sesmax_core)
class Tracer(sx.State):
    c: Array


# today (sesmax_core)
class Advect(sx.Term):
    grid: eqx.Module
    wind: eqx.Module
    horizontal: eqx.Module
    vertical: eqx.Module

    def __call__(self, t, state, args=None):
        # today (plumax): advection reads a ghost that holds the exterior value
        ghosted = apply_boundary_conditions(
            state.c,
            horizontal_bc=self.horizontal.advection_variant(),
            vertical_bc=self.vertical.advection_variant(),
            plume_grid=self.grid,
        )
        # today (plumax)
        u, v, w = self.wind(t)  # → G each
        periodic = periodic_axes(self.horizontal)
        # today (plumax)
        # ∂c/∂t ⊇ −∇·(u c)
        tendency = advection_tendency(  # → G
            ghosted, u, v, w, self.grid, method="weno5", periodic=periodic
        )
        # today (finitevolx)
        return Tracer(c=fvx.interior(tendency[1:-1, 1:-1, 1:-1], tendency))


# today (sesmax_core)
class Diffuse(sx.Term):
    grid: eqx.Module
    eddy: eqx.Module
    horizontal: eqx.Module
    vertical: eqx.Module

    def __call__(self, t, state, args=None):
        # today (plumax): diffusion reads a ghost that reflects about the face
        ghosted = apply_boundary_conditions(
            state.c,
            horizontal_bc=self.horizontal,
            vertical_bc=self.vertical,
            plume_grid=self.grid,
        )
        periodic = periodic_axes(self.horizontal)
        # today (plumax)
        # ∂c/∂t ⊇ ∇·(K ∇c)
        tendency = diffusion_tendency(  # → G
            ghosted, self.eddy, self.grid, periodic=periodic
        )
        # today (finitevolx)
        return Tracer(c=fvx.interior(tendency[1:-1, 1:-1, 1:-1], tendency))


# today (sesmax_core)
class Emit(sx.Term):
    source: eqx.Module

    def __call__(self, t, state, args=None):
        # today (plumax)
        # ∂c/∂t ⊇ q(t) ρ(s)
        tendency = self.source(t)  # → G
        # today (finitevolx)
        return Tracer(c=fvx.interior(tendency[1:-1, 1:-1, 1:-1], tendency))


# today (plumax)
grid = make_grid((0.0, 2000.0, 40), (-400.0, 400.0, 16), (0.0, 200.0, 8))
wind = uniform_wind_field(grid, u=3.0, v=0.5, w=0.0)
eddy = make_eddy_diffusivity((10.0, 2.0))
horizontal, vertical = build_default_concentration_bc()


def rate(t):
    # q(t) = 1 + ½ sin(2π t / 100)
    return 1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * t / 100.0)  # → ()


source = make_gaussian_source(grid, rate, (500.0, 0.0, 50.0))

# today (sesmax_core)
# ∂c/∂t = −∇·(u c) + ∇·(K ∇c) + q(t) ρ(s)
model = sx.TermModel(
    terms=Advect(grid, wind, horizontal, vertical)
    + Diffuse(grid, eddy, horizontal, vertical)
    + Emit(source)
)
# today (sesmax_core, diffrax)
solution = model.integrate(  # c → 1 × G
    Tracer(c=jnp.zeros(grid.shape)),
    t0=0.0,
    t1=200.0,
    dt=1.0,
    solver=dfx.Heun(),
    max_steps=1000,
)
```

One thing did not map directly, and it matters for the design.
The core's `apply_boundary_conditions(state)` fills the ghosts once per state, but advection and diffusion need different Dirichlet ghosts, as plumax's comments explain, so here each term fills its own and the model's method stays a pass-through.
The contract in `AGENTS.md` should say that this is allowed.

(repetition-audit)=
## Repetition audit

Before adding anything, we parsed every definition in finitevolX, spectraldiffx, gaussx, geonnax, vardax, filterax, pipekit, pipekit-cycle, pipekit-jax, sesmax, somax, plumax and radtrax, then compared function bodies with local names normalised and listed the definitions for each concept.

Very little is copied. Leaving out sesmax and somax, whose 143 shared functions are the deliberate copy that ends when somax is archived, only two functions of five or more statements are identical across repositories: `_coerce_step` in pipekit-cycle and pipekit-jax, and `split_keys` in gaussx and geonnax.
The repetition is re-implementation under other names, which a body comparison cannot catch, so the table below comes from reading the definitions.

| Concept | Owner | Also written in | Action |
| --- | --- | --- | --- |
| Horizontal advection and diffusion | finitevolX `Advection*`, `Diffusion*` | thin wrappers: plumax `advection_tendency`, `diffusion_tendency`; sesmax `_burgers_flux`, `_advective_form` | keep the wrappers thin; SC-6 removes the periodic logic they repeat |
| Vertical advection and diffusion | nobody: `Advection3D` and `Diffusion3D` act per level | plumax `vertical_advection_tendency`, `vertical_diffusion_tendency` | move to finitevolX (SC-7) |
| Periodic edge handling | nobody | plumax `advection_tendency`; sesmax `_burgers_flux`, `_advective_form`, `IncompressibleNS2D._wrap` | move to finitevolX (SC-6) |
| Solver, controller, adjoint and step budget | pipekit-jax `DiffraxForwardModel` | plumax `solver_kwargs`, `_pick_solver`; vardax `_DynamicalForwardModel`; the core's `integrate(**kw)` | reuse it (SC-1) |
| One step as a `ForwardModel` | pipekit-jax `DiffraxForwardModel` | plumax `EulerianForwardModel`; the core's `ModelOp`, `VardaxForwardModel`, `FilteraxDynamics` | plumax drops its adapter; the core keeps what adds flattening or pipelines |
| Explicit Runge–Kutta schemes | diffrax, finitevolX `RK3SSP` and others | no copies found | use them through `solver=` |
| Time-varying fields | diffrax interpolation, through the core's `InterpolatedForcing` | plumax `WindSchedule`, `interpolate_in_time`, `_time_bracket`, `_interpolated_emission` | plumax can move at S4 |
| Vertical modes | finitevolX `decompose_vertical_modes`, fixed in its #252 | the core's `ModalTransform.from_physics` | see below the table |
| Helmholtz and Poisson solves | spectraldiffx | nothing since the core's `HelmholtzCache` family, which only forwarded to it, was removed | call spectraldiffx directly |
| Spatial bases | geonnax `basis` | vardax's constructors and the core's `spatial_from_*`, both delegating to it | see below the table |
| Stable step | nobody | plumax `stable_step_bound`; somax `check_cfl`; the core's `Scales.dt_from_cfl`, from scales | SC-4 |
| Observation operators | vardax | pipekit-cycle has its own `CallableObs` and `LinearObs`; the core's `SubsampleObs` | #27; the two names are [vardax#217](https://github.com/jejjohnson/vardax/issues/217) |
| Whitening | gaussx | vardax `_whitener`; radtrax `WhiteningTransform`; the core's `BasisForcing.whiten` | check when vardax work reaches them; not compared |
| Filter statistics | gaussx | filterax `kalman_gain` and `ensemble_covariance` delegate; its `gaspari_cohn`, `inflate_*` and `sigma_points` are written out | filterax already tracks this in its #119 to #124; `gaspari_cohn` differs by up to 0.49 at the same arguments, so its `radius` and gaussx's `c` are not the same parameter |
| Lorenz systems | none | `vardax.Lorenz63`, `vardax.Lorenz96`; `sesmax_core.testing` | accept: vardax cannot import sesmax, which sits above it |
| Error classes | each package | vardax, plumax and radtrax each define `ParameterError`, `ShapeError` or `GridError` | accept: one base per package is the house style |

Three of these need more words.

**Vertical modes.** The workspace design says finitevolX still has the eigensolver bug, and that is out of date. Rossby radii from the two agree to $7 \times 10^{-11}$ on three stratifications in double precision, so `ModalTransform` no longer needs its own eigensolve for correctness. On the first of those, its transforms equal finitevolX's to $10^{-16}$ after one overall factor of 0.1, so delegating would rescale modal coefficients, and the switch needs a decision and a test rather than a silent swap.

**Spatial bases.** Decision SX-5 said to take vardax's bases, but vardax's constructors delegate to `geonnax.basis`, as the core's already do, so that would add a hop without removing a repeat. The core stays on geonnax, and #25 and #26 are closed as not planned.

**The core's own repeats.** `scales._require_positive` repeats `_validation.require_positive`, and among the test models, the same `apply_boundary_conditions` body appears in four 1D models and the same `diagnose` in the same four. Both are small cleanups.

This audit also changed this page's candidates: a first draft proposed an `Integrator` class and two tracer terms, and neither survived.

## The candidates

| ID | Candidate | Home | `sesmax-ocean` | plumax | radtrax |
| --- | --- | --- | --- | --- | --- |
| [SC-1](https://github.com/jejjohnson/sesmax/issues/85) | `Model.as_forward_model`, over `DiffraxForwardModel` | `sesmax_core.Model` | cycling DA | replaces `EulerianForwardModel` | no |
| [SC-2](https://github.com/jejjohnson/sesmax/issues/86) | `HatInTime`, a temporal basis | `sesmax_core.forcing` | time-resolved wind control | emission time series | no |
| [SC-4](https://github.com/jejjohnson/sesmax/issues/87) | `stable_step` | `sesmax_core.solvers` | runner preflight | `stable_step_bound` | no |
| [SC-5](https://github.com/jejjohnson/sesmax/issues/88) | `WindowOperator` | `sesmax_core.da` | forcing estimation | `fourdvar.py` | no |
| [SC-6](https://github.com/jejjohnson/finitevolX/issues/403) | `wall="periodic"` | upstream, finitevolX | through the core | `advection.py` | no |
| [SC-7](https://github.com/jejjohnson/finitevolX/issues/404) | vertical advection and diffusion | upstream, finitevolX | later | `_vertical_ops.py` | no |

SC-3, tracer transport terms in the core, is withdrawn; its section says why.
None of the proposed names exists in the capability index today, and the concepts were checked against the libraries above.

Some of what plumax needs already ships, so these are not candidates:

| plumax today | Stack today |
| --- | --- |
| `EulerianForwardModel`, which takes one explicit Euler step | `pipekit_jax.DiffraxForwardModel` |
| `WindSchedule`, `MetField.wind_at`, for a field through time | `InterpolatedForcing`, whose `from_data` already interpolates a tuple of components |
| `GaussianSource` | `BasisForcing` with a one-column `SpatialBasis`, lifted by `ForcingTerm` |
| `_to_dataset`, in part | `snapshots_to_dataset` |

### SC-1: reuse `DiffraxForwardModel`

A run needs a solver, a step controller, an adjoint and a step budget, and each consumer has grown its own handling of them: plumax has `_pick_solver` and a `solver_kwargs` dict, and #80 added `adjoint` to `VardaxForwardModel`.
`pipekit_jax.DiffraxForwardModel` already bundles all four, satisfies `pipekit_cycle.ForwardModel`, and has `as_dynamics()` for the `(state, t0, t1)` form that filterax uses. pipekit-jax is already a base dependency of the core, so a new settings class would be one more repeat.

| Check, at the same solver and step | Result |
| --- | --- |
| plumax's `EulerianDispersionRHS` through it, against a direct `diffeqsolve` | largest difference 0 |
| A core `Lorenz96` through it, against `Model.step` | largest difference 0 |
| `jax.jvp` with `adjoint=DirectAdjoint()`, the #80 case | finite |
| `jax.jvp` with the default adjoint | the same `TypeError` as #80 |

plumax can use this today, with no core at all, once it declares pipekit-jax:

```python
# Shapes: G = Nz × Ny × Nx with the ghosts
import diffrax as dfx
import jax.numpy as jnp
from pipekit_cycle import ForwardModel
from pipekit_jax import DiffraxForwardModel
from plumax.les_fvm import EulerianDispersionRHS, make_grid, uniform_wind_field
from plumax.les_fvm.boundary import build_default_concentration_bc
from plumax.les_fvm.diffusion import make_eddy_diffusivity
from plumax.les_fvm.source import make_gaussian_source

# today (plumax)
grid = make_grid((0.0, 2000.0, 40), (-400.0, 400.0, 16), (0.0, 200.0, 8))
horizontal, vertical = build_default_concentration_bc()
rhs = EulerianDispersionRHS(
    plume_grid=grid,
    wind_field=uniform_wind_field(grid, u=3.0, v=0.5),
    eddy_diffusivity=make_eddy_diffusivity((10.0, 2.0)),
    source=make_gaussian_source(grid, 1.0, (500.0, 0.0, 50.0)),
    horizontal_bc=horizontal,
    vertical_bc=vertical,
)
# today (pipekit_jax): in place of plumax's EulerianForwardModel
forward = DiffraxForwardModel(
    vector_field=rhs, solver=dfx.Heun(), dt0=1.0, dt=5.0, max_steps=100
)
assert isinstance(forward, ForwardModel)
# c(t + Δt) = step(c(t))
next_c = forward.step(jnp.zeros(grid.shape), 5.0)  # → G
```

The core needs only a thin method, so that boundary conditions and constrained parameters are handled as `Model.build_terms` handles them:

```python
# proposed (SC-1)
class Model(eqx.Module):
    def as_forward_model(self, *, dt: float, dt0: float, **settings):
        """This model as a DiffraxForwardModel, applying its boundary conditions."""
```

In `sesmax-ocean` and `sesmax_core.testing`, this is how a model enters a cycle or a filter:

```python
# Shapes: D_x = 40 variables

# proposed (SC-1)
forward = model.as_forward_model(dt=0.05, dt0=0.05, solver=dfx.Tsit5())
# today (filterax's form, from pipekit_jax)
dynamics = forward.as_dynamics()
# proposed (SC-1): the same model, with forward-mode gradients
tangent_ready = model.as_forward_model(dt=0.05, dt0=0.05, adjoint=dfx.DirectAdjoint())
```

What it does not replace is worth stating.
It holds one vector field, so it cannot express the `MultiTerm` an IMEX split needs, and `Model.integrate(**kw)` with `imex_solver` stays.
Its `step` integrates over $[0, \Delta t]$, so a model with time-dependent forcing needs `as_dynamics()` or SC-5.
The runner's step budget stays in the runner (#28).

### SC-2: `HatInTime`

`BasisForcing` already separates a control from its space and time shapes, $\text{field}(s, t) = \boldsymbol{\Phi} (\mathbf{w} \odot \mathbf{b}(t))$, where the coefficients $\mathbf{w}$ are the control.
Its temporal bases are constant, Fourier and Gaussian windows, whereas plumax's control is a value at each time, linearly interpolated, as in `_interpolated_emission`.
A hat basis gives that shape: each $w_k$ is then the value at knot $t_k$.

$$b_k(t) = \max\left(0, 1 - \frac{\vert t - t_k \vert}{\Delta t_k}\right)$$ (shared-hat)

```python
# proposed (SC-2)
class HatInTime(TemporalBasis):
    """Piecewise-linear weights on knots, so coeffs are the values at the knots."""

    knots: Float[Array, " m"]

    def weights(self, t: float) -> Float[Array, " m"]:
        ...
```

In plumax, an emission time series at a known location becomes a forcing whose dictionary has one column per knot:

```python
# Shapes: Nz, Ny, Nx interior; m = knots

# Today (plumax, les_fvm/fourdvar.py):
source = make_gaussian_source(
    grid, lambda t: jnp.interp(t, save_times, q_series), location
)
# After:
# today (sesmax_core)
# Φ = the source's density, repeated once per knot
spatial = sx.SpatialBasis.from_array(dictionary, std=prior_std)  # Φ → (N, m)
# proposed (SC-2)
# ρ(s) Σₖ wₖ bₖ(t)
emission = sx.BasisForcing(
    coeffs=q_series, spatial=spatial, temporal=HatInTime(save_times), grid_shape=shape
)
# today (sesmax_core)
term = sx.ForcingTerm(emission, place=sx.add_to("c"))
```

In `sesmax-ocean`, a wind amplitude that varies through a window becomes a control without changing the model's physics:

```python
# Shapes: Ny × Nx = grid; m = knots

# Today (somax, qg/barotropic.py): one amplitude for the whole run
dq_dt = dq_dt + tau0 * self.wind_forcing
# After:
# proposed (SC-2)
# τ(s, t) = F_wind(s) Σₖ wₖ bₖ(t)
wind = sx.BasisForcing(
    coeffs=tau0_series,
    spatial=wind_pattern,
    temporal=HatInTime(knots),
    grid_shape=shape,
)
# proposed (C2): the term tree assembled in sesmax-ocean
rhs = advection + drag + viscosity + sx.ForcingTerm(wind, place=sx.add_to("q"))
```

`test_basis.py` already checks that a one-column `BasisForcing` reproduces the hand-written wind forcing to the bit, so only the temporal basis is new.

### SC-3: withdrawn

A first draft proposed `TracerAdvection` and `TracerDiffusion` in the core.
They would have been a third layer over finitevolX's operators, after plumax's `advection_tendency` and the wrappers in sesmax, and only plumax would use them now, since the ocean's dynamics keep their own operators, such as the Arakawa Jacobian.

So the term classes stay in plumax, written as in the proof above: a `Term` of about ten lines around finitevolX. The core already supplies what they need, namely `Term`, `TermModel` and the `+` algebra.
The stencil gaps go upstream instead (SC-6, SC-7).
A core class can follow if a second consumer appears, such as a passive tracer in `sesmax-ocean`.

The proof's one finding still stands, so the `AGENTS.md` contract gains a line: a term may fill its own ghosts, because advection and diffusion need different Dirichlet ghost values.

### SC-4: `stable_step`

An explicit step takes advection and diffusion together, so the bound comes from their summed rate.
plumax's `stable_step_bound` has this right for 3D, and the core's `Scales.dt_from_cfl` answers a different question, from characteristic scales before a run has fields.

$$\Delta t_\text{max} = \frac{\mathrm{CFL}}{\sum_i \vert u_i \vert / \Delta x_i + 2 \sum_i K_i / \Delta x_i^2}$$ (shared-stable-step)

```python
# proposed (SC-4)
def stable_step(
    spacing: Sequence[float],
    *,
    speed: Sequence[float] = (),
    diffusivity: Sequence[float] = (),
    cfl: float = 1.0,
) -> float:
    """Largest stable explicit step; inf when nothing moves or diffuses."""
```

```python
# Shapes: three axes in plumax, two in sesmax-ocean

# Today (plumax, les_fvm/simulate.py):
bound = stable_step_bound(
    dx=dx, dy=dy, dz=dz, max_u=u, max_v=v, max_w=w, k_h=k_h, k_z=k_z
)
# After:
# proposed (SC-4)
# Δt_max = CFL / (Σᵢ |uᵢ| / Δxᵢ + 2 Σᵢ Kᵢ / Δxᵢ²)
bound = sx.solvers.stable_step(  # → ()
    (dx, dy, dz), speed=(u, v, w), diffusivity=(k_h, k_h, k_z)
)

# proposed (SC-4, C2): the ocean runner's preflight, from a state
bound = sx.solvers.stable_step((dx, dy), speed=(u_max, v_max), diffusivity=(nu, nu))
```

The bound is a safety net rather than a stability certificate for high-order reconstructions, as plumax's docstring says, and the core's should say so too.

### SC-5: `WindowOperator`

#73 asks how 4D-Var can drive a model with time-dependent forcing, because `VardaxForwardModel` holds `t0` fixed on every substep.
plumax has already answered this differently: it treats the whole window as one map from control to observations and hands that to `vardax.ThreeDVar`.
There is then no clock problem, because the model integrates the window itself.
That option is not in #73, and this page recommends it there.

$$\mathcal{G}(\boldsymbol{\theta}) = \mathrm{vec}\left[\mathbf{H} \mathbf{x}(t_k; \boldsymbol{\theta})\right]_{k=1}^{N_T}$$ (shared-window)

$$\mathcal{J}(\boldsymbol{\theta}) = \frac{1}{2} \Vert \boldsymbol{\theta} - \boldsymbol{\mu}_\theta \Vert^2_{\boldsymbol{\Sigma}_\theta^{-1}} + \frac{1}{2} \Vert \mathbf{y} - \mathcal{G}(\boldsymbol{\theta}) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}}$$ (shared-window-cost)

```python
# proposed (SC-5)
class WindowOperator(eqx.Module):
    """𝒢(θ): a control through a model run to the stacked observations."""

    build_model: Callable[[Array], Model]  # θ ↦ the model, closing over the rest
    initial_state: PyTree
    obs_op: Callable[[PyTree], Array]
    save_times: Float[Array, " time"]
    dt: float = eqx.field(static=True)

    def __call__(self, control: Float[Array, " param"]) -> Float[Array, " obs"]:
        ...
```

In plumax, this is the generic part of `FourDVarProblem.source_receptor_operator`, and `fourdvar.py` keeps its priors, whitening and posterior:

```python
# Shapes: m = knots; N_T = save times; D_y = N_T × observations per time

# Today (plumax, les_fvm/fourdvar.py):
forward = build_forward(...)
operator = vardax.CallableObs(fn=whitened_prediction, jac="bwd", materialise=True)
# After:
# proposed (SC-5)
# 𝒢(q) = vec[H c(tₖ; q)]
window = sx.da.WindowOperator(  # (m,) → (D_y,)
    build_model=model_for_emission,
    initial_state=state0,
    obs_op=column_operator,
    save_times=save_times,
    dt=dt,
)
# today (vardax)
# q* = argmin 𝒥(q)
analysis = vardax.ThreeDVar(
    obs_op=window.as_obs_op(),
    prior_mean=prior_mean,  # (m,)
    prior_cov_op=prior_covariance,  # (m, m) PSD operator
    obs_cov_op=obs_covariance,  # (D_y, D_y) PSD operator
)
```

In `sesmax-ocean`, the same operator estimates the coefficients of a wind forcing from sea-surface height, with no ocean-specific code in the core:

```python
# Shapes: m = basis coefficients; N_T = overpasses; D_y = N_T × observed cells


def model_for_wind(coefficients):
    # today (equinox)
    return eqx.tree_at(lambda m: m.forcing.coeffs, base_model, coefficients)


# proposed (SC-5, C2)
# 𝒢(w) = vec[H η(tₖ; w)]
window = sx.da.WindowOperator(  # (m,) → (D_y,)
    build_model=model_for_wind,
    initial_state=state0,
    obs_op=ssh_at_tracks,
    save_times=overpasses,
    dt=dt,
)
```

### SC-6 and SC-7: upstream in finitevolX

These are the two stencil gaps that make everyone write around finitevolX. Stencils belong there, so both are upstream asks and not core API.

**SC-6, `wall="periodic"`.**

Filling the ghosts of the interpolated face velocities and passing `wall="open"` makes a periodic domain conservative.
That logic now exists in `plumax.les_fvm.advection.advection_tendency`, in sesmax's `_burgers_flux.py` and `_advective_form.py`, and in `IncompressibleNS2D._wrap`, and forgetting it caused #67.

```python
# Shapes: G = Ny × Nx with the ghosts

# Today (sesmax_core, _advective_form.py):
u_on_U = enforce_periodic(interp.T_to_U(u))
v_on_V = enforce_periodic(interp.T_to_V(v))
tendency = advection(u, u_on_U, v_on_V, method=method, wall="open")
# After:
# proposed (finitevolX)
tendency = advection(u, u_on_U, v_on_V, method=method, wall="periodic")
```

**SC-7, vertical operators.** `Advection3D` and `Diffusion3D` act per level, so plumax writes its own vertical tendencies with `vertical_advection_tendency` and `vertical_diffusion_tendency`.

```python
# Shapes: G = Nz × Ny × Nx with the ghosts

# Today (plumax, les_fvm/advection.py):
vertical = vertical_advection_tendency(concentration, w, plume_grid.dz)
# After:
# proposed (finitevolX)
# ∂c/∂t ⊇ −∂(w c)/∂z
vertical = advection_3d.vertical(concentration, w, method="upwind1")  # → G
```

## Related work this changes

The run loop (#28) and the levels (#29) are already designed for one consumer, and plumax's four `simulate_*` functions, each ending in an xarray dataset, are a second.
Both issues should list plumax's needs before they are decided, in particular a derived diagnostic such as column concentration.

## What stays out

| Item | Stays in | Why |
| --- | --- | --- |
| All of radtrax | radtrax | It is a static forward model with no time stepping, and its base install has none of diffrax, finitevolX or pipekit, so the core would add weight and no function |
| `var3d` control transforms and background builders | radtrax, then vardax or gaussx | They belong with the variational methods, which is the direction #27 already takes |
| Observation operators (`ColumnObservationOperator`, `RadianceObservationModel`, `SubsampleObs`) | vardax | #27 |
| `met`, `surface_layer`, `pbl`, the eddy-diffusivity closures | plumax, or a later `sesmax-atmos` | They are atmosphere physics |
| `gauss_plume`, `gauss_puff`, `population` | plumax | They are analytic plume models, not grid models, and they would gain nothing from `Model` |
| `lagrangian` particles | plumax for now | A stochastic particle `lax.scan` has no counterpart in the core, and one consumer does not justify an SDE layer yet |
| `check_munk_width` and the other resolution checks | `sesmax-ocean` | They read ocean boundary-layer scales |

radtrax still reaches plumax's models through plumax's own coupling, so it needs no core API at all, and no usage is sketched for it here.

## What has to happen first

The core and plumax cannot share an environment today.
With plumax's `main` and the core on one path, importing `sesmax_core` failed for want of `paramax`, importing `sesmax_core.train` failed for want of `pipekit_jax`, and in the other direction, sesmax's pinned geonnax is too old for radtrax's import of `EARTH_MEAN_RADIUS_M`.

| Package | sesmax's environment | plumax's environment |
| --- | --- | --- |
| finitevolX | 0.0.48 | 0.0.45 |
| geonnax | 0.0.4 | 0.0.10 |
| vardax | 0.1.8 | 0.2.1 |
| spectraldiffx | 0.1.1 | 0.0.13 |
| jax | 0.12.0.1 | 0.11.0 |
| paramax | 0.0.5 | not installed |
| filterax | 0.0.3 | not installed |

The core's direct git references are the larger problem, because PyPI rejects them, as the [workspace design](workspace_and_core.md) already records, and plumax uses version ranges.
Candidate SC-5 also needs the vardax bump, since `CallableObs` is not in sesmax's pin.

## Phases

| Phase | Scope | Gate |
| --- | --- | --- |
| S0 | Bump the pins to what plumax and radtrax use; replace git references with PyPI ranges where the stack has released | `uv lock` resolves with radtrax importable; the core's suite passes |
| S1 | SC-1 and SC-4, the `AGENTS.md` note on per-term ghosts, and the core's own small repeats | Existing tests pass; `as_forward_model` matches `Model.step` exactly |
| S2 | SC-2 | A forcing built on `HatInTime` matches `jnp.interp` of its coefficients |
| S3 | SC-5, and the #73 decision | A twin experiment recovering a time-varying source on a core model |
| S4 | plumax drops `EulerianForwardModel`, moves `les_fvm` onto `TermModel` and adopts SC-4 and SC-5, as plumax PRs | plumax's existing tests pass, and the proof's comparison holds on its real configurations |
| Upstream | SC-6 and SC-7 on finitevolX | The three sesmax copies and plumax's copy deleted |

S1 to S3 touch only sesmax, so they can go ahead before S4.
S0 is what makes S4 possible, and S4 does not block the ocean move.
Replacing `EulerianForwardModel` needs only a pipekit-jax dependency, so it could go ahead at any time.

## Risks

- **The base install is heavy for plumax.** The core's top-level import needs `paramax`, and `sesmax_core.train` needs `pipekit_jax`, so plumax would gain both, plus spectraldiffx and einops. Fallback: move `gfd` and `testing` behind an extra, which the workspace design already lists.
- **A second consumer can bend the API.** Fallback: a candidate joins the core only if both consumers use it in this page's form; otherwise it stays in plumax. SC-3 already went this way.
- **The audit reads names and bodies, not behaviour.** A repeat under an unrelated name can escape it, and the rows marked as not compared are candidates, not findings.
- **The proof ran one configuration.** It used a uniform wind and plumax's default boundaries. Gate S4 runs plumax's own.

## Open questions

| ID | Question | Leaning |
| --- | --- | --- |
| SC-Q1 | Does `ModalTransform` delegate to finitevolX, given their different normalisation? | Decide in C2, when the multilayer models it serves arrive |
| SC-Q2 | Are SC-1 and SC-4 worth doing if plumax never switches? | Yes: the DA adapters and the ocean runner use them alone |

## Relationships

- Depends on: [the workspace and core design](workspace_and_core.md)
- Changes: #28, #29, #73, and closes #25 and #26 as not planned
- Tracked in: #8

| Phase | Issue |
| --- | --- |
| S0 | [#84](https://github.com/jejjohnson/sesmax/issues/84) pins and PyPI ranges |
| S1 | [#85](https://github.com/jejjohnson/sesmax/issues/85) SC-1, [#87](https://github.com/jejjohnson/sesmax/issues/87) SC-4, [#89](https://github.com/jejjohnson/sesmax/issues/89) the core's own repeats |
| S2 | [#86](https://github.com/jejjohnson/sesmax/issues/86) SC-2 |
| S3 | [#88](https://github.com/jejjohnson/sesmax/issues/88) SC-5 |
| S4 | [plumax#231](https://github.com/jejjohnson/plumax/issues/231), [plumax#232](https://github.com/jejjohnson/plumax/issues/232) |
| Upstream | [finitevolX#403](https://github.com/jejjohnson/finitevolX/issues/403), [finitevolX#404](https://github.com/jejjohnson/finitevolX/issues/404), then [#90](https://github.com/jejjohnson/sesmax/issues/90); [vardax#217](https://github.com/jejjohnson/vardax/issues/217) |
| Layering | [geostack#120](https://github.com/jejjohnson/geostack/issues/120) |
