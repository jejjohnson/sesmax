# Elliptic inversions

A layered-fluid model carries a prognostic field, such as the vorticity or the potential vorticity, and has to recover a diagnostic one, such as the streamfunction or the pressure, at every step. That recovery is an elliptic solve. `sesmax_core.gfd` has the three a C-grid model needs:

| Function | Equation | Unknown | $\lambda$ |
|---|---|---|---|
| [`streamfunction_from_vorticity`](xref:api#sesmax_core.streamfunction_from_vorticity) | $(\nabla^2 - \lambda)\,\psi = \zeta$ | $\psi$ at X points | $0$ (Poisson) or $\ge 0$ |
| [`pressure_from_divergence`](xref:api#sesmax_core.pressure_from_divergence) | $\nabla^2 p = \nabla \cdot \mathbf{u}$ | $p$ at T points | $0$ |
| [`pv_inversion`](xref:api#sesmax_core.pv_inversion) | $(\nabla^2 - \lambda_k)\,\psi_k = q_k$ | $\psi_k$ at X points, per layer or mode $k$ | $1/R_{d,k}^2$ |

Here $\nabla^2$ is the five-point Laplacian on steps $\Delta x$ and $\Delta y$, and $R_{d,k}$ is the deformation radius of layer or vertical mode $k$. For $\lambda \ge 0$ the operator $(\nabla^2 - \lambda)$ is negative semi-definite, the sign convention of the whole stack. Each function composes pieces that live elsewhere: gridax builds what comes from the grid and the mask (the walls, the masked operators, the coast ring, the capacitance and multigrid set-up), and gaussx and spectraldiffx run the solves. They moved here from gridax-jax's `solvers/elliptic.py`, and they return its arrays bit for bit.

## Quick start

Every function takes the full `[Ny, Nx]` array, ghost ring included, and returns the full array. It knows which nodes are unknowns, so don't slice the interior by hand.

```python
import jax
import jax.numpy as jnp
import numpy as np
from gridax import CartesianCGrid2D, Mask2D
import gridax.solvers
import sesmax_core.gfd as gfd

jax.config.update("jax_enable_x64", True)

# A 64 x 64-cell basin, 100 km on a side: arrays are [Ny, Nx] = [66, 66]
grid = CartesianCGrid2D.from_interior(64, 64, 1e5, 1e5)
dx, dy, Ny, Nx = grid.spacing("x"), grid.spacing("y"), grid.Ny, grid.Nx

# vorticity at the X points, where psi lives (X node i sits at x = i dx)
x = jnp.arange(Nx)[None, :] * dx
y = jnp.arange(Ny)[:, None] * dy
zeta = jnp.sin(jnp.pi * x / grid.Lx) * jnp.sin(jnp.pi * y / grid.Ly)  # [Ny, Nx]

# (∇² − λ) ψ = ζ with ψ = 0 on the C-grid walls
psi = gfd.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")  # [Ny, Nx]
```

The masked examples below use a circular basin with an island, and a deformation radius of 20 km:

```python
# ocean at T points (True = wet); Mask2D derives the psi (X) and p (T) masks
jj, ii = np.mgrid[:Ny, :Nx]
ocean = np.zeros((Ny, Nx), dtype=bool)
ocean[1:-1, 1:-1] = True
ocean &= (ii - Nx / 2) ** 2 + (jj - Ny / 2) ** 2 < (0.45 * Nx) ** 2
ocean &= (ii - 0.62 * Nx) ** 2 + (jj - 0.55 * Ny) ** 2 > (0.1 * Nx) ** 2
mask = Mask2D.from_mask(ocean)  # psi uses mask.wet(at="X", kind="strict"), p uses mask.wet()

lambda_ = 1 / 2e4**2  # λ = 1 / R_d²
```

## The walls

The spectral transforms treat every node they are given as an unknown, so where the wall ends up depends on which nodes they see. Each function slices the unknowns of its staggering out of the array, solves and embeds the result in zeros, so the wall is the C-grid wall. On a `from_interior` grid the T cells are `[1:-1]` and the walls are the faces at $x = 0$ and $x = L$. X node $i$ sits at $x = i\,\Delta x$, so the walls are the X nodes `0` and `N-2`, and `N-1` is outside the domain.

| Unknown | `bc` | Unknowns | Transform | Wall |
|---|---|---|---|---|
| $\psi$ (X) | `"dst"` | `[1:-2, 1:-2]` | DST-I | $\psi = 0$ on the wall nodes `0`, `N-2` |
| $\psi$ (X) | `"dct"` | `[0:-1, 0:-1]` | DCT-I | $\partial\psi/\partial n = 0$ about the wall nodes |
| $\psi$ (X) | `"fft"` | `[1:-1, 1:-1]` | FFT | periodic, period `N-2` |
| $p$ (T) | `"dct"` | `[1:-1, 1:-1]` | DCT-II | $\partial p/\partial n = 0$ on the faces |
| $p$ (T) | `"dst"` | `[1:-1, 1:-1]` | DST-II | $p = 0$ on the faces |
| $p$ (T) | `"fft"` | `[1:-1, 1:-1]` | FFT | periodic, period `N-2` |

The masked methods put the wall in the same place: for $\psi$ it is $\psi = 0$ on the dry X nodes next to the wet ones (`Mask2D.xy_corner_strict`), and for $p$ it is on every wet–dry face of `Mask2D.h`. Every combination converges at second order. A `bc` a method cannot honour raises rather than being ignored.

## Choosing a method

| `method` | Domain | What it needs | Cost per solve |
|---|---|---|---|
| `"spectral"` (default) | the rectangle | nothing | one transform pair |
| `"cg"` | any mask | `mask=` | CG steps, each one masked stencil and one spectral preconditioner |
| `"capacitance"` | any mask, $\psi$ only | `capacitance_solver=` | two transform pairs and a small dense solve |
| `"multigrid"` | any mask | `mask=` and `multigrid_solver=` | a few CG steps, each one V-cycle |

The capacitance and multigrid set-ups depend only on the geometry and $\lambda$, so build them once and reuse them for every right-hand side.

::::{tab-set}
:::{tab-item} CG

```python
# psi = 0 on the coast nodes of mask.wet(at="X", kind="strict")
psi = gfd.streamfunction_from_vorticity(
    zeta, dx, dy, lambda_=lambda_, method="cg", mask=mask
)
```
:::
:::{tab-item} Capacitance

```python
# once, for this mask and this λ: one spectral solve per coast node
solver = gridax.solvers.build_capacitance_solver(mask, dx, dy, lambda_=lambda_)
psi = gfd.streamfunction_from_vorticity(
    zeta, dx, dy, lambda_=lambda_, method="capacitance", capacitance_solver=solver
)
```
:::
:::{tab-item} Multigrid

```python
# once, for this mask and this λ: the level hierarchy
hierarchy = gridax.solvers.build_multigrid_solver(mask, dx, dy, lambda_=lambda_)
psi = gfd.streamfunction_from_vorticity(
    zeta, dx, dy, lambda_=lambda_, method="multigrid", mask=mask,
    multigrid_solver=hierarchy,
)
```
:::
::::

`method="cg"` and `method="multigrid"` stop when the residual satisfies $\lVert \mathbf{b} - \mathbf{A}\mathbf{x} \rVert_\infty \le 10^{-6} + 10^{-6} \lVert \mathbf{b} \rVert_\infty$ in float64, and below float64 at $\sqrt{\varepsilon}$ relative to the right-hand side. They raise if CG has not converged in 500 steps, also under `jit`. The multigrid hierarchy must be built for the solve's mask, $\lambda$ and wall: Dirichlet (the default) for $\psi$, and `bc="neumann"` on `mask.wet()` for the pressure with `bc="dct"`.

## The pressure

$p$ is fixed only up to a constant by the Neumann problem, so `bc="dct"` returns the solution with zero mean, on each connected wet region for a masked domain:

```python
div_u = zeta  # any T-point field; the Neumann solve drops its mean

p = gfd.pressure_from_divergence(div_u, dx, dy)  # bc="dct"
p = gfd.pressure_from_divergence(div_u, dx, dy, method="cg", mask=mask)
```

`bc="dst"` holds $p = 0$ on the faces, half a cell out. The capacitance method and known values pin values at cell centres, not on the faces where the pressure wall is, so `pressure_from_divergence` does not offer them.

## Layers and vertical modes

`pv_inversion` takes leading batch axes. With an array `lambda_` of shape `(nl,)`, axis `-3` of the potential vorticity is the layer axis, and each layer is solved with its own $\lambda_k$. A layered model inverts in mode space, where the layers decouple:

```python
from sesmax_core.vertical import (
    build_coupling_matrix,
    decompose_vertical_modes,
    layer_to_mode,
    mode_to_layer,
)

H = jnp.array([500.0, 1500.0, 3000.0])  # layer thicknesses [m]
g_prime = jnp.array([9.81, 0.025, 0.0125])  # reduced gravities [m s⁻²]
radii, layer_to_modes, modes_to_layer = decompose_vertical_modes(
    build_coupling_matrix(H, g_prime), f0=1e-4
)

q_layers = jnp.stack([zeta, 0.5 * zeta, 0.1 * zeta])  # [nl, Ny, Nx]
q_modes = layer_to_mode(q_layers, layer_to_modes)
psi_modes = gfd.pv_inversion(q_modes, dx, dy, 1.0 / radii**2)  # (∇² − λ_k) ψ_k = q_k
psi_layers = mode_to_layer(psi_modes, modes_to_layer)
```

The capacitance and multigrid methods hold one $\lambda$, so they need a scalar `lambda_`.

## Known boundary values

By default $\psi = 0$ on the coast. When the boundary values are known but not zero, from a parent model, a tide gauge or a reanalysis, pass them as `known_values`. The known cells are the coast ring, the dry cells next to a wet cell around the outer walls and any islands, which is exactly where `method="cg"` puts its wall, plus any wet cells marked in `known_mask`. Every other wet cell is solved for, and the result equals `known_values` exactly on the known cells.

```python
# psi rising from 0 (south) to 1e3 (north): a throughflow of 1e3 m² s⁻¹
g = 1e3 * y / grid.Ly * jnp.ones((Ny, Nx))  # only the known cells are read
psi = gfd.streamfunction_from_vorticity(
    zeta, dx, dy, method="cg", mask=mask, known_values=g
)

# sparse interior observations, pinned like the coast
obs = jnp.zeros((Ny, Nx), dtype=bool).at[20, 30].set(True)
psi = gfd.streamfunction_from_vorticity(
    zeta, dx, dy, method="cg", mask=mask, known_values=g.at[20, 30].set(120.0),
    known_mask=obs,
)
```

This is the lifting trick. With $K$ the known cells and $E$ the unknowns, the five-point equations on $E$ split into

$$
\mathbf{A}_{EE}\,\boldsymbol{\psi}_E = \mathbf{f}_E - \mathbf{A}_{EK}\,\mathbf{g}_K
$$

so the solve runs on $E$ with a corrected right-hand side (gaussx's `lift_known_values`), and the known values are added back. The spectral method takes known values on the rectangle's wall nodes with `bc="dst"` and no mask. The capacitance method takes them with a solver built on the same wet mask, and no `known_mask`. `pv_inversion` broadcasts `known_values` against its input, and each layer's correction uses its own $\lambda_k$.

`bc` also accepts a `gridax.BC`. Its face types choose the transform (all `Dirichlet` gives `"dst"`, all zero `Neumann` gives `"dct"`, all `Periodic` gives `"fft"`), and non-zero `Dirichlet` values become known values on each face's wall nodes. That needs a basin whose outer coast lies on the four walls, a rectangle with islands allowed. For any other coastline, pass `known_values` on `gridax.solvers.coast_ring(mask)`.

## Examples

- [Inversion methods compared](../notebooks/inversion_methods.ipynb): the spectral, capacitance, CG and multigrid methods of $(\nabla^2 - \lambda)\psi = f$ run on a rectangle, a basin, a circle and a notch with a variable coefficient, comparing CG iteration counts, residuals and timings, with a short check of known boundary values.
- [Streamfunction inversion](../notebooks/streamfunction_inversion.ipynb): the same Poisson equation needs DST-I when $\psi$ lives at X points and DST-II at T points, shown on a Lamb–Oseen vortex with the velocity recovered from $\psi$ and a convergence test.
- [Pressure Poisson](../notebooks/pressure_poisson.ipynb): solving $\nabla^2 p = \nabla \cdot \mathbf{u}$ and projecting a velocity field to be divergence-free.
- [Helmholtz screening](../notebooks/helmholtz_screening.ipynb): how $\lambda = 1/L_d^2$ shortens the reach of an isolated PV anomaly, with Dirichlet and Neumann walls and the solve under `jit`, `vmap` and `grad`.
- [Known boundary values](../notebooks/inhomogeneous_bcs_tutorial.ipynb): the lifting trick derived from scratch, checked on a manufactured solution, with interior observations, multilayer PV, time-varying boundary data under `jit` and gradients with respect to the boundary data.
