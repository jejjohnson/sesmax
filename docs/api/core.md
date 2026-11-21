# Types and models

The state and parameter types every model is built from, the parameter constraints that keep a learned parameter inside its physical range, and the model base classes. All are importable from the top level, `sesmax_core`.

## Types

::: sesmax_core.State

::: sesmax_core.Params

::: sesmax_core.PhysConsts

::: sesmax_core.Diagnostics

## Parameter constraints

::: sesmax_core.positive

::: sesmax_core.interval

::: sesmax_core.frozen

::: sesmax_core.as_parameter

::: sesmax_core.trainable_mask

## Models

::: sesmax_core.Model

::: sesmax_core.TermModel

## The PDE driver

`solve_pde` is the function-level form of `Model.integrate`, for a right-hand side function rather than a `Model` subclass. It runs `diffrax.diffeqsolve` on dy/dt = P(f(t, B(y))), where the state correction B (`state_bc`, which fills ghost cells from the interior) acts before each evaluation of f and the tendency correction P (`tendency_bc`, which zeroes the tendency on walls or dry cells) acts after it, at every stage. Any other keyword goes to `diffeqsolve`. A fixed-step run with concrete times gets `max_steps = ceil((t1 - t0) / dt0) + 16` by default, so a long run is not cut off by diffrax's 4,096 steps. It moved from gridax-jax's `solve_ocean_pde`, whose `bc_fn` is `tendency_bc` here.

```python
from integrix.march import RK3SSP
from sesmax_core import solve_pde

def rhs(t, y, args):
    return compute_tendency(y)

def zero_wall_tendency(dydt):
    return dydt.at[0, :].set(0).at[-1, :].set(0)

sol = solve_pde(rhs, RK3SSP(), y0, 0.0, 1000.0, 1.0, tendency_bc=zero_wall_tendency)
```

::: sesmax_core.solve_pde
