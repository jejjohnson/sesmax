---
name: add-forcing-basis
description: Add a forcing or a reduced-order basis to sesmax — a ForcingProtocol subclass, a spatial_from_* constructor with its per-mode prior, or a TemporalBasis, with the public exports, the API page and the tests they join. Use when asked to add a forcing field, a spatial dictionary, a temporal gate or a control basis for assimilation.
---

# Add a forcing or a basis

Read "Forcing" under "The contracts" in `AGENTS.md` first. A forcing returns a field; `ForcingTerm` turns that field into a tendency, so a new forcing never touches the term algebra.

## 1. Which one

| You need | Write | It goes in |
|---|---|---|
| A field as a function of time | a `ForcingProtocol` subclass | `packages/sesmax-core/src/sesmax_core/_src/forcing.py` |
| A new spatial dictionary | a `spatial_from_<name>(domain, ...)` returning a `SpatialBasis` | `packages/sesmax-core/src/sesmax_core/_src/forcing_bank.py` |
| A new time gate | a `TemporalBasis` subclass implementing `weights(t)` | `packages/sesmax-core/src/sesmax_core/_src/forcing_bank.py` |
| Atoms with several components | a `spatial_from_<name>(domain, ...)` returning a `VectorSpatialBasis`, which feeds `VectorBasisForcing` and `add_vector_to` | `packages/sesmax-core/src/sesmax_core/_src/forcing_bank.py` |

The dictionary's maths belongs upstream: a constructor evaluates a basis from `geonnax.basis` on `domain.coords` and does not re-derive it (#25 records the planned move to vardax's bases).

## 2. Write it

A spatial constructor builds `Phi` with shape `(Ngrid, m)` and gives each mode a prior standard deviation: from a spectral density for a spectral basis (`matern_spectral_density`), prescribed for a frame.

The pieces then compose, and `coeffs` is the only control:

```python
import equinox as eqx
import jax.numpy as jnp

import sesmax_core as sx


class Tracer(sx.State):
    q: jnp.ndarray


forcing = sx.BasisForcing(
    coeffs=jnp.zeros(2),
    spatial=sx.SpatialBasis.from_array(jnp.eye(4)[:, :2]),
    temporal=sx.ConstantInTime(m=2),
    grid_shape=(2, 2),
)
term = sx.ForcingTerm(forcing, place=sx.add_to("q"))
tendency = term(0.0, Tracer(q=jnp.zeros((2, 2))))
control, fixed = eqx.partition(forcing, sx.control_filter(forcing))
```

## 3. Export and document it

Add the name to `__all__` in `sesmax_core/forcing.py` and in the top-level `sesmax_core/__init__.py`. `docs/api/forcing.md` renders it from there; update its row in `docs/api/index.md`, then run `make capabilities`.

## 4. Test it

Tests join `packages/sesmax-core/tests/core/test_forcing.py` for a forcing and `packages/sesmax-core/tests/core/test_basis.py` for a basis. Check the field's shape, that `eqx.filter_grad` reaches the coefficients, and, for a spectral basis, the prior against its density. The bar is `test_basis.py`'s check that a one-column `BasisForcing` reproduces a hand-written wind forcing.

```bash
uv run pytest --no-cov packages/sesmax-core/tests/core/test_forcing.py packages/sesmax-core/tests/core/test_basis.py -v
```

Finish with `pre-pr-check`.
