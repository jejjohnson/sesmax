---
name: add-term
description: Add a physics term or a term model to sesmax — a Term subclass returning a tendency with the state's structure, its explicit or implicit tag, where the file goes, the public exports, the API page, and the tests it joins. Use when asked to add a tendency, a physics contribution, an IMEX split or a TermModel.
---

# Add a term

Read "Terms" and "Models" under "The contracts" in `AGENTS.md` first, and search `docs/api/capabilities.md`: finitevolX already owns most stencils, so a term usually calls one rather than writing it.

## 1. Where it goes

| The term is | It goes in |
|---|---|
| Part of one model | beside that model, as in `packages/sesmax-core/src/sesmax_core/_src/testing/swm/linear_swm_terms.py` |
| Reusable by any model | `packages/sesmax-core/src/sesmax_core/_src/terms.py` |
| A pipekit operator or a learned replacement | no new class: wrap it with `OperatorTerm(op, inputs=..., outputs=...)` |
| Ocean-specific | `sesmax-ocean`, once it exists (phase C2) |

## 2. Write it

```python
import equinox as eqx
import jax
import jax.numpy as jnp

import sesmax_core as sx


class Drag(sx.Term):
    """Linear drag on every field: ∂x/∂t = −r x.

    Args:
        rate: Drag coefficient r, a leaf so that `jax.grad` reaches it.
    """

    rate: jax.Array = eqx.field(converter=jnp.asarray)

    def __call__(self, t, state, args=None):
        # ∂x/∂t = −r x
        return jax.tree.map(lambda leaf: -self.rate * leaf, state)
```

- `__call__(t, state, args=None)` returns a pytree with the state's structure, zero in any field the term does not touch.
- Coefficients are leaves; grids, shapes and flags are `eqx.field(static=True)`.
- Tag a stiff term where the model assembles it, `sx.implicit(Drag(1.0))`, and pair it with `imex_solver` and `imex_stepsize_controller` (`from sesmax_core.solvers import imex_solver, imex_stepsize_controller`; both live in `integrix.march`).

## 3. Assemble it

A `TermModel` holds the tree, and `build_diffrax_terms` routes each summand to its stage.

```python
class DampedModel(sx.TermModel):
    pass


model = DampedModel(terms=Drag(1.0) + sx.implicit(Drag(0.5)))
```

Override `apply_boundary_conditions` when the model has boundaries; `TermModel`'s default passes the state through.

## 4. Export and document it

A term that stays private to a model needs no export. A reusable one joins `__all__` in `sesmax_core/terms.py` and in the top-level `sesmax_core/__init__.py`, and `docs/api/terms.md` renders it from there. A test problem's term joins `sesmax_core/testing.py` instead, and `docs/api/testing.md`. Then run `make capabilities`.

## 5. Test it

Check the tendency against a closed form, then that `jit` and `eqx.filter_grad` go through it. Tests join `packages/sesmax-core/tests/core/test_terms.py` for a reusable term, and the model's file under `packages/sesmax-core/tests/testing/` otherwise.

```bash
uv run pytest --no-cov packages/sesmax-core/tests/core/test_terms.py -v
```

Finish with `pre-pr-check`.
