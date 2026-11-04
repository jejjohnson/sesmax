---
name: add-da-adapter
description: Add a data-assimilation adapter to sesmax — a flat-vector bridge from a model's pytree step to a DA library, with its template, transform, optional extra and ImportError, the public exports, the API page and a twin-experiment test. Use when asked to connect sesmax models to a new filter, variational method or assimilation library, or to change the filterax and vardax bridges.
---

# Add a DA adapter

Read "Data-assimilation adapters" under "The contracts" in `AGENTS.md` first, and read the two existing bridges, `_src/da/filterax_bridge.py` and `_src/da/vardax_bridge.py`, which a new one should follow.

## 1. Where it goes

| What | Where |
|---|---|
| The adapter | `packages/sesmax-core/src/sesmax_core/_src/da/<library>_bridge.py` |
| Its export | `_src/da/__init__.py` and `sesmax_core/da.py` |
| The dependency | the `[da]` extra in `packages/sesmax-core/pyproject.toml`, then `uv lock` |
| An observation operator | `_src/da/obs.py` |

## 2. Write it

The adapter is an `eqx.Module` that matches the library's interface and does four things per call: unravel with the stored `template`, apply `transform.inverse` if there is one, call `model.step`, then apply `transform.forward` and ravel.

```python
import equinox as eqx
from jax.flatten_util import ravel_pytree


class ExampleDynamics(eqx.Module):
    """Advance a flat state with a model's pytree step."""

    model: object
    template: object
    transform: object = None

    def __call__(self, flat_state, t0, t1):
        _, unravel = ravel_pytree(self.template)
        state = unravel(flat_state)
        if self.transform is not None:
            state = self.transform.inverse(state)
        state_next = self.model.step(state, t1 - t0, t0=t0)
        if self.transform is not None:
            state_next = self.transform.forward(state_next)
        flat_next, _ = ravel_pytree(state_next)
        return flat_next
```

- Store the `template`, not the `unravel` closure, so the module stays a clean pytree under `jit` and `filter_vmap`.
- Drive the model only through `step`, never `integrate`.
- Use `state_to_vector` and `make_ensemble` for the shared layout; do not write a second flattening.
- Say in the docstring if the library's interface carries no per-substep clock, as `VardaxForwardModel` does, because the adapter is then exact only for autonomous models.

## 3. Keep the base install clean

`sesmax_core/da.py` wraps its imports and raises an `ImportError` naming `[da]`, and `sesmax_core/__init__.py` does not import it. A dependency that fits neither library's role gets its own extra and its own public module, with a page in `docs/api/`, a `nav` entry in `mkdocs.yml`, a row in `docs/api/index.md` and an entry in `SUBMODULES` in `packages/sesmax-core/tests/test_public_api.py`. Run `make capabilities`.

## 4. Test it

Add a twin experiment at `packages/sesmax-core/tests/da/test_<library>.py`, opening with `pytest.importorskip("<library>")`. Use a truth run and sparse noisy observations, and assert that the analysis beats the background, bounded by the observation noise rather than a tolerance tuned on one draw. `test_filterax.py` and `test_vardax.py` are the models. Extend `tests/da/test_missing_extra.py` if you added a module.

```bash
uv run pytest --no-cov packages/sesmax-core/tests/da -v
```

Finish with `pre-pr-check`.
