"""Public surface for sesmax-core's data-assimilation adapters.

Wires sesmax forward models into the ``jejjohnson`` DA stack — a thin
adapter layer so any sesmax model can be driven by an external DA solver.

Ensemble filtering with `filterax`:

- :class:`FilteraxDynamics` adapts a sesmax model's pytree ``step`` to
  filterax's flat-vector ``AbstractDynamics``.
- :class:`SubsampleObs` is a sparse observation operator on the flat state.

Variational assimilation with `vardax`:

- :class:`VardaxForwardModel` adapts a sesmax model to vardax's flat-vector
  ``pipekit_cycle.ForwardModel`` (``dt`` + ``step``), consumed by
  ``StrongFourDVar`` / ``IncrementalFourDVar`` / ``VarDACycle``. Intended
  for autonomous models (the ``ForwardModel`` rollout carries no
  per-substep time). Structured background / observation covariances come
  from `gaussx` operators (e.g. ``LowRankUpdate``), which are
  ``lineax``-compatible.

Shared bridge:

- :func:`state_to_vector` / :func:`make_ensemble` convert sesmax pytree
  states to the flat vectors / ensembles both stacks expect.

Importing this module requires the optional ``da`` extra
(``pip install sesmax-core[da]``), which provides ``filterax`` and
``vardax`` (and, transitively, ``gaussx``); without it the import raises
an ``ImportError`` that says so. It is kept out of ``sesmax_core``'s
top-level ``__init__`` so plain ``import sesmax_core`` never needs the DA
stack.
"""

from __future__ import annotations


try:
    from sesmax_core._src.da import (
        FilteraxDynamics,
        SubsampleObs,
        VardaxForwardModel,
        make_ensemble,
        state_to_vector,
    )
except ImportError as exc:
    raise ImportError(
        "sesmax_core.da requires filterax and vardax, which are optional "
        "dependencies. Install them with `pip install sesmax-core[da]` (or "
        "`uv sync --extra da`)."
    ) from exc


__all__ = [
    "FilteraxDynamics",
    "SubsampleObs",
    "VardaxForwardModel",
    "make_ensemble",
    "state_to_vector",
]
