"""Public surface for sesmax-core's layered-fluid pieces.

:class:`StratificationProfile` holds a layered fluid's thicknesses and
reduced gravities; :class:`ModalTransform` projects between layers and
vertical modes, with the eigendecomposition delegated to finitevolX. An
ocean and an atmosphere share them, so they live in the core.

All are also importable from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.gfd.modal import ModalTransform, StratificationProfile


__all__ = [
    "ModalTransform",
    "StratificationProfile",
]
