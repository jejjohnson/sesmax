"""Public surface for sesmax-core's layered-fluid pieces.

:class:`StratificationProfile` holds a layered fluid's thicknesses and
reduced gravities; :class:`ModalTransform` projects between layers and
vertical modes, diagonalising the layer coupling matrix through its
symmetric similarity so the transform is right for unequal layers; the
Helmholtz caches solve ``(nabla^2 - lambda) psi = q`` for a fixed
``lambda`` with spectraldiffx, one per vertical mode in the multimodal
case. An ocean and an atmosphere share them, so they live in the core.

All are also importable from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.gfd.helmholtz import (
    DirichletHelmholtzCache,
    HelmholtzCache,
    MultimodalHelmholtzCache,
    NeumannHelmholtzCache,
    PeriodicHelmholtzCache,
)
from sesmax_core._src.gfd.modal import ModalTransform, StratificationProfile


__all__ = [
    "DirichletHelmholtzCache",
    "HelmholtzCache",
    "ModalTransform",
    "MultimodalHelmholtzCache",
    "NeumannHelmholtzCache",
    "PeriodicHelmholtzCache",
    "StratificationProfile",
]
