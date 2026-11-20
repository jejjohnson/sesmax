"""Public surface for the vertical structure of a layered fluid.

:func:`build_coupling_matrix` builds the layer coupling matrix ``A`` from layer
thicknesses and reduced gravities, :func:`decompose_vertical_modes` diagonalises
it into Rossby radii and the layer-to-mode transforms, and :func:`layer_to_mode`
and :func:`mode_to_layer` apply them. :func:`multilayer` lifts a 2-D horizontal
operator over the layer axis, and :func:`barotropic_filter` removes the fast
external gravity wave of a multilayer shallow-water model implicitly, with the
Helmholtz solve passed in.

:class:`~sesmax_core.ModalTransform` wraps the first two for a stratification
profile.
"""

from __future__ import annotations

from sesmax_core._src.vertical.barotropic import barotropic_filter
from sesmax_core._src.vertical.multilayer import multilayer
from sesmax_core._src.vertical.vertical_modes import (
    build_coupling_matrix,
    decompose_vertical_modes,
    layer_to_mode,
    mode_to_layer,
)


__all__ = [
    "barotropic_filter",
    "build_coupling_matrix",
    "decompose_vertical_modes",
    "layer_to_mode",
    "mode_to_layer",
    "multilayer",
]
