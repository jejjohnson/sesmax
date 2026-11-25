# Ocean: QG models

The quasi-geostrophic models, moved from somax. Each reproduces somax's tendencies, end states and diagnostics to a relative 1e-12 in float64 (`packages/sesmax-ocean/tests/qg/test_equal_to_somax.py`), except beside land: no operation of gridax reads a dry cell, so the Laplacian takes no flux through a coast, and the rotated gradient averages the wet cells around a corner. somax's finitevolX read what land held there.

::: sesmax_ocean.qg
