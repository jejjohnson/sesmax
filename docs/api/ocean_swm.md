# Ocean: shallow-water models

The shallow-water models, moved from somax. Each reproduces somax's tendencies, end states and diagnostics to a relative 1e-12 in float64 (`packages/sesmax-ocean/tests/swm/test_equal_to_somax.py`), except for what was fixed since:

- The models conserve mass. somax took no flux through the edge faces, so the cells along the edge had a wrong thickness tendency.
- The Coriolis term and the Bernoulli gradient are right at a periodic edge. somax read ghosts there that no one had filled.

::: sesmax_ocean.swm
