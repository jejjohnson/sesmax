# Ocean: shallow-water models

The shallow-water models, moved from somax. Each reproduces somax's tendencies, end states and diagnostics to a relative 1e-12 in float64 (`packages/sesmax-ocean/tests/swm/test_equal_to_somax.py`), except for what was fixed since:

- The models conserve mass. somax took no flux through the edge faces, so the cells along the edge had a wrong thickness tendency.
- The Coriolis term and the Bernoulli gradient are right at a periodic edge. somax read ghosts there that no one had filled.
- The viscous term is the Laplacian of a velocity component at its own points, with the ghosts of the boundary conditions. somax diffused it as a tracer, with no flux through the edge faces.
- Walls and coasts are free slip. The potential vorticity on them is `f / h`, and no friction acts along a coast. In somax it was zero on coasts and on two of the four walls, which halved the Coriolis term beside them.
- The states a run returns have no flow through the walls. In somax the tendency on two of the walls added up there, and spoiled the kinetic energy beside them.

::: sesmax_ocean.swm
