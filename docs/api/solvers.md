# Solvers

Memory-safe IMEX integration for stiff term-based models: each implicit stage is solved by a Newton iteration with GMRES linear solves, so memory grows linearly with the state size.

`stable_step` is the largest stable step for an explicit solve, from the actual spacings, speeds and diffusivities.

::: sesmax_core.solvers
