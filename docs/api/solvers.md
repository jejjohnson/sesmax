# Solvers

Memory-safe IMEX integration for stiff term-based models: each implicit stage is solved by a Newton iteration with GMRES linear solves, so memory grows linearly with the state size.

::: sesmax_core.solvers
