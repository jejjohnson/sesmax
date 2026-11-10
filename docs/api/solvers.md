# Solvers

Memory-safe IMEX integration for stiff term-based models: each implicit stage is solved by a Newton iteration with GMRES linear solves, so memory grows linearly with the state size.

`stable_step` is the largest stable step for an explicit solve, from the actual spacings, speeds and diffusivities.

The three functions live in [integrix](https://github.com/jejjohnson/integrix) (`integrix.march`), and `sesmax_core.solvers` re-exports them unchanged, so either import gives the same objects.
`stable_step` returns a 0-d JAX array in JAX's default float (float32 unless `jax_enable_x64` is set) rather than a Python `float`, and it traces under `jax.jit`.
Python and NumPy inputs are still checked eagerly and raise a `ValueError` (integrix's `ValidationError`) whose message starts with `stable_step:`; JAX-array inputs are checked with `equinox.error_if`.

::: sesmax_core.solvers
