# Vertical modes

The vertical structure of a layered fluid: the coupling matrix built from layer thicknesses and reduced gravities, its decomposition into Rossby radii and layer-to-mode transforms, and the lift that applies a horizontal operator to every layer.

`barotropic_filter` removes the fast external gravity wave of a multilayer shallow-water model implicitly. It takes its variable-coefficient Helmholtz solve as an argument, for example the multigrid solver of `gridax.solvers`.

::: sesmax_core.vertical
