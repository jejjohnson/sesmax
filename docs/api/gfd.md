# Layered fluids

The pieces a layered ocean and a layered atmosphere share: stratification profiles, the transform between layers and vertical modes, and cached Helmholtz solvers.

The modal transform diagonalises the layer coupling matrix through its symmetric similarity, because the matrix itself is not symmetric when the layers differ in thickness, and a symmetric eigensolver applied to it directly decomposes the wrong matrix.

::: sesmax_core.gfd
