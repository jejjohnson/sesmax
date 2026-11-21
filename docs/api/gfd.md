# Layered fluids

The pieces a layered ocean and a layered atmosphere share: stratification profiles, the transform between layers and vertical modes, and the three elliptic inversions, the streamfunction from vorticity, the pressure from a divergence and the QG potential-vorticity inversion.

The modal transform diagonalises the layer coupling matrix through its symmetric similarity, because the matrix itself is not symmetric when the layers differ in thickness, and a symmetric eigensolver applied to it directly decomposes the wrong matrix.

The inversions take the full `[Ny, Nx]` array with its ghost ring and put the wall on the C-grid wall of the unknown's staggering. `method` picks the solve: `"spectral"` on the rectangle, or `"cg"`, `"capacitance"` or `"multigrid"` on a masked domain, the last two with a solver pre-built by gridax-jax for the mask and the Helmholtz parameter. The [Elliptic inversions guide](https://jejjohnson.github.io/sesmax/elliptic-inversions/) explains the walls, the methods and known boundary values.

::: sesmax_core.gfd
