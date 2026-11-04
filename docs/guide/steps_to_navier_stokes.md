# Steps to Navier–Stokes

The geophysical models sesmax is built for are made of a few elementary processes: transport (advection), smoothing (diffusion), their nonlinear coupling, and an elliptic constraint that ties velocity to vorticity. This series builds up to the incompressible Navier–Stokes equations one process at a time, after Lorena Barba's CFD Python, in one dimension and then in two {cite}`leveque2002finite,durran2010numerical`. Each step derives its PDE, runs a model from `sesmax_core.testing`, checks it against an analytic solution or a benchmark where one exists, and differentiates through the run. The [C-grid page](../notebooks/cgrid_and_boundaries.ipynb) explains the grid they all share.

| Steps | Topic | Models |
| --- | --- | --- |
| [1](../notebooks/step01_linear_convection_1d.ipynb), [2](../notebooks/step02_nonlinear_convection_1d.ipynb), [3](../notebooks/step03_diffusion_1d.ipynb), [4](../notebooks/step04_burgers_1d.ipynb) | Convection, diffusion and Burgers in 1D | `LinearConvection1D`, `NonlinearConvection1D`, `Diffusion1D`, `Burgers1D` |
| [5](../notebooks/step05_linear_convection_2d.ipynb), [6](../notebooks/step06_nonlinear_convection_2d.ipynb), [7](../notebooks/step07_diffusion_2d.ipynb), [8](../notebooks/step08_burgers_2d.ipynb) | The same four in 2D | `LinearConvection2D`, `NonlinearConvection2D`, `Diffusion2D`, `Burgers2D` |
| [9](../notebooks/step09_laplace_2d.ipynb), [10](../notebooks/step10_poisson_2d.ipynb), [11](../notebooks/step11_helmholtz_2d.ipynb) | Elliptic solves | `PoissonSolver2D`, `HelmholtzSolver2D` |
| [12](../notebooks/step12_cavity_flow.ipynb), [13](../notebooks/step13_channel_flow.ipynb) | Navier–Stokes in a cavity, a channel and a periodic box | `IncompressibleNS2D` |

## The numerical lesson in each process

Each process carries a lesson that survives all the way up to the shallow-water and quasi-geostrophic models, and each one limits an explicit time step differently.

| Process | Equation | Explicit step limit |
| --- | --- | --- |
| Advection | $\partial u / \partial t + c \partial u / \partial x = 0$ | hyperbolic, $c \Delta t / \Delta x$ of order one |
| Diffusion | $\partial u / \partial t = \nu \partial^2 u / \partial x^2$ | parabolic, $\nu \Delta t / \Delta x^2 \le 1/2$ |
| Burgers | $\partial u / \partial t + u \partial u / \partial x = \nu \partial^2 u / \partial x^2$ | both, plus the grid Péclet number below |
| Poisson | $\nabla^2 \psi = -\omega$ | none: it has no time derivative |

**Advection** is the one-dimensional ancestor of the gravity-wave CFL condition that governs every shallow-water run: an explicit scheme sees information only within a few cells per step, so $c \Delta t / \Delta x$ must stay of order one.

**Diffusion** carries a limit that scales with $\Delta x^2$, not with $\Delta x$, so refining the grid by two demands a four-fold smaller step. That is why stiff diffusion is the term to tag `implicit` for an IMEX integration.

**Burgers** couples the two {cite}`burgers1948mathematical`. Where $u$ is large the wave moves faster, so a smooth profile steepens, and viscosity resists until the two balance in a front of width of order $\nu / U$. The grid Péclet number $\mathrm{Pe}_\Delta = U \Delta x / \nu$ says whether that front is resolved: above about 2 it is not, and the advection scheme decides what the front looks like.

**The Poisson solve** is what two dimensions add. An incompressible flow derives from a streamfunction, with $u = \partial \psi / \partial y$ and $v = -\partial \psi / \partial x$, so it cannot evolve its vorticity $\omega$ without solving $\nabla^2 \psi = -\omega$ at every step. This is a global, instantaneous coupling with no time step of its own {cite}`durran2010numerical,cushman2011introduction`. The cavity in step 12 welds advection, diffusion and this inversion into one loop, and the same structure reappears in the quasi-geostrophic ocean models.

## Symbols

| Symbol | Meaning | Shape |
| --- | --- | --- |
| $x$, $y$, $t$ | spatial coordinates and time | $\mathbb{R}$ |
| $u$, $v$ | the field, or the velocity components | $\mathbb{R}$ |
| $\omega$, $\psi$ | vorticity and streamfunction | $\mathbb{R}$ |
| $c$, $\nu$ | wave speed and viscosity | $\mathbb{R}$, $\mathbb{R}_{>0}$ |
| $U$ | velocity scale | $\mathbb{R}_{>0}$ |
| $\Delta x$, $\Delta t$ | grid spacing and time step | $\mathbb{R}_{>0}$ |
| $\mathrm{Pe}_\Delta$ | grid Péclet number | $\mathbb{R}_{\ge 0}$ |
