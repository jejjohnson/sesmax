# Rotating flows

Rotation is what makes a flow geophysical. The Coriolis force turns whatever moves, so a bump of water does not simply collapse under gravity: it ends up circled by a current that holds it up. This series runs the two families of models built on that balance {cite}`cushman2011introduction,vallis2017atmospheric`. Rotating shallow water holds the slow balanced flow and the fast gravity waves that bring it about; quasi-geostrophy keeps the first and leaves out the second. Each step derives its PDE, runs a model, checks it where an exact or a balanced solution is known, and differentiates through the run. The [C-grid page](../notebooks/cgrid_and_boundaries.ipynb) explains the grid they all share.

| Step | Topic | Model |
| --- | --- | --- |
| [14](../notebooks/step14_gravity_waves_1d.ipynb) | Gravity waves, without rotation | `LinearShallowWater1D` |
| [15](../notebooks/step15_shallow_water_1d.ipynb) | Steepening, and the velocity that rotation adds | `NonlinearShallowWater1D` |
| [16](../notebooks/step16_geostrophic_adjustment.ipynb) | Geostrophic adjustment | `LinearShallowWater2D` |
| [17](../notebooks/step17_shallow_water_2d.ipynb) | The double gyre, with gravity waves | `NonlinearShallowWater2D` |
| [18](../notebooks/step18_barotropic_qg.ipynb) | The double gyre, without them | `BarotropicQG` |

The models of steps 14 to 16 are test problems in `sesmax_core.testing`, and those of steps 17 and 18 are the ocean's, in `sesmax_ocean.swm` and `sesmax_ocean.qg`.

## Balance

A flow in **geostrophic balance** has its pressure gradient held by the Coriolis force:

$$f u = -g \frac{\partial h}{\partial y}, \qquad f v = g \frac{\partial h}{\partial x}$$ (rotating-geostrophy)

where $f$ is the Coriolis parameter, $g$ gravity and $h$ the height of the surface, so the current runs along the height contours. A state that is out of balance adjusts towards {eq}`rotating-geostrophy` by radiating gravity waves. Which of the two fields gives way depends on the size of the disturbance against the **deformation radius**:

$$L_d = \frac{\sqrt{g H_0}}{f}$$ (rotating-deformation-radius)

where $H_0$ is the depth at rest. A disturbance much wider than $L_d$ keeps its height, and the velocity adjusts to it; a much narrower one keeps its vorticity, and the height adjusts. Step 16 runs the adjustment of a step in height.

## Two families of models

The **rotating shallow-water equations** evolve the height and both velocity components:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} + f \hat{\mathbf{z}} \times \mathbf{u} = -g \nabla h, \qquad \frac{\partial h}{\partial t} + \nabla \cdot (h \mathbf{u}) = 0$$ (rotating-shallow-water)

where $\mathbf{u}$ is the velocity and $\hat{\mathbf{z}}$ the vertical unit vector. When the Rossby number $\mathrm{Ro} = U / (f L)$ is small, the flow stays close to balance, and {eq}`rotating-shallow-water` reduces to the **quasi-geostrophic equation**, which evolves one scalar. For a layer under a rigid lid it is

$$\frac{\partial q}{\partial t} + J(\psi, q + \beta y) = 0, \qquad \nabla^2 \psi = q$$ (rotating-quasi-geostrophy)

where $q$ is the potential vorticity anomaly, $\psi$ the streamfunction, $J$ the Jacobian and $\beta = \partial f / \partial y$. The total potential vorticity $q + \beta y$ is carried by the flow and not created, and the flow is recovered from $q$ by the Poisson solve of [step 10](../notebooks/step10_poisson_2d.ipynb).

| | Shallow water | Quasi-geostrophy |
| --- | --- | --- |
| What evolves | $h$, $u$, $v$ | $q$ |
| Solve at each step | none | one Poisson solve |
| Fastest signal | gravity waves, at $\sqrt{g H_0}$ | the currents and the Rossby waves |
| Holds when | the layer is thin against its width | $\mathrm{Ro}$ is small as well |

**The time step** follows from the fastest signal. An explicit shallow-water run needs $\sqrt{g H_0} \, \Delta t / \Delta x$ of order one, the condition of [step 14](../notebooks/step14_gravity_waves_1d.ipynb), although the currents are far slower than the waves. A quasi-geostrophic run needs only $U \Delta t / \Delta x$ of order one, and pays for it with the Poisson solve.

**The Jacobian** decides what a discrete quasi-geostrophic flow conserves. `BarotropicQG` uses Arakawa's form {cite}`arakawa1977computational`, which neither creates nor destroys energy, and step 18 measures this.

## Why the gyres lean west

On a sphere $f$ grows with latitude, and the gradient $\beta$ makes the two directions along a parallel different. A parcel that moves north keeps its total potential vorticity, so its own vorticity falls, and one that moves south gains some. A chain of such displacements is a **Rossby wave**, whose crests travel west. In a basin, the wind's curl $\mathcal{F}$ sets the interior flow through **Sverdrup balance**:

$$\beta v = \mathcal{F}$$ (rotating-sverdrup)

where $v$ is the northward velocity and $\mathcal{F}$ the tendency of the vorticity under the wind. What flows one way in the interior returns in a current on the western wall, whose width is set by friction: $\delta_M = (\nu / \beta)^{1/3}$ for a viscosity $\nu$ (Munk), or $\delta_S = \kappa / \beta$ for a bottom drag $\kappa$ (Stommel), whichever is wider. That width has to be resolved by the grid, and [`check_munk_width`](xref:api#sesmax_ocean.resolution.check_munk_width) says whether it is. Steps 17 and 18 spin up the same two gyres with both families of models.

## Symbols

| Symbol | Meaning | Shape |
| --- | --- | --- |
| $x$, $y$, $t$ | spatial coordinates (m) and time (s) | $\mathbb{R}$ |
| $\mathbf{u} = (u, v)$ | velocity (m/s) | $\mathbb{R}^2$ |
| $\hat{\mathbf{z}}$ | vertical unit vector | $\mathbb{R}^3$ |
| $h$, $H_0$ | height of the surface and depth at rest (m) | $\mathbb{R}$, $\mathbb{R}_{>0}$ |
| $f$, $\beta$ | Coriolis parameter (1/s) and its meridional gradient (1/(m s)) | $\mathbb{R}$, $\mathbb{R}_{>0}$ |
| $g$ | gravity (m/s²) | $\mathbb{R}_{>0}$ |
| $L_d$ | deformation radius (m) | $\mathbb{R}_{>0}$ |
| $q$, $\psi$ | potential vorticity anomaly (1/s) and streamfunction (m²/s) | $\mathbb{R}$ |
| $J(\cdot, \cdot)$ | Jacobian | $\mathbb{R}$ |
| $\mathcal{F}$ | tendency of the vorticity under the wind (1/s²) | $\mathbb{R}$ |
| $U$, $L$, $\mathrm{Ro}$ | velocity scale (m/s), length scale (m) and Rossby number | $\mathbb{R}_{>0}$ |
| $\nu$, $\kappa$ | viscosity (m²/s) and bottom drag (1/s) | $\mathbb{R}_{>0}$ |
| $\delta_M$, $\delta_S$ | Munk and Stommel widths (m) | $\mathbb{R}_{>0}$ |
| $\Delta x$, $\Delta t$ | grid spacing (m) and time step (s) | $\mathbb{R}_{>0}$ |
