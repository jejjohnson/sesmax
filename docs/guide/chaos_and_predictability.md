# Chaos and predictability

Before the full fluid models, it is worth meeting deterministic chaos in its smallest form. Edward Lorenz's systems distil the property that makes weather and ocean forecasting hard: a perfectly known, perfectly deterministic system whose trajectories nonetheless become unpredictable, because minuscule differences in the initial state grow exponentially {cite}`lorenz1963deterministic,lorenz1996predictability`. Being cheap, they are also the standard test beds for data assimilation.

| Page | System | What it shows |
| --- | --- | --- |
| [Lorenz '63](../notebooks/lorenz63_simulation.ipynb) | three variables from Rayleigh–Bénard convection {cite}`lorenz1963deterministic` | the strange attractor, gradients through the solver, the adjoint methods, and the Lyapunov exponent |
| [Lorenz '96](../notebooks/lorenz96_simulation.ipynb) | $D_x$ variables on a periodic ring {cite}`lorenz1996predictability` | spatiotemporal chaos, forcing regimes, and ensemble spread |
| [Lorenz '96 two-tier](../notebooks/lorenz96t_simulation.ipynb) | slow variables coupled to fast ones | multiscale dynamics, the setting for learned sub-grid closures |

## The limit on forecast skill

Two states that start a tiny distance $\delta_0$ apart diverge, on average, as

$$\delta(t) \sim \delta_0 e^{\lambda t}$$ (chaos-lyapunov)

where $\lambda > 0$ is the leading Lyapunov exponent. Growth saturates once the gap reaches the size of the attractor itself, at which point the two forecasts are effectively independent. The horizon {eq}`chaos-lyapunov` implies is intrinsic to the system and not a numerical artefact, and the time it sets, measured in units of $1 / \lambda$, is the window in which a forecast or an assimilation can add information.

## Symbols

| Symbol | Meaning | Shape |
| --- | --- | --- |
| $\delta(t)$, $\delta_0$ | separation of two trajectories, and its initial value | $\mathbb{R}_{\ge 0}$ |
| $\lambda$ | leading Lyapunov exponent | $\mathbb{R}$ |
| $D_x$ | number of variables on the Lorenz '96 ring | count |
| $t$ | time | $\mathbb{R}$ |
