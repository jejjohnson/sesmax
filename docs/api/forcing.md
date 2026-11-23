# Forcing

Forcing fields and the reduced-order bases that let a data-assimilation run estimate a forcing: a few coefficients, each with a prior standard deviation, times spatial and temporal modes.

The module also holds the forcing operators, which take the fields they act on and return a tendency: wind stress and wind-stress curl, linear and quadratic bottom drag, Rayleigh damping, their raw tendency functions, and the mask-free `linear_drag` and `rayleigh_relaxation`.
A `ForcingProtocol` is a field of time and place, and an `AbstractForcing` is an operator on a state, so a model uses the first as an input and the second inside its tendency.

`InterpolatedForcing.from_data` reads samples of a forcing between their times.
`method="linear"` and `method="cubic"` build diffrax's paths, and a functax 1-D scheme (`CubicSpline()` for a C² spline, `Pchip()` for a reconstruction that stays between neighbouring samples, ...) builds an integrix `SampledPath`.
`outside` says what a scheme does past the samples; it extrapolates by default, as diffrax's paths do.
The reconstruction has a kink at every sample time, where a step that straddles it loses order, so pass `step_ts` to the step control (`imex_stepsize_controller(step_ts=forcing.step_ts)`) or take fixed steps that divide the sample spacing.

::: sesmax_core.forcing
