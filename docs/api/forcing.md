# Forcing

Forcing fields and the reduced-order bases that let a data-assimilation run estimate a forcing: a few coefficients, each with a prior standard deviation, times spatial and temporal modes.

The module also holds the forcing operators, which take the fields they act on and return a tendency: wind stress and wind-stress curl, linear and quadratic bottom drag, Rayleigh damping, their raw tendency functions, and the mask-free `linear_drag` and `rayleigh_relaxation`.
A `ForcingProtocol` is a field of time and place, and an `AbstractForcing` is an operator on a state, so a model uses the first as an input and the second inside its tendency.

::: sesmax_core.forcing
