# Ocean: checks of a model

A run spec asks for them by name, `pv_inversion` and `static_stability`, beside those of the [resolution](ocean_resolution.md). In somax `pv_inversion` inverted the initial state of the run, which is zero for every barotropic model, so it passed whatever the solver did; here it inverts a random field.

::: sesmax_ocean.checks
