# Ocean: scenarios

A run spec names a scenario and a model; `build_ocean_run` builds the two, and the package registers it as a component, so `sesmax-experiment` finds the models by name. `SCENARIOS` and `MODELS` hold the entries. Of the six scenarios of somax only the double gyre can be built: the others wait for the data of their basins (sesmax#115).

The eight run configs of somax are in `packages/sesmax-ocean/configs/`. Each one starts where somax's does, and its first 20 steps agree to a relative 1e-12 in float64, with the fixes since the port put back (`packages/sesmax-ocean/tests/scenarios/test_equal_to_somax.py`).

Some things differ from somax:

- A key of a block that is not known is an error. somax refused one in `scenario.forcing` only, so a misspelt `lateral_viscosity` ran without viscosity.
- A dimensionless run takes no `model.params` and no `model.stratification`, because its numbers determine the model. somax refused some of them and dropped the rest.
- A model refuses a mask, a depth or a field of forcing that its builder does not read. somax dropped them.
- A dimensionless run on the sphere has the ghosts of its mask, and is dry on land.
- The linear shallow-water model is a test problem of the core, so the testing component offers `linear_swm`.

::: sesmax_ocean.scenarios
