# Ocean: basin data

Needs the `[data]` extra (`pip install sesmax-ocean[data]`). A bundle is the mask, the depth and the forcing of a basin on one grid, as a Zarr store. Those that `placeholder_bundle` makes are somax's, value for value: made-up shapes with the layout of a real bundle. No scenario reads one yet (sesmax#115).

The command `sesmax-ocean data` builds and fetches the bundles with DVC. `init` takes the URL of a DVC remote and writes it for the checkout alone; in somax it asked for a backend as well and changed the configuration that git tracks. `write` makes one bundle without DVC.

::: sesmax_ocean.basins
