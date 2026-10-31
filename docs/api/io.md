# I/O

Needs the `[xarray]` extra (`pip install sesmax-core[xarray]`). Round-trippable conversion between model states and self-describing xarray Datasets, and Zarr save and load. A Dataset records its state class, so `dataset_to_state` recovers it; stores written by somax load as the matching sesmax classes.

::: sesmax_core.io
