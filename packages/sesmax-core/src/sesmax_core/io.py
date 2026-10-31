"""Public IO API for sesmax-core (zarr-backed xarray); needs the ``[xarray]`` extra.

Re-exports from :mod:`sesmax_core._src.io`. See that subpackage for the canonical
docstrings.
"""

from sesmax_core._src.io import (
    append_to_dataset,
    apply_scale_metadata,
    dataset_to_state,
    load_dataset,
    save_dataset,
    scales_attrs,
    snapshots_to_dataset,
    state_to_dataset,
    transform_attrs,
)


__all__ = [
    "append_to_dataset",
    "apply_scale_metadata",
    "dataset_to_state",
    "load_dataset",
    "save_dataset",
    "scales_attrs",
    "snapshots_to_dataset",
    "state_to_dataset",
    "transform_attrs",
]
