"""The ocean, as a component a run spec can name."""

from __future__ import annotations

from typing import Any

import numpy as np
import paramax

from sesmax_core import Model
from sesmax_core.component import Component
from sesmax_ocean._src.checks import check_pv_inversion, check_static_stability
from sesmax_ocean._src.eval import compute_eval_metrics
from sesmax_ocean._src.resolution import (
    check_deformation_radius,
    check_equatorial_deformation_radius,
    check_munk_width,
    check_stommel_width,
)
from sesmax_ocean._src.scenarios.pairs import (
    MODELS,
    build_ocean_run,
    compatible_models,
)
from sesmax_ocean._src.scenarios.scenarios import SCENARIOS


_VELOCITY = {"h": "m", "u": "m/s", "v": "m/s"}
_VORTICITY = {"q": "1/s"}
#: The models that step a potential vorticity; the others step a velocity.
_QG = ("barotropic_qg", "multilayer_qg", "spherical_qg")

#: What decides a model's numbers beside its equations.
_SETTINGS = ("bc_type", "poisson_bc", "scheme", "cg_tol", "cg_max_steps")


def _describe_numerics(model: Model) -> dict[str, Any]:
    """The class, the discrete choices and the friction of a model."""
    out: dict[str, Any] = {"model_class": type(model).__name__}
    for name in _SETTINGS:
        value = getattr(model, name, None)
        if value is None and hasattr(model, "swm"):
            value = getattr(model.swm, name, None)
        if value is not None:
            out[name] = value if isinstance(value, int | float | str) else repr(value)
    params = paramax.unwrap(model).params
    for name in ("lateral_viscosity", "bottom_drag"):
        out[name] = np.asarray(getattr(params, name)).tolist()
    return out


#: The ocean's models, scenarios, checks and metrics.
COMPONENT = Component(
    name="ocean",
    models=dict.fromkeys(MODELS, build_ocean_run),
    scenarios={
        name: compatible_models(name)
        for name, entry in SCENARIOS.items()
        if entry.build is not None
    },
    checks={
        "deformation_radius": check_deformation_radius,
        "equatorial_deformation_radius": check_equatorial_deformation_radius,
        "munk_width": check_munk_width,
        "pv_inversion": check_pv_inversion,
        "static_stability": check_static_stability,
        "stommel_width": check_stommel_width,
    },
    evaluate=compute_eval_metrics,
    field_units={name: _VORTICITY if name in _QG else _VELOCITY for name in MODELS},
    describe_numerics=_describe_numerics,
)
