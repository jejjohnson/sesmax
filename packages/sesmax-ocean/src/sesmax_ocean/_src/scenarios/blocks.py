"""Reading the blocks of a run spec: a key that is not known is an error.

A misspelt ``wind_amplitude`` or ``lateral_viscosity`` would otherwise leave the
default in place, and the run would be another one than was asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from gridax.schemes import TVD, WENO, Scheme, Upwind

from sesmax_ocean._src.scenarios.types import Scenario


#: The schemes of the face values, under the names somax's configs use.
SCHEMES: dict[str, Scheme] = {
    "upwind1": Upwind(1),
    "upwind2": Upwind(2),
    "upwind3": Upwind(3),
    "weno3": WENO(3),
    "weno5": WENO(5),
    "weno7": WENO(7),
    "weno9": WENO(9),
    "wenoz5": WENO(5, variant="z"),
    "minmod": TVD("minmod"),
    "van_leer": TVD("van_leer"),
    "superbee": TVD("superbee"),
    "mc": TVD("mc"),
}


def scheme(name: str) -> Scheme:
    """The scheme of a name.

    Raises:
        ValueError: If there is none.
    """
    if name not in SCHEMES:
        raise ValueError(f"unknown method {name!r}; known: {sorted(SCHEMES)}")
    return SCHEMES[name]


def known(where: str, block: Mapping[str, Any], allowed: Iterable[str]) -> None:
    """Every key of ``block`` is one of ``allowed``.

    Raises:
        ValueError: If one is not.
    """
    if unknown := sorted(set(block) - set(allowed)):
        raise ValueError(
            f"{where}: unknown key(s) {unknown}; supported: {sorted(allowed)}"
        )


def read(
    where: str, block: Mapping[str, Any], readers: Mapping[str, Callable[[Any], Any]]
) -> dict[str, Any]:
    """The values of ``block``, each through the reader of its key.

    Raises:
        ValueError: If a key has no reader.
    """
    known(where, block, readers)
    return {key: readers[key](value) for key, value in block.items()}


def model_params(
    name: str, model: Mapping[str, Any], readers: Mapping[str, Callable[[Any], Any]]
) -> dict[str, Any]:
    """The ``model.params`` of a run spec, as arguments of the model's ``create``."""
    params = read(f"{name}: model.params", model.get("params", {}), readers)
    if "method" in params:
        params["scheme"] = params.pop("method")
    return params


def initial_params(
    name: str, scenario: Scenario, defaults: Mapping[str, float]
) -> dict[str, float]:
    """The numbers of an initial condition, with those that are not given."""
    given = read(
        f"{name}: scenario.initial_condition.params",
        scenario.initial_condition.params,
        dict.fromkeys(defaults, float),
    )
    return {**defaults, **given}


def starts_from(name: str, scenario: Scenario, *kinds: str) -> str:
    """The initial condition of a scenario, if it is one of ``kinds``.

    Raises:
        NotImplementedError: If it is another one.
    """
    kind = scenario.initial_condition.kind
    if kind not in kinds:
        raise NotImplementedError(
            f"{name}: initial_condition.type={kind!r} is not supported, only {kinds}"
        )
    if kind == "at_rest":
        initial_params(name, scenario, {})
    return kind


def wind_profile(scenario: Scenario) -> dict[str, Any]:
    """The wind profile a scenario names, as an argument; its amplitude is not read."""
    return {k: v for k, v in scenario.forcing_params.items() if k == "wind_profile"}


def layers(
    name: str, stratification: Mapping[str, Any]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The thicknesses and the reduced gravities of a layered model.

    Args:
        name: The model, for the message.
        stratification: The ``model.stratification`` block.

    Returns:
        ``H`` (m) and ``g_prime`` (m/s²), one of each per layer.

    Raises:
        ValueError: If one is missing, or they differ in length.
    """
    where = f"{name}: model.stratification"
    block = read(where, stratification, dict.fromkeys(("H", "g_prime"), _floats))
    if missing := sorted({"H", "g_prime"} - set(block)):
        raise ValueError(f"{where} missing required key(s) {missing}")
    H, g_prime = block["H"], block["g_prime"]
    if len(H) != len(g_prime):
        raise ValueError(
            f"{where}: {len(H)} thicknesses and {len(g_prime)} reduced gravities; "
            "both must match the number of layers"
        )
    return H, g_prime


def _floats(values: Iterable[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def no_layers(name: str, model: Mapping[str, Any]) -> None:
    """A one-layer model is given no stratification.

    Raises:
        ValueError: If it is.
    """
    if model.get("stratification"):
        raise ValueError(f"{name} has one layer: it reads no model.stratification")
