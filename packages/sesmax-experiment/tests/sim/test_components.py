"""Components are found by entry point, and models by name."""

from __future__ import annotations

import sys
import types
from importlib.metadata import EntryPoint

import pytest

from sesmax_core.component import Component
from sesmax_experiment import components as public
from sesmax_experiment._src import components as private


def _component(name: str, *models: str) -> Component:
    return Component(name=name, models=dict.fromkeys(models, lambda spec: spec))


def _entries(monkeypatch: pytest.MonkeyPatch, **registered: object) -> None:
    """Entry points that load these objects."""
    module = types.ModuleType("registered_components")
    vars(module).update(registered)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    found = [EntryPoint(k, f"{module.__name__}:{k}", public.GROUP) for k in registered]
    monkeypatch.setattr(private, "entry_points", lambda group: found)


def test_the_testing_component_is_installed() -> None:
    testing = public.components()["testing"]
    assert {"lorenz96", "linear_swm"} <= set(testing.models)


def test_a_model_is_found_with_its_component() -> None:
    component, build = public.find_model("lorenz96")
    assert component.name == "testing"
    assert build is component.models["lorenz96"]


def test_a_model_nobody_offers_lists_those_they_do() -> None:
    with pytest.raises(KeyError, match="lorenz96"):
        public.find_model("no_such_model")


def test_a_model_two_components_offer_is_an_error() -> None:
    among = {"a": _component("a", "qg"), "b": _component("b", "qg")}
    with pytest.raises(ValueError, match="offered by"):
        public.find_model("qg", among)


def test_two_components_of_one_name_are_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _entries(monkeypatch, a=_component("ocean", "qg"), b=_component("ocean", "swm"))
    with pytest.raises(ValueError, match="two components are named 'ocean'"):
        public.components()


def test_an_entry_point_must_be_a_component(monkeypatch: pytest.MonkeyPatch) -> None:
    _entries(monkeypatch, a="a string")
    with pytest.raises(TypeError, match="not a Component"):
        public.components()
