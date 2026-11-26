"""The checks of a model before a run."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from equality import float64
from gridax import Mask2D

from sesmax_core.component import AssertionFailedError
from sesmax_ocean.checks import check_pv_inversion, check_static_stability
from sesmax_ocean.qg import BaroclinicQG, BarotropicQG, ReparameterizedQG
from sesmax_ocean.spherical import SphericalQG
from sesmax_ocean.swm import MultilayerShallowWater2D, NonlinearShallowWater2D


def _island() -> Mask2D:
    wet = np.ones((34, 34), dtype=bool)
    wet[12:18, 10:20] = False
    return Mask2D.from_mask(wet)


def test_the_inversion_undoes_the_laplacian() -> None:
    check_pv_inversion(None, BarotropicQG.create(nx=32, ny=24))


@pytest.mark.parametrize("mask", [None, "island"])
def test_the_inversion_on_the_sphere(mask: str | None) -> None:
    land = None if mask is None else _island()
    check_pv_inversion(None, SphericalQG.create(nx=32, ny=32, mask=land))


def test_the_tolerance() -> None:
    model = BarotropicQG.create(nx=32, ny=32)
    with pytest.raises(AssertionFailedError, match="FAILED"):
        check_pv_inversion(None, model, tol=0.0)
    with float64():
        check_pv_inversion(None, BarotropicQG.create(nx=32, ny=32), tol=1e-11)


@pytest.mark.parametrize("factor", [2.0, float("nan")])
def test_a_broken_inversion_fails(factor: float) -> None:
    class Broken(BarotropicQG):
        def _invert_pv(self, q: Any) -> Any:
            return factor * super()._invert_pv(q)

    sound = BarotropicQG.create(nx=32, ny=32)
    fields = {f.name: getattr(sound, f.name) for f in dataclasses.fields(sound)}
    with pytest.raises(AssertionFailedError, match="FAILED"):
        check_pv_inversion(None, Broken(**fields))


@pytest.mark.parametrize(
    "model_class", [BaroclinicQG, ReparameterizedQG, NonlinearShallowWater2D]
)
def test_the_inversion_of_another_model_is_refused(model_class: Any) -> None:
    with pytest.raises(AssertionFailedError, match="no barotropic QG model"):
        check_pv_inversion(None, model_class.create(nx=16, ny=16))


@pytest.mark.parametrize(
    "model_class", [BaroclinicQG, ReparameterizedQG, MultilayerShallowWater2D]
)
def test_stable_layers(model_class: Any) -> None:
    check_static_stability(None, model_class.create(nx=16, ny=16))


def test_unstable_layers_fail() -> None:
    layers = {"H": (500.0, 500.0, 4000.0), "g_prime": (9.81, 0.025, -0.01)}
    model = MultilayerShallowWater2D.create(nx=16, ny=16, **layers)
    with pytest.raises(AssertionFailedError, match=r"interface\(s\) \[2\]"):
        check_static_stability(None, model)


def test_one_layer_has_no_stability() -> None:
    with pytest.raises(AssertionFailedError, match="no layers"):
        check_static_stability(None, BarotropicQG.create(nx=16, ny=16))
