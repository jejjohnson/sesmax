"""Element-wise maths on states whose fields may know their point.

A field of a state is an array or a `gridax.CField`, an array with named dims
and the point of the grid it sits on. A `CField` has no arithmetic operators:
`each` is how a model writes a formula once for both, and it is also the check
that the operands sit on one point.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu
from gridax import CField, combine
from jaxtyping import Array, PyTree


__all__ = ["each", "is_field", "map_fields", "placed", "values"]


def is_field(x: Any) -> bool:
    """Whether ``x`` is a `gridax.CField` (the ``is_leaf`` of a state)."""
    return isinstance(x, CField)


def each(fn: Callable[..., Any], *operands: Any) -> Any:
    """``fn`` of arrays, or `gridax.combine` of it if an operand is a field.

    Args:
        fn: An element-wise function of the operands' arrays.
        *operands: Arrays, fields and numbers.

    Returns:
        An array, or a field on the point of the operands.

    Raises:
        gridax.StaggerMismatchError: If two fields sit on different points.
    """
    if any(is_field(x) for x in operands):
        return combine(fn, *operands)
    return fn(*operands)


def values(x: Any) -> Any:
    """The array of a field; an array as it is."""
    return x.data if is_field(x) else x


def placed(array: Array, like: Any, at: str) -> Any:
    """``array`` as a field at the point ``at``, if ``like`` is a field.

    Args:
        array: Values on the last dims of ``like``.
        like: A field or an array of the state.
        at: The point ``array`` sits on, ``"T"``, ``"U"``, ``"V"`` or ``"X"``.

    Returns:
        A field with the last dims of ``like``, or ``array`` itself.
    """
    if not is_field(like):
        return array
    return CField(array, like.dims[like.ndim - array.ndim :], loc=at)


def map_fields(fn: Callable[..., Any], state: PyTree, *rest: PyTree) -> PyTree:
    """``fn`` of the arrays of each field of ``state``, on the field's point.

    ``jax.tree_util.tree_map`` for trees of which some hold a plain array or a
    number where ``state`` holds a `gridax.CField`.

    Args:
        fn: A function of arrays that keeps the shape of the first.
        state: A state, with array or `gridax.CField` fields.
        *rest: Trees of the same structure, up to that difference.

    Returns:
        A state of the structure of ``state``.
    """

    def one(x: Any, *others: Any) -> Any:
        out = fn(values(x), *(values(other) for other in others))
        return x.with_data(out) if is_field(x) else out

    return jtu.tree_map(one, state, *rest, is_leaf=is_field)
