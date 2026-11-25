"""Public surface for sesmax-core's located-field helpers.

A field of a state is an array or a `gridax.CField`, an array that knows its
point of the grid. Use these to write a model once for both: :func:`each` is
element-wise maths that checks the points agree, :func:`map_fields` the same
over a whole state, :func:`values` and :func:`placed` cross between the two.

Not re-exported from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.fields import each, is_field, map_fields, placed, values


__all__ = ["each", "is_field", "map_fields", "placed", "values"]
