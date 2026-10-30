"""Public surface for sesmax-core's pipekit ``Operator`` bridge.

Import this module to expose sesmax forward models as
:class:`pipekit.Operator` s. pipekit is a base dependency, so no extra
install step is needed. Kept out of ``sesmax_core``'s top-level
``__init__`` so plain ``import sesmax_core`` doesn't import pipekit (the
core stays structural-only); ``import sesmax_core.operators`` pulls it in.
"""

from __future__ import annotations

from sesmax_core._src.operator_term import OperatorTerm
from sesmax_core._src.operators import ModelOp


__all__ = [
    "ModelOp",
    "OperatorTerm",
]
