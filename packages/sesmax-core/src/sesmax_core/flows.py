"""Optional flowjax bridge — requires the ``flows`` extra.

``pip install sesmax-core[flows]`` (or ``uv sync --extra flows``) pulls in
flowjax; importing this module without it raises with that instruction
rather than a bare ``ModuleNotFoundError`` naming a package the user
never asked for.

See :mod:`sesmax_core._src.flows` for what the bridge does and why
flowjax is not a core dependency.
"""

from __future__ import annotations

from sesmax_core._src.flows import (
    StateAffineBijection,
    learned_prior,
    to_flowjax,
)


__all__ = [
    "StateAffineBijection",
    "learned_prior",
    "to_flowjax",
]
