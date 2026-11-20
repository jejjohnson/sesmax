"""Forcing operators: tendency contributions of wind, drag and relaxation.

Moved from gridax-jax 0.2.0 (``gridax_jax/_src/forcing/`` and
``operators/relaxation.py``). A :class:`AbstractForcing` is called with the
fields it acts on and returns a tendency; ``ForcingProtocol`` in
``sesmax_core._src.forcing`` is a field of time and place instead.
"""
