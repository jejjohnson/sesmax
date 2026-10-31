"""pipekit Operators for the test problems.

Moved from somax 9cb19b3: ``Burgers2DOp`` of ``somax/_src/operators.py``,
the flat-recipe (serialisable) operator for the term-based 2D Burgers model.
The general ``ModelOp`` it subclasses lives in ``sesmax_core.operators``.
"""

from __future__ import annotations

from sesmax_core._src.operators import ModelOp
from sesmax_core._src.testing.pde2d.burgers_terms import Burgers2DTermModel


class Burgers2DOp(ModelOp):
    """Serializable pipekit Operator for the term-based 2D Burgers model.

    Wraps :class:`~sesmax_core._src.testing.pde2d.burgers_terms.Burgers2DTermModel`.
    All constructor arguments are JSON primitives, so::

        op = Burgers2DOp(nx=32, ny=32, nu=0.05, dt=1e-3)
        assert pipekit.loads(pipekit.dumps(op)).get_config() == op.get_config()

    round-trips the build recipe. ``op(state)`` advances the Burgers
    state by one ``dt`` step; the Operator drives ``pipekit_cycle.Cycle``
    and composes with the rest of pipekit.

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x.
        Ly: Domain length in y.
        nu: Kinematic viscosity (diffusion coefficient).
        method: Advection reconstruction method.
        imex: Tag diffusion implicit for IMEX integration (see
            :meth:`Burgers2DTermModel.create`).
        dt: Default step size for :meth:`_apply`.
    """

    # Flat primitive recipe → faithful serial round-trip (unlike the
    # general ModelOp wrapper).
    forbid_in_yaml = False
    __config_mixin_auto__ = True

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        method: str = "upwind1",
        imex: bool = False,
        dt: float = 0.001,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nu = nu
        self.method = method
        self.imex = imex
        self.dt = dt
        self._model = Burgers2DTermModel.create(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            nu=nu,
            method=method,
            imex=imex,
        )


__all__ = [
    "Burgers2DOp",
    "ModelOp",
]
