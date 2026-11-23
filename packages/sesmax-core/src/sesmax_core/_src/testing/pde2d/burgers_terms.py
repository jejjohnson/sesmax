"""2D Burgers equation expressed as a composable :class:`Term` algebra.

This is the worked example for the term-kernel decomposition (the
pipekit-refactor "Level 3" idea). The canonical :class:`Burgers2D`
hand-writes its right-hand side as one ``vector_field`` method::

    du/dt = -ops.advect(u, v) + nu * laplacian(u)        (and likewise v)

Here the same physics is split into two reusable kernels —
:class:`Burgers2DAdvection` and :class:`Burgers2DDiffusion` — that
compose into the RHS::

    rhs = advection + nu * diffusion

Both kernels are :class:`~sesmax_core._src.terms.Term` instances, so the
assembled tree is differentiable end to end (the viscosity ``nu`` enters
as a :class:`~sesmax_core._src.terms.Scaled` coefficient, a JAX leaf) and
the assembled :class:`Burgers2DTermModel` still satisfies the sesmax model
contract (and hence ``pipekit_cycle.ForwardModel`` via ``step``).

The kernels are *integration-strategy neutral*: the same advection and
diffusion terms can be summed for a fully-explicit solve, or the
diffusion term can be tagged :func:`~sesmax_core._src.terms.implicit` so a
splitting (IMEX) solver routes the stiff Laplacian through its implicit
stage. ``Burgers2DTermModel.create(..., imex=True)`` does exactly that —
something the monolithic model cannot express without rewriting its
``build_terms``.

The accompanying tests assert this model is numerically identical to the
canonical :class:`Burgers2D` (same tendencies, same explicit trajectory),
which is the evidence that the decomposition is faithful.
"""

from __future__ import annotations

import equinox as eqx
from gridax import (
    CartesianCGrid2D,
    GridOps,
    Mask2D,
    fill_ghosts,
)
from gridax.schemes import Scheme, Upwind
from jaxtyping import Array, PyTree

from sesmax_core._src.model import TermModel
from sesmax_core._src.terms import IMPLICIT, Kind, Scaled, Term, implicit
from sesmax_core._src.testing.pde2d._advective_form import PERIODIC, advective_tendency
from sesmax_core._src.testing.pde2d.burgers import Burgers2D, Burgers2DState
from sesmax_core._src.types import as_parameter


#: The default scheme of the advective face values.
UPWIND = Upwind(1)


class Burgers2DAdvection(Term):
    r"""Advective tendency kernel ``-(u·∂x + v·∂y)`` for both velocities.

    Reproduces the advection half of :meth:`Burgers2D.vector_field`: the
    transport velocities are interpolated to faces and the C-grid
    advection operator is applied to each component. Marked explicit
    (the default) — advection is non-stiff.

    Args:
        ops: gridax's operations on the grid and the mask.
        scheme: Scheme passed to the advection operator.
    """

    ops: GridOps
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def __call__(
        self, t: float, state: Burgers2DState, args: PyTree | None = None
    ) -> Burgers2DState:
        adv_u, adv_v = advective_tendency(
            state.u,
            state.v,
            self.ops,
            scheme=self.scheme,
        )
        return Burgers2DState(u=adv_u, v=adv_v)


class Burgers2DDiffusion(Term):
    r"""Diffusive tendency kernel ``laplacian(·)`` for both velocities.

    Reproduces the (unscaled) diffusion half of
    :meth:`Burgers2D.vector_field`. The viscosity ``nu`` is *not* baked
    in here — it is applied by scaling this term (``nu * diffusion``) so
    the coefficient stays a differentiable JAX leaf and the kernel itself
    is reusable across models.

    Args:
        ops: gridax's operations on the grid and the mask.
            ``laplacian``).
        kind: IMEX integration label. Defaults to ``"explicit"``;
            :meth:`Burgers2DTermModel.create` tags it implicit when
            ``imex=True``.
    """

    ops: GridOps
    _kind: Kind = eqx.field(static=True, default="explicit")

    def __call__(
        self, t: float, state: Burgers2DState, args: PyTree | None = None
    ) -> Burgers2DState:
        return Burgers2DState(
            u=self.ops.laplacian(state.u),
            v=self.ops.laplacian(state.v),
        )

    @property
    def kind(self) -> Kind:
        return self._kind


class Burgers2DTermModel(TermModel):
    r"""2D Burgers model assembled from composable term kernels.

    A faithful, term-based reconstruction of :class:`Burgers2D`. The RHS
    is ``advection + nu * diffusion``; periodic boundary conditions are
    enforced before each term evaluation (matching the canonical model).

    The viscosity is **not** stored as a separate field — it lives inside
    the term tree as the :class:`~sesmax_core._src.terms.Scaled`
    coefficient on the diffusion kernel, which is the single source of
    truth for the dynamics (and the leaf ``jax.grad`` flows to). The
    :attr:`nu` property reads it back for inspection / diagnostics. This
    avoids duplicating a parameter outside the tree that actually drives
    integration — a leaf stored twice would let the two copies diverge
    and silently disconnect gradients.

    Args:
        terms: The assembled RHS term tree (built by :meth:`create`).
        grid: The Arakawa C-grid (static; used for diagnostics).
    """

    grid: CartesianCGrid2D

    def apply_boundary_conditions(self, state: Burgers2DState) -> Burgers2DState:
        """Enforce periodic boundary conditions on both velocity fields."""
        bcs = Burgers2DState(u=PERIODIC, v=PERIODIC)
        return fill_ghosts(state, self.grid, bcs)

    @property
    def nu(self) -> Array:
        """Kinematic viscosity, read back from the diffusion kernel's coeff.

        Walks the assembled term tree to the
        :class:`~sesmax_core._src.terms.Scaled` wrapping the
        :class:`Burgers2DDiffusion` kernel and returns its coefficient.
        On a gradient pytree this returns the gradient w.r.t. ``nu``.
        """
        from sesmax_core._src.terms import Scaled, _Kinded

        def _find(term: Term) -> Array | None:
            if isinstance(term, Scaled) and isinstance(term.term, Burgers2DDiffusion):
                return term.coeff
            if isinstance(term, Scaled | _Kinded):
                return _find(term.term)
            return None

        for summand in self.terms.terms:
            coeff = _find(summand)
            if coeff is not None:
                return coeff
        raise ValueError("no diffusion kernel found in term tree")

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        scheme: Scheme = UPWIND,
        mask: Mask2D | None = None,
        *,
        imex: bool = False,
    ) -> Burgers2DTermModel:
        """Build a term-based Burgers model.

        Args mirror :meth:`Burgers2D.create`, plus:

        Args:
            imex: When ``True``, tag the diffusion term implicit so that
                :func:`~sesmax_core._src.terms.build_diffrax_terms`
                produces a ``diffrax.MultiTerm`` and a splitting solver
                (e.g. ``diffrax.KenCarp3``) integrates the stiff Laplacian
                implicitly. When ``False`` (default) the whole RHS is one
                explicit ``ODETerm`` — numerically identical to
                :class:`Burgers2D`.

        Returns:
            A ``Burgers2DTermModel`` instance.
        """
        grid = CartesianCGrid2D.from_interior(nx, ny, Lx, Ly)
        nu_arr = as_parameter(nu)

        ops = GridOps(grid, mask)
        advection = Burgers2DAdvection(
            ops=ops,
            scheme=scheme,
        )
        diffusion = Burgers2DDiffusion(
            ops=ops,
            _kind=IMPLICIT if imex else "explicit",
        )

        scaled_diffusion = Scaled(diffusion, nu_arr)
        rhs: Term = advection + (
            implicit(scaled_diffusion) if imex else scaled_diffusion
        )

        return Burgers2DTermModel(terms=rhs, grid=grid)

    @staticmethod
    def from_model(model: Burgers2D, *, imex: bool = False) -> Burgers2DTermModel:
        """Build a term model matching an existing :class:`Burgers2D`.

        Reuses the source model's grid, mask, and advection scheme so the
        two are numerically comparable.

        Args:
            model: A canonical Burgers2D instance.
            imex: Tag diffusion implicit (see :meth:`create`).

        Returns:
            A ``Burgers2DTermModel`` mirroring ``model``.
        """
        # Already a Params leaf — plain array or wrapper — so it is
        # passed through rather than converted; TermModel.build_terms
        # unwraps it at evaluation time.
        nu_arr = model.params.nu
        advection = Burgers2DAdvection(
            ops=model.ops,
            scheme=model.scheme,
        )
        diffusion = Burgers2DDiffusion(
            ops=model.ops,
            _kind=IMPLICIT if imex else "explicit",
        )
        scaled_diffusion = Scaled(diffusion, nu_arr)
        rhs: Term = advection + (
            implicit(scaled_diffusion) if imex else scaled_diffusion
        )
        return Burgers2DTermModel(terms=rhs, grid=model.grid)
