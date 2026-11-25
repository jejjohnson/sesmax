"""Reduced-order forcing presets named after ocean fields.

Moved from somax 9cb19b3: the ocean presets of
``somax/_src/core/forcing_bank.py``. The constructors they call are in
:mod:`sesmax_core.forcing`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from sesmax_core import (
    BasisForcing,
    ConstantInTime,
    TemporalBasis,
    VectorBasisForcing,
    spatial_from_divfree,
    spatial_from_fourier,
    spatial_from_gabor,
    spatial_from_rbf,
    tile_in_time,
)
from sesmax_core.domain import Domain


def ssh_geostrophic(
    domain: Domain,
    *,
    n_scales: int = 6,
    base_scale: float = 20e3,
    slope: float = 4.0,
    amplitude: float = 2e-6,
    oversample: float = 1.0,
    windows: tuple[Float[Array, " m_t"], Float[Array, " m_t"]] | None = None,
) -> BasisForcing:
    """SSH geostrophic preset: a radial-Gabor frame with a wavenumber-law prior.

    The overcomplete multiscale Gabor frame is the workhorse behind sea-surface
    -height mapping; the prior follows the steep mesoscale wavenumber law
    ``sigma^2 ~ k ** -slope`` (``slope`` near four). By default the forcing is
    constant in time (the static-coefficient case that the existing ``jax.grad``
    parameter path handles); pass ``windows`` to spread it over Gaussian time
    windows for the time-distributed (weak-constraint) case.

    Args:
        domain: The model domain.
        n_scales: Number of dyadic scales in the frame.
        base_scale: Finest envelope scale ``L_0`` (in domain units).
        slope: Spectral slope of the wavenumber prior law.
        amplitude: Overall prior variance scale.
        oversample: Centre density per scale.
        windows: Optional ``(centers, widths)`` for the temporal gate; ``None``
            keeps the forcing constant in time.

    Returns:
        A :class:`~sesmax_core.BasisForcing` with zero initial
        coefficients, ready to drop into a model RHS via
        :class:`~sesmax_core.ForcingTerm`.
    """
    spatial = spatial_from_gabor(
        domain,
        n_scales=n_scales,
        base_scale=base_scale,
        slope=slope,
        amplitude=amplitude,
        oversample=oversample,
    )
    if windows is None:
        temporal: TemporalBasis = ConstantInTime(m=spatial.Phi.shape[1])
    else:
        spatial, temporal = tile_in_time(spatial, windows[0], windows[1])
    return BasisForcing(
        coeffs=jnp.zeros(spatial.Phi.shape[1]),
        spatial=spatial,
        temporal=temporal,
        grid_shape=tuple(domain.Nx),
    )


def sss_coastal(
    domain: Domain,
    centers: Float[Array, "m ndim"],
    widths: Float[Array, " m"],
    *,
    kernel: str = "wendland_c2",
    std: Float[Array, " m"] | float = 1.0,
    windows: tuple[Float[Array, " m_t"], Float[Array, " m_t"]] | None = None,
) -> BasisForcing:
    """Coastal SSS preset: placeable radial atoms with a prescribed prior.

    Sea-surface-salinity forcing localised where the physics is — radial atoms
    placed at coastlines / river mouths rather than spread over the open ocean.
    Uses a compactly supported Wendland kernel by default so each atom is
    exactly zero past its width. As with :func:`ssh_geostrophic`, ``windows``
    switches from the constant-in-time to the time-distributed regime.

    Args:
        domain: The model domain.
        centers: Atom centres of shape ``(m, ndim)`` (e.g. river-mouth locations).
        widths: Per-atom width of shape ``(m,)``.
        kernel: Radial kernel name (compact-support Wendland by default).
        std: Prescribed per-centre prior std; a scalar is broadcast.
        windows: Optional ``(centers, widths)`` for the temporal gate.

    Returns:
        A :class:`~sesmax_core.BasisForcing` over the placed atoms.
    """
    spatial = spatial_from_rbf(domain, centers, widths, kernel=kernel, std=std)
    if windows is None:
        temporal: TemporalBasis = ConstantInTime(m=spatial.Phi.shape[1])
    else:
        spatial, temporal = tile_in_time(spatial, windows[0], windows[1])
    return BasisForcing(
        coeffs=jnp.zeros(spatial.Phi.shape[1]),
        spatial=spatial,
        temporal=temporal,
        grid_shape=tuple(domain.Nx),
    )


def sst_frontal(
    domain: Domain,
    *,
    num_basis_per_dim: int | tuple[int, ...] = 12,
    length_scale: float = 1.0,
    nu: float = 1.5,
    variance: float = 1.0,
    windows: tuple[Float[Array, " m_t"], Float[Array, " m_t"]] | None = None,
) -> BasisForcing:
    """SST preset: a smooth Matérn (HSGP) field over the box-Laplacian eigenbasis.

    A principled smooth-field prior for sea-surface temperature: the box
    eigenbasis weighted by the Matérn spectral density. Constant in time by
    default; pass ``windows`` for the time-distributed regime.

    Args:
        domain: The model domain.
        num_basis_per_dim: Per-axis number of 1D modes.
        length_scale: Matérn length scale.
        nu: Matérn smoothness.
        variance: Marginal variance.
        windows: Optional ``(centers, widths)`` Gaussian temporal gate.

    Returns:
        A :class:`~sesmax_core.BasisForcing`.
    """
    spatial = spatial_from_fourier(
        domain,
        num_basis_per_dim=num_basis_per_dim,
        length_scale=length_scale,
        nu=nu,
        variance=variance,
    )
    if windows is None:
        temporal: TemporalBasis = ConstantInTime(m=spatial.Phi.shape[1])
    else:
        spatial, temporal = tile_in_time(spatial, windows[0], windows[1])
    return BasisForcing(
        coeffs=jnp.zeros(spatial.Phi.shape[1]),
        spatial=spatial,
        temporal=temporal,
        grid_shape=tuple(domain.Nx),
    )


def geostrophic_currents(
    domain: Domain,
    *,
    num_basis_per_dim: int | tuple[int, ...] = 8,
    length_scale: float = 1.0,
    nu: float = 1.5,
    variance: float = 1.0,
) -> VectorBasisForcing:
    """Vector preset: incompressible ``(u, v)`` current-error forcing.

    A divergence-free velocity frame (geonnax ``divfree_basis``) with a Matérn
    kinetic-energy prior, constant in time. Lift it onto a model's velocity
    components with ``ForcingTerm(forcing, place=add_vector_to(("u", "v")))``.

    Args:
        domain: A 2D model domain.
        num_basis_per_dim: Per-axis number of 1D stream-function modes.
        length_scale: Matérn length scale of the prior.
        nu: Matérn smoothness of the prior.
        variance: Marginal variance of the prior.

    Returns:
        A :class:`~sesmax_core.VectorBasisForcing` with zero initial
        coefficients.
    """
    spatial = spatial_from_divfree(
        domain,
        num_basis_per_dim=num_basis_per_dim,
        length_scale=length_scale,
        nu=nu,
        variance=variance,
    )
    return VectorBasisForcing(
        coeffs=jnp.zeros(spatial.Phi.shape[1]),
        spatial=spatial,
        temporal=ConstantInTime(m=spatial.Phi.shape[1]),
        grid_shape=tuple(domain.Nx),
    )
