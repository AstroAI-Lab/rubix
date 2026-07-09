"""Flux- and S/N-scaled per-voxel noise models for IFU likelihoods.

The synthetic VI cycle historically used a single constant ``sigma`` cube, which
has to be hand-tuned to each cube's brightness (Phase 3 gate in
``docs/vi_science_validation_plan.md``). This module builds a per-voxel Gaussian
sigma tied to the flux instead, combining three physically-motivated terms:

- a **relative / calibration** term ``relative_noise * |flux|`` (its inverse is
  the bright-end signal-to-noise ratio),
- a **shot / Poisson** term with variance ``poisson_scale * max(flux, 0)``, and
- an additive **floor** ``floor`` (read-noise / background) that keeps sigma
  strictly positive in empty voxels.

    sigma = sqrt( (relative_noise * |flux|)^2 + poisson_scale * max(flux, 0)
                  + floor^2 )

With ``relative_noise = poisson_scale = 0`` this reduces to a constant ``floor``
cube, so it is a strict generalization of the constant model. The result is
differentiable and suitable for use as the ``sigma`` argument of
:func:`rubix.inference.optimize_variational_ifu_cube`.
"""

from __future__ import annotations

import jax.numpy as jnp


def flux_scaled_sigma(
    flux: jnp.ndarray,
    relative_noise: float = 0.0,
    floor: float = 1e-6,
    poisson_scale: float = 0.0,
) -> jnp.ndarray:
    """Build a per-voxel Gaussian sigma cube tied to the flux.

    Args:
        flux (jnp.ndarray): Flux cube (typically the target IFU cube).
        relative_noise (float, optional): Fractional noise; ``1 / relative_noise``
            is the approximate bright-end signal-to-noise ratio. Defaults to 0.0.
        floor (float, optional): Additive noise floor (read-noise / background)
            in flux units, also the numerical floor keeping sigma positive.
            Defaults to 1e-6.
        poisson_scale (float, optional): Shot-noise variance per unit flux.
            Defaults to 0.0.

    Raises:
        ValueError: If any coefficient is negative.

    Returns:
        jnp.ndarray: Per-voxel sigma with the same shape as ``flux``.
    """
    if relative_noise < 0.0:
        raise ValueError("relative_noise must be non-negative")
    if floor < 0.0:
        raise ValueError("floor must be non-negative")
    if poisson_scale < 0.0:
        raise ValueError("poisson_scale must be non-negative")

    flux = jnp.asarray(flux)
    dtype = flux.dtype
    relative_term = (jnp.asarray(relative_noise, dtype) * jnp.abs(flux)) ** 2
    poisson_term = jnp.asarray(poisson_scale, dtype) * jnp.maximum(
        flux, jnp.asarray(0.0, dtype)
    )
    floor_term = jnp.asarray(floor, dtype) ** 2
    return jnp.sqrt(relative_term + poisson_term + floor_term)
