from typing import Optional

import jax.numpy as jnp

from .api import LossFn


def masked_weighted_mse(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Compute masked/weighted squared error for IFU cubes.

    Args:
        prediction (jnp.ndarray): Model prediction cube.
        target (jnp.ndarray): Target cube with the same shape as ``prediction``.
        mask (Optional[jnp.ndarray], optional): Binary mask selecting voxels that
            contribute to the objective. Must have the same shape as ``target``.
            Defaults to ``None`` (all voxels active).
        weights (Optional[jnp.ndarray], optional): Per-voxel non-negative weights
            with the same shape as ``target``. Defaults to ``None``.
        normalize (bool, optional): If ``True``, divide by the effective weight
            sum (or number of active voxels if no weights are provided).
            Defaults to ``True``.
        eps (float, optional): Numerical floor for normalization denominator.
            Defaults to 1e-12.

    Raises:
        ValueError: If input shapes do not match.

    Returns:
        jnp.ndarray: Scalar weighted squared-error objective.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    if mask is not None and mask.shape != target.shape:
        raise ValueError("mask must have the same shape as target")

    if weights is not None and weights.shape != target.shape:
        raise ValueError("weights must have the same shape as target")

    residual_sq = (prediction - target) ** 2

    if mask is not None:
        mask_f = mask.astype(residual_sq.dtype)
        residual_sq = residual_sq * mask_f

    if weights is not None:
        weights_f = weights.astype(residual_sq.dtype)
        residual_sq = residual_sq * weights_f
        if mask is not None:
            denom = jnp.sum(weights_f * mask_f)
        else:
            denom = jnp.sum(weights_f)
    else:
        if mask is not None:
            denom = jnp.sum(mask_f)
        else:
            denom = jnp.asarray(residual_sq.size, dtype=residual_sq.dtype)

    numerator = jnp.sum(residual_sq)

    if not normalize:
        return numerator

    denom = jnp.maximum(denom, jnp.asarray(eps, dtype=residual_sq.dtype))
    return numerator / denom


def build_ifu_cube_loss(
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> LossFn:
    """Build a reusable IFU-cube loss function with optional mask and weights.

    Args:
        mask (Optional[jnp.ndarray], optional): Binary voxel selection mask.
            Defaults to ``None``.
        weights (Optional[jnp.ndarray], optional): Per-voxel weights.
            Defaults to ``None``.
        normalize (bool, optional): Whether to normalize by effective weights.
            Defaults to ``True``.
        eps (float, optional): Denominator floor used when normalizing.
            Defaults to 1e-12.

    Returns:
        LossFn: Callable ``loss_fn(prediction, target) -> scalar``.
    """

    def _loss_fn(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return masked_weighted_mse(
            prediction=prediction,
            target=target,
            mask=mask,
            weights=weights,
            normalize=normalize,
            eps=eps,
        )

    return _loss_fn
