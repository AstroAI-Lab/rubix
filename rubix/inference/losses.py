from typing import Optional, Sequence

import jax.numpy as jnp

from .api import LossFn


def masked_gaussian_nll(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    sigma: Optional[jnp.ndarray] = None,
    inv_variance: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Compute masked Gaussian negative log-likelihood for cube residuals.

    Args:
        prediction (jnp.ndarray): Predicted IFU cube.
        target (jnp.ndarray): Target IFU cube with the same shape.
        sigma (Optional[jnp.ndarray], optional): Per-voxel Gaussian standard
            deviation. Must match ``target`` shape when provided.
            Defaults to ``None``.
        inv_variance (Optional[jnp.ndarray], optional): Per-voxel inverse
            variance. Must match ``target`` shape when provided.
            Defaults to ``None``.
        mask (Optional[jnp.ndarray], optional): Binary selection mask with same
            shape as ``target``. Defaults to ``None``.
        normalize (bool, optional): If ``True``, divide by number of active
            voxels. Defaults to ``True``.
        eps (float, optional): Numerical floor for log/denominator safety.
            Defaults to 1e-12.

    Raises:
        ValueError: If array shapes are inconsistent or both ``sigma`` and
            ``inv_variance`` are provided.

    Returns:
        jnp.ndarray: Scalar Gaussian NLL.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    if sigma is not None and sigma.shape != target.shape:
        raise ValueError("sigma must have the same shape as target")

    if inv_variance is not None and inv_variance.shape != target.shape:
        raise ValueError("inv_variance must have the same shape as target")

    if mask is not None and mask.shape != target.shape:
        raise ValueError("mask must have the same shape as target")

    if sigma is not None and inv_variance is not None:
        raise ValueError("provide only one of sigma or inv_variance")

    residual = prediction - target
    dtype = residual.dtype
    eps_arr = jnp.asarray(eps, dtype=dtype)

    if inv_variance is not None:
        precision = jnp.maximum(inv_variance.astype(dtype), eps_arr)
        log_variance = -jnp.log(precision)
    elif sigma is not None:
        sigma_safe = jnp.maximum(sigma.astype(dtype), eps_arr)
        variance = sigma_safe**2
        precision = 1.0 / variance
        log_variance = jnp.log(variance)
    else:
        precision = jnp.ones_like(residual)
        log_variance = jnp.zeros_like(residual)

    per_voxel_nll = 0.5 * (residual**2 * precision + log_variance)

    if mask is None:
        mask_f = jnp.ones_like(per_voxel_nll)
    else:
        mask_f = mask.astype(dtype)

    total_nll = jnp.sum(per_voxel_nll * mask_f)

    if not normalize:
        return total_nll

    denom = jnp.maximum(jnp.sum(mask_f), eps_arr)
    return total_nll / denom


def huber_data_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    delta: float = 1.0,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Compute masked and weighted Huber loss on IFU cube residuals.

    Args:
        prediction (jnp.ndarray): Predicted IFU cube.
        target (jnp.ndarray): Target IFU cube with same shape as prediction.
        delta (float, optional): Huber transition threshold. Defaults to 1.0.
        mask (Optional[jnp.ndarray], optional): Optional voxel mask.
            Defaults to ``None``.
        weights (Optional[jnp.ndarray], optional): Optional per-voxel weights.
            Defaults to ``None``.
        normalize (bool, optional): If ``True``, divide by effective weighted
            mask sum. Defaults to ``True``.
        eps (float, optional): Denominator floor. Defaults to 1e-12.

    Raises:
        ValueError: If ``delta`` is not strictly positive or array shapes are
            inconsistent.

    Returns:
        jnp.ndarray: Scalar Huber loss.
    """
    if delta <= 0:
        raise ValueError("delta must be strictly positive")

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    if mask is not None and mask.shape != target.shape:
        raise ValueError("mask must have the same shape as target")

    if weights is not None and weights.shape != target.shape:
        raise ValueError("weights must have the same shape as target")

    residual = prediction - target
    abs_residual = jnp.abs(residual)
    delta_arr = jnp.asarray(delta, dtype=residual.dtype)

    quadratic = 0.5 * residual**2
    linear = delta_arr * (abs_residual - 0.5 * delta_arr)
    huber = jnp.where(abs_residual <= delta_arr, quadratic, linear)

    if mask is None:
        mask_f = jnp.ones_like(huber)
    else:
        mask_f = mask.astype(huber.dtype)

    if weights is None:
        weights_f = jnp.ones_like(huber)
    else:
        weights_f = weights.astype(huber.dtype)

    weighted = huber * mask_f * weights_f
    total = jnp.sum(weighted)

    if not normalize:
        return total

    denom = jnp.maximum(
        jnp.sum(mask_f * weights_f),
        jnp.asarray(eps, dtype=huber.dtype),
    )
    return total / denom


def combine_loss_fns(
    loss_fns: Sequence[LossFn],
    weights: Optional[Sequence[float]] = None,
) -> LossFn:
    """Combine multiple prediction-target losses into one scalar objective.

    Args:
        loss_fns (Sequence[LossFn]): Individual loss callables.
        weights (Optional[Sequence[float]], optional): Optional scalar weights
            with same length as ``loss_fns``. Defaults to equal weighting.

    Raises:
        ValueError: If no losses are provided or weight lengths mismatch.

    Returns:
        LossFn: Combined loss function.
    """
    if len(loss_fns) == 0:
        raise ValueError("loss_fns must contain at least one loss function")

    if weights is None:
        coeffs = [1.0] * len(loss_fns)
    else:
        if len(weights) != len(loss_fns):
            raise ValueError("weights must have the same length as loss_fns")
        coeffs = list(weights)

    def _combined(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        values = [
            jnp.asarray(c, dtype=prediction.dtype) * fn(prediction, target)
            for c, fn in zip(coeffs, loss_fns)
        ]
        return sum(values)

    return _combined
