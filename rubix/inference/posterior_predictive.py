from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp

from rubix.core.data import RubixData

from .api import LossFn, forward
from .parameterization import TransformTree, apply_transforms
from .variational import sample_diag_gaussian

ParamsTree = Mapping[str, Mapping[str, Any]]


def sample_posterior_predictive_cubes(
    pipeline: Any,
    posterior_mean_params: ParamsTree,
    posterior_log_std_params: ParamsTree,
    static_data: RubixData,
    num_samples: int,
    transforms: Optional[TransformTree] = None,
    noise_key: Optional[jnp.ndarray] = None,
    seed: int = 0,
) -> jnp.ndarray:
    """Draw posterior predictive cubes from a diagonal Gaussian posterior.

    Args:
        pipeline (Any): Pipeline-like object exposing ``run_sharded``.
        posterior_mean_params (ParamsTree): Mean parameters in unconstrained
            space.
        posterior_log_std_params (ParamsTree): Log-std parameters in
            unconstrained space.
        static_data (RubixData): Baseline model data.
        num_samples (int): Number of posterior predictive draws.
        transforms (Optional[TransformTree], optional): Optional transform tree
            to constrained space. Defaults to ``None``.
        noise_key (Optional[jnp.ndarray], optional): Optional noise key passed
            to each forward pass. Defaults to ``None``.
        seed (int, optional): Random seed. Defaults to 0.

    Raises:
        ValueError: If ``num_samples`` is not strictly positive.

    Returns:
        jnp.ndarray: Predictive samples stacked as ``(num_samples, *cube_shape)``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be strictly positive")

    base_key = jax.random.PRNGKey(seed)
    sample_keys = jax.random.split(base_key, num_samples)

    cubes = []
    for key in sample_keys:
        sampled_unconstrained = sample_diag_gaussian(
            mean=posterior_mean_params,
            log_std=posterior_log_std_params,
            key=key,
        )
        if transforms is None:
            sampled_constrained = sampled_unconstrained
        else:
            sampled_constrained = apply_transforms(
                params=sampled_unconstrained,
                transforms=transforms,
                direction="forward",
            )
        cubes.append(
            forward(
                pipeline=pipeline,
                params=sampled_constrained,
                static_data=static_data,
                noise_key=noise_key,
            )
        )

    return jnp.stack(cubes, axis=0)


def summarize_predictive_cube_samples(samples: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Summarize predictive cube samples via moments and quantiles.

    Args:
        samples (jnp.ndarray): Predictive cubes with leading sample axis.

    Raises:
        ValueError: If ``samples`` does not include a sample axis.

    Returns:
        dict[str, jnp.ndarray]: Summary maps with keys ``mean``, ``std``,
        ``p16``, ``p50``, and ``p84``.
    """
    if samples.ndim < 2:
        raise ValueError("samples must include a leading sample axis")

    return {
        "mean": jnp.mean(samples, axis=0),
        "std": jnp.std(samples, axis=0),
        "p16": jnp.percentile(samples, 16.0, axis=0),
        "p50": jnp.percentile(samples, 50.0, axis=0),
        "p84": jnp.percentile(samples, 84.0, axis=0),
    }


def compute_residual_products(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    sigma: Optional[jnp.ndarray] = None,
    inv_variance: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
) -> dict[str, jnp.ndarray]:
    """Compute residual, absolute residual, and chi2-like products.

    Args:
        prediction (jnp.ndarray): Predicted IFU cube.
        target (jnp.ndarray): Target IFU cube with matching shape.
        sigma (Optional[jnp.ndarray], optional): Per-voxel sigma cube.
            Defaults to ``None``.
        inv_variance (Optional[jnp.ndarray], optional): Per-voxel inverse
            variance cube. Defaults to ``None``.
        mask (Optional[jnp.ndarray], optional): Optional binary mask.
            Defaults to ``None``.

    Raises:
        ValueError: If shapes are inconsistent or both ``sigma`` and
            ``inv_variance`` are provided.

    Returns:
        dict[str, jnp.ndarray]: Residual-derived maps.
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
    abs_residual = jnp.abs(residual)

    if inv_variance is not None:
        chi2 = residual**2 * inv_variance.astype(residual.dtype)
    elif sigma is not None:
        eps_arr = jnp.asarray(1e-12, dtype=residual.dtype)
        sigma_safe = jnp.maximum(sigma.astype(residual.dtype), eps_arr)
        chi2 = residual**2 / (sigma_safe**2)
    else:
        chi2 = residual**2

    if mask is None:
        mask_f = jnp.ones_like(residual)
    else:
        mask_f = mask.astype(residual.dtype)

    return {
        "residual": residual,
        "abs_residual": abs_residual,
        "chi2": chi2,
        "masked_residual": residual * mask_f,
        "masked_chi2": chi2 * mask_f,
    }


def summarize_masked_metrics(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    loss_fn: Optional[LossFn] = None,
) -> dict[str, float]:
    """Compute scalar masked summary metrics for science reporting.

    Args:
        prediction (jnp.ndarray): Predicted IFU cube.
        target (jnp.ndarray): Target IFU cube.
        mask (Optional[jnp.ndarray], optional): Optional binary mask.
            Defaults to ``None``.
        loss_fn (Optional[LossFn], optional): Optional custom scalar loss.
            Defaults to ``None``.

    Raises:
        ValueError: If prediction/target shapes (or mask shape) are invalid.

    Returns:
        dict[str, float]: Summary metrics including ``mse``, ``mae``, and
        optionally ``custom_loss``.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    if mask is not None and mask.shape != target.shape:
        raise ValueError("mask must have the same shape as target")

    residual = prediction - target
    if mask is None:
        mask_f = jnp.ones_like(residual)
    else:
        mask_f = mask.astype(residual.dtype)

    denom = jnp.maximum(jnp.sum(mask_f), jnp.asarray(1e-12, dtype=residual.dtype))
    mse = jnp.sum((residual**2) * mask_f) / denom
    mae = jnp.sum(jnp.abs(residual) * mask_f) / denom

    metrics = {"mse": float(mse), "mae": float(mae)}
    if loss_fn is not None:
        metrics["custom_loss"] = float(loss_fn(prediction, target))
    return metrics
