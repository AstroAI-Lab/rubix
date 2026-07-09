"""Structured variational posterior families.

The default mean-field posterior (diagonal Gaussian in unconstrained space)
cannot represent correlated parameter geometry such as the age--metallicity
ridge diagnosed in ``docs/vi_science_validation_plan.md``. This module adds a
**low-rank-plus-diagonal Gaussian**

    q(z) = N(mu, Sigma),   Sigma = diag(exp(2 * log_std)) + W W^T

over the *raveled* unconstrained latent vector ``z``. The rank-``r`` factor
``W`` (shape ``(D, r)``) introduces off-diagonal correlations while keeping the
cost ``O(D * r + r^3)`` rather than the ``O(D^2)`` of a full-covariance
posterior. Setting ``W = 0`` recovers the mean-field posterior exactly, so this
family strictly generalizes the diagonal one.

All functions operate through :func:`jax.flatten_util.ravel_pytree`, so the
mean/log-std pytrees keep the same structure used elsewhere and the factor rows
follow the ravel order of that pytree.
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
from beartype.typing import Any
from jax.flatten_util import ravel_pytree

ParamsTree = Mapping[str, Mapping[str, Any]]


def init_low_rank_factor(
    mean: ParamsTree,
    rank: int,
    key: jnp.ndarray,
    init_scale: float = 1e-2,
) -> jnp.ndarray:
    """Initialize the low-rank factor ``W`` for a mean pytree.

    Args:
        mean (ParamsTree): Posterior mean pytree (defines the latent dimension).
        rank (int): Number of factor columns ``r``.
        key (jnp.ndarray): PRNG key.
        init_scale (float, optional): Standard deviation of the Gaussian factor
            initialization. Small values keep the initial posterior close to the
            diagonal one. Defaults to 1e-2.

    Raises:
        ValueError: If ``rank`` is not strictly positive.

    Returns:
        jnp.ndarray: Factor matrix of shape ``(D, rank)``.
    """
    if rank <= 0:
        raise ValueError("rank must be strictly positive")
    flat, _ = ravel_pytree(mean)
    return init_scale * jax.random.normal(
        key, (flat.shape[0], int(rank)), dtype=flat.dtype
    )


def sample_low_rank_gaussian(
    mean: ParamsTree,
    log_std: ParamsTree,
    factor: jnp.ndarray,
    key: jnp.ndarray,
) -> dict[str, dict[str, Any]]:
    """Draw a reparameterized sample from a low-rank-plus-diagonal Gaussian.

    ``z = mu + exp(log_std) * eps_diag + W @ eps_rank`` with independent standard
    normal ``eps_diag`` (shape ``(D,)``) and ``eps_rank`` (shape ``(r,)``), which
    reproduces the covariance ``diag(exp(2*log_std)) + W W^T``.

    Args:
        mean (ParamsTree): Posterior mean pytree.
        log_std (ParamsTree): Posterior diagonal log-std pytree.
        factor (jnp.ndarray): Low-rank factor ``W`` of shape ``(D, r)``.
        key (jnp.ndarray): PRNG key.

    Returns:
        dict[str, dict[str, Any]]: Sample pytree matching ``mean``'s structure.
    """
    flat_mean, unravel = ravel_pytree(mean)
    flat_log_std, _ = ravel_pytree(log_std)
    n_dim = flat_mean.shape[0]
    rank = factor.shape[1]

    key_diag, key_rank = jax.random.split(key)
    eps_diag = jax.random.normal(key_diag, (n_dim,), dtype=flat_mean.dtype)
    eps_rank = jax.random.normal(key_rank, (rank,), dtype=flat_mean.dtype)

    z = flat_mean + jnp.exp(flat_log_std) * eps_diag + factor @ eps_rank
    return unravel(z)


def kl_low_rank_to_standard_normal(
    mean: ParamsTree,
    log_std: ParamsTree,
    factor: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ``KL(q || N(0, I))`` for a low-rank-plus-diagonal Gaussian ``q``.

    Uses the matrix determinant lemma so the log-determinant costs ``O(D r + r^3)``
    instead of ``O(D^3)``. Reduces exactly to the diagonal KL when ``W = 0``.

    Args:
        mean (ParamsTree): Posterior mean pytree.
        log_std (ParamsTree): Posterior diagonal log-std pytree.
        factor (jnp.ndarray): Low-rank factor ``W`` of shape ``(D, r)``.

    Returns:
        jnp.ndarray: Scalar KL divergence.
    """
    flat_mean, _ = ravel_pytree(mean)
    flat_log_std, _ = ravel_pytree(log_std)
    n_dim = flat_mean.shape[0]
    rank = factor.shape[1]

    diag_var = jnp.exp(2.0 * flat_log_std)  # d_i
    trace = jnp.sum(diag_var) + jnp.sum(factor**2)
    quad = jnp.sum(flat_mean**2)
    logdet_diag = jnp.sum(2.0 * flat_log_std)

    # log det(Sigma) = log det(D) + log det(I_r + W^T D^{-1} W).
    dinv_w = factor / diag_var[:, None]
    capacitance = jnp.eye(rank, dtype=flat_mean.dtype) + factor.T @ dinv_w
    logdet_capacitance = jnp.linalg.slogdet(capacitance)[1]
    logdet_sigma = logdet_diag + logdet_capacitance

    return 0.5 * (trace + quad - n_dim - logdet_sigma)


def low_rank_marginal_log_std(
    log_std: ParamsTree,
    factor: jnp.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return the per-latent marginal log-std of a low-rank-plus-diagonal Gaussian.

    The marginal variance of latent ``i`` is ``exp(2*log_std_i) + sum_j W_ij^2``.
    Reporting this (rather than the raw diagonal ``log_std``) lets diagonal
    downstream samplers reproduce the correct *marginal* posterior widths.

    Args:
        log_std (ParamsTree): Posterior diagonal log-std pytree.
        factor (jnp.ndarray): Low-rank factor ``W`` of shape ``(D, r)``.

    Returns:
        dict[str, dict[str, Any]]: Marginal log-std pytree matching ``log_std``.
    """
    flat_log_std, unravel = ravel_pytree(log_std)
    marginal_var = jnp.exp(2.0 * flat_log_std) + jnp.sum(factor**2, axis=1)
    return unravel(0.5 * jnp.log(marginal_var))
