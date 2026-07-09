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


# ---------------------------------------------------------------------------
# Per-group (block-diagonal) Gaussian posterior.
#
# A single global low-rank factor cannot independently correlate many separate
# parameter pairs (e.g. P independent per-particle age-metallicity ridges would
# each need their own factor direction). The block-diagonal family gives every
# specified group of latents its own dense k x k covariance while the remaining
# latents stay diagonal -- the natural family for per-spaxel/per-particle
# coupling (validation-plan next step 3).
# ---------------------------------------------------------------------------


def build_particle_block_index_map(
    mean: ParamsTree,
    couplings: "list[tuple[str, str]]",
) -> jnp.ndarray:
    """Build a per-particle index map over the raveled latent for coupled fields.

    Each entry of ``couplings`` names a leaf (component, field) whose leading
    axis indexes particles and which contributes one scalar per particle (leaf
    shape ``(P,)`` or ``(P, 1)``). The returned map groups, for every particle,
    the raveled-latent positions of those fields into one block.

    Args:
        mean (ParamsTree): Posterior mean pytree (defines the ravel order).
        couplings (list[tuple[str, str]]): ``(component, field)`` leaves to
            couple per particle, e.g. ``[("stars", "age"), ("stars",
            "metallicity")]``.

    Raises:
        ValueError: If fewer than two couplings are given or the coupled leaves
            do not share a single particle count.

    Returns:
        jnp.ndarray: Integer index map of shape ``(num_particles, len(couplings))``.
    """
    if len(couplings) < 2:
        raise ValueError("couplings must name at least two leaves to couple")
    flat, unravel = ravel_pytree(mean)
    index_tree = unravel(jnp.arange(flat.shape[0], dtype=flat.dtype))
    columns = []
    particle_count = None
    for component, field in couplings:
        leaf = jnp.asarray(index_tree[component][field])
        column = leaf.reshape(leaf.shape[0], -1)[:, 0]
        if particle_count is None:
            particle_count = column.shape[0]
        elif column.shape[0] != particle_count:
            raise ValueError("coupled leaves must share the same particle count")
        columns.append(column)
    return jnp.stack(columns, axis=1).astype(jnp.int32)


def init_block_cholesky(
    num_groups: int,
    block_size: int,
    key: jnp.ndarray,
    init_log_std: float = -2.0,
    offdiag_scale: float = 1e-2,
) -> jnp.ndarray:
    """Initialize raw block-Cholesky parameters near a diagonal posterior.

    Args:
        num_groups (int): Number of blocks ``G``.
        block_size (int): Latents per block ``k``.
        key (jnp.ndarray): PRNG key for the off-diagonal initialization.
        init_log_std (float, optional): Initial log-std placed on the block
            diagonals. Defaults to -2.0.
        offdiag_scale (float, optional): Std of the strictly-lower entries.
            Small values start near a diagonal posterior. Defaults to 1e-2.

    Returns:
        jnp.ndarray: Raw parameters of shape ``(num_groups, block_size,
        block_size)``.
    """
    raw = offdiag_scale * jax.random.normal(
        key, (int(num_groups), int(block_size), int(block_size))
    )
    diag_idx = jnp.arange(int(block_size))
    return raw.at[:, diag_idx, diag_idx].set(init_log_std)


def build_block_cholesky(raw: jnp.ndarray) -> jnp.ndarray:
    """Map raw block parameters to lower-triangular Cholesky factors.

    The strictly-lower entries of ``raw`` are used directly and the diagonal is
    exponentiated to guarantee a positive-definite ``L L^T``.

    Args:
        raw (jnp.ndarray): Raw parameters of shape ``(G, k, k)``.

    Returns:
        jnp.ndarray: Lower-triangular factors ``L`` of shape ``(G, k, k)``.
    """
    block_size = raw.shape[-1]
    strictly_lower = jnp.tril(raw, k=-1)
    diag = jnp.exp(jnp.diagonal(raw, axis1=-2, axis2=-1))  # (G, k)
    eye = jnp.eye(block_size, dtype=raw.dtype)
    return strictly_lower + eye * diag[..., None, :]


def sample_block_gaussian(
    mean: ParamsTree,
    log_std: ParamsTree,
    block_raw: jnp.ndarray,
    block_index_map: jnp.ndarray,
    key: jnp.ndarray,
) -> dict[str, dict[str, Any]]:
    """Sample a block-diagonal Gaussian (dense blocks + diagonal remainder).

    Grouped latents are sampled jointly from their block covariance
    ``L_g L_g^T``; all other latents keep the diagonal ``exp(log_std)`` scale.

    Args:
        mean (ParamsTree): Posterior mean pytree.
        log_std (ParamsTree): Diagonal log-std pytree (used for ungrouped latents).
        block_raw (jnp.ndarray): Raw block parameters ``(G, k, k)``.
        block_index_map (jnp.ndarray): Index map ``(G, k)`` into the raveled latent.
        key (jnp.ndarray): PRNG key.

    Returns:
        dict[str, dict[str, Any]]: Sample pytree matching ``mean``'s structure.
    """
    flat_mean, unravel = ravel_pytree(mean)
    flat_log_std, _ = ravel_pytree(log_std)
    n_dim = flat_mean.shape[0]
    num_groups, block_size = block_index_map.shape

    key_diag, key_block = jax.random.split(key)
    eps_diag = jax.random.normal(key_diag, (n_dim,), dtype=flat_mean.dtype)
    z = flat_mean + jnp.exp(flat_log_std) * eps_diag

    chol = build_block_cholesky(block_raw)  # (G, k, k)
    eps_block = jax.random.normal(
        key_block, (num_groups, block_size), dtype=flat_mean.dtype
    )
    mu_block = flat_mean[block_index_map]  # (G, k)
    block_z = mu_block + jnp.einsum("gij,gj->gi", chol, eps_block)
    z = z.at[block_index_map.reshape(-1)].set(block_z.reshape(-1))
    return unravel(z)


def kl_block_to_standard_normal(
    mean: ParamsTree,
    log_std: ParamsTree,
    block_raw: jnp.ndarray,
    block_index_map: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ``KL(q || N(0, I))`` for a block-diagonal Gaussian ``q``.

    The ungrouped latents contribute the usual diagonal KL; each block
    contributes the exact multivariate-Gaussian KL against ``N(0, I_k)``.

    Args:
        mean (ParamsTree): Posterior mean pytree.
        log_std (ParamsTree): Diagonal log-std pytree.
        block_raw (jnp.ndarray): Raw block parameters ``(G, k, k)``.
        block_index_map (jnp.ndarray): Index map ``(G, k)`` into the raveled latent.

    Returns:
        jnp.ndarray: Scalar KL divergence.
    """
    flat_mean, _ = ravel_pytree(mean)
    flat_log_std, _ = ravel_pytree(log_std)
    n_dim = flat_mean.shape[0]
    _, block_size = block_index_map.shape
    grouped = block_index_map.reshape(-1)

    is_grouped = jnp.zeros((n_dim,), dtype=bool).at[grouped].set(True)
    diag_var = jnp.exp(2.0 * flat_log_std)
    diag_terms = diag_var + flat_mean**2 - 1.0 - 2.0 * flat_log_std
    diag_kl = 0.5 * jnp.sum(jnp.where(is_grouped, 0.0, diag_terms))

    chol = build_block_cholesky(block_raw)  # (G, k, k)
    mu_block = flat_mean[block_index_map]  # (G, k)
    trace = jnp.sum(chol**2, axis=(1, 2))  # tr(L L^T) per block
    quad = jnp.sum(mu_block**2, axis=1)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)), axis=1)
    block_kl = 0.5 * jnp.sum(trace + quad - block_size - logdet)
    return diag_kl + block_kl


def block_marginal_log_std(
    log_std: ParamsTree,
    block_raw: jnp.ndarray,
    block_index_map: jnp.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return per-latent marginal log-std for a block-diagonal Gaussian.

    Grouped latents take the diagonal of their block covariance ``L L^T`` (the
    row-sum of squared Cholesky entries); ungrouped latents keep ``log_std``.

    Args:
        log_std (ParamsTree): Diagonal log-std pytree.
        block_raw (jnp.ndarray): Raw block parameters ``(G, k, k)``.
        block_index_map (jnp.ndarray): Index map ``(G, k)`` into the raveled latent.

    Returns:
        dict[str, dict[str, Any]]: Marginal log-std pytree matching ``log_std``.
    """
    flat_log_std, unravel = ravel_pytree(log_std)
    variance = jnp.exp(2.0 * flat_log_std)
    chol = build_block_cholesky(block_raw)  # (G, k, k)
    block_var = jnp.sum(chol**2, axis=2)  # diag of L L^T per block, (G, k)
    variance = variance.at[block_index_map.reshape(-1)].set(block_var.reshape(-1))
    return unravel(0.5 * jnp.log(variance))
