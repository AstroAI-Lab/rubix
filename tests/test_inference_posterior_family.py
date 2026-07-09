import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from rubix.inference.posterior_family import (
    build_block_cholesky,
    build_particle_block_index_map,
    init_block_cholesky,
    init_low_rank_factor,
    kl_block_to_standard_normal,
    kl_low_rank_to_standard_normal,
    low_rank_marginal_log_std,
    sample_block_gaussian,
    sample_low_rank_gaussian,
)
from rubix.inference.variational import kl_diag_gaussian_to_standard_normal


def _example_tree():
    mean = {"stars": {"age": jnp.array([1.0, -0.5]), "metallicity": jnp.array([0.3])}}
    log_std = {
        "stars": {"age": jnp.array([-0.7, 0.1]), "metallicity": jnp.array([-1.0])}
    }
    return mean, log_std


def _dense_kl(mean, log_std, factor):
    """Reference KL(q || N(0, I)) built from the dense covariance."""
    flat_mean, _ = ravel_pytree(mean)
    flat_log_std, _ = ravel_pytree(log_std)
    n = flat_mean.shape[0]
    sigma = jnp.diag(jnp.exp(2.0 * flat_log_std)) + factor @ factor.T
    logdet = jnp.linalg.slogdet(sigma)[1]
    return 0.5 * (jnp.trace(sigma) + flat_mean @ flat_mean - n - logdet)


def test_kl_reduces_to_diagonal_when_factor_zero():
    mean, log_std = _example_tree()
    flat_mean, _ = ravel_pytree(mean)
    factor = jnp.zeros((flat_mean.shape[0], 2))
    low_rank = kl_low_rank_to_standard_normal(mean, log_std, factor)
    diagonal = kl_diag_gaussian_to_standard_normal(mean, log_std)
    assert float(low_rank) == pytest.approx(float(diagonal), rel=1e-5, abs=1e-6)


def test_kl_matches_dense_reference():
    mean, log_std = _example_tree()
    flat_mean, _ = ravel_pytree(mean)
    key = jax.random.PRNGKey(0)
    factor = 0.4 * jax.random.normal(key, (flat_mean.shape[0], 2))
    lemma = kl_low_rank_to_standard_normal(mean, log_std, factor)
    dense = _dense_kl(mean, log_std, factor)
    assert float(lemma) == pytest.approx(float(dense), rel=1e-5, abs=1e-6)


def test_sampling_recovers_low_rank_covariance():
    mean, log_std = _example_tree()
    flat_mean, _ = ravel_pytree(mean)
    n = flat_mean.shape[0]
    key = jax.random.PRNGKey(1)
    factor = 0.5 * jax.random.normal(key, (n, 2))
    expected_cov = jnp.diag(jnp.exp(2.0 * ravel_pytree(log_std)[0])) + factor @ factor.T

    keys = jax.random.split(jax.random.PRNGKey(2), 40000)
    samples = jax.vmap(
        lambda k: ravel_pytree(sample_low_rank_gaussian(mean, log_std, factor, k))[0]
    )(keys)

    emp_mean = jnp.mean(samples, axis=0)
    emp_cov = jnp.cov(samples, rowvar=False)
    assert jnp.allclose(emp_mean, flat_mean, atol=0.03)
    assert jnp.allclose(emp_cov, expected_cov, atol=0.05)


def test_marginal_log_std_matches_definition():
    mean, log_std = _example_tree()
    flat_log_std, _ = ravel_pytree(log_std)
    n = flat_log_std.shape[0]
    factor = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2) * 0.1
    marginal = low_rank_marginal_log_std(log_std, factor)
    flat_marginal, _ = ravel_pytree(marginal)
    expected_var = jnp.exp(2.0 * flat_log_std) + jnp.sum(factor**2, axis=1)
    assert jnp.allclose(flat_marginal, 0.5 * jnp.log(expected_var), atol=1e-6)
    # Marginal std is always at least the diagonal std.
    assert bool(jnp.all(flat_marginal >= flat_log_std - 1e-6))


def test_init_low_rank_factor_shape_and_validation():
    mean, _ = _example_tree()
    flat_mean, _ = ravel_pytree(mean)
    factor = init_low_rank_factor(mean, rank=3, key=jax.random.PRNGKey(0))
    assert factor.shape == (flat_mean.shape[0], 3)
    with pytest.raises(ValueError, match="rank must be strictly positive"):
        init_low_rank_factor(mean, rank=0, key=jax.random.PRNGKey(0))


def _dense_block_kl(mean, log_std, block_raw, block_index_map):
    """Reference block KL built from the full dense covariance."""
    flat_mean, _ = ravel_pytree(mean)
    flat_log_std, _ = ravel_pytree(log_std)
    n = flat_mean.shape[0]
    cov = jnp.diag(jnp.exp(2.0 * flat_log_std))
    chol = build_block_cholesky(block_raw)
    for g in range(block_index_map.shape[0]):
        idx = block_index_map[g]
        block_cov = chol[g] @ chol[g].T
        for a in range(idx.shape[0]):
            for b in range(idx.shape[0]):
                cov = cov.at[idx[a], idx[b]].set(block_cov[a, b])
    logdet = jnp.linalg.slogdet(cov)[1]
    return 0.5 * (jnp.trace(cov) + flat_mean @ flat_mean - n - logdet)


def test_build_particle_block_index_map_groups_fields_per_particle():
    mean = {
        "stars": {
            "age": jnp.zeros((3,)),
            "metallicity": jnp.zeros((3,)),
            "velocity": jnp.zeros((3, 1)),
        }
    }
    idx = build_particle_block_index_map(
        mean, [("stars", "age"), ("stars", "metallicity"), ("stars", "velocity")]
    )
    # Ravel order is field-major: age[0..2], metallicity[3..5], velocity[6..8].
    assert idx.shape == (3, 3)
    assert idx.tolist() == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]


def test_block_kl_reduces_to_diagonal_when_offdiagonal_zero():
    mean = {
        "stars": {"age": jnp.array([0.5, -0.3]), "metallicity": jnp.array([1.0, 0.2])}
    }
    log_std = {
        "stars": {"age": jnp.array([-0.5, 0.2]), "metallicity": jnp.array([0.1, -0.4])}
    }
    idx = build_particle_block_index_map(
        mean, [("stars", "age"), ("stars", "metallicity")]
    )
    # Block raw with diagonal = the grouped latents' log_std, no off-diagonal.
    flat_log_std, _ = ravel_pytree(log_std)
    block_raw = jnp.zeros((2, 2, 2))
    for g in range(2):
        block_raw = block_raw.at[g, 0, 0].set(flat_log_std[idx[g, 0]])
        block_raw = block_raw.at[g, 1, 1].set(flat_log_std[idx[g, 1]])
    block = kl_block_to_standard_normal(mean, log_std, block_raw, idx)
    diagonal = kl_diag_gaussian_to_standard_normal(mean, log_std)
    assert float(block) == pytest.approx(float(diagonal), rel=1e-5, abs=1e-6)


def test_block_kl_matches_dense_reference():
    mean = {
        "stars": {"age": jnp.array([0.5, -0.3]), "metallicity": jnp.array([1.0, 0.2])}
    }
    log_std = {
        "stars": {"age": jnp.array([-0.5, 0.2]), "metallicity": jnp.array([0.1, -0.4])}
    }
    idx = build_particle_block_index_map(
        mean, [("stars", "age"), ("stars", "metallicity")]
    )
    key = jax.random.PRNGKey(0)
    block_raw = init_block_cholesky(2, 2, key, init_log_std=-0.3, offdiag_scale=0.5)
    lemma = kl_block_to_standard_normal(mean, log_std, block_raw, idx)
    dense = _dense_block_kl(mean, log_std, block_raw, idx)
    assert float(lemma) == pytest.approx(float(dense), rel=1e-5, abs=1e-6)


def test_block_sampling_recovers_block_covariance():
    mean = {
        "stars": {"age": jnp.array([0.0, 0.0]), "metallicity": jnp.array([0.0, 0.0])}
    }
    log_std = {
        "stars": {
            "age": jnp.array([-3.0, -3.0]),
            "metallicity": jnp.array([-3.0, -3.0]),
        }
    }
    idx = build_particle_block_index_map(
        mean, [("stars", "age"), ("stars", "metallicity")]
    )
    key = jax.random.PRNGKey(0)
    block_raw = init_block_cholesky(2, 2, key, init_log_std=-0.2, offdiag_scale=0.6)
    chol = build_block_cholesky(block_raw)

    keys = jax.random.split(jax.random.PRNGKey(3), 40000)
    samples = jax.vmap(
        lambda k: ravel_pytree(sample_block_gaussian(mean, log_std, block_raw, idx, k))[
            0
        ]
    )(keys)
    emp_cov = jnp.cov(samples, rowvar=False)
    # Block 0 couples flat indices (0, 2) -> its 2x2 covariance is chol0 chol0^T.
    expected0 = chol[0] @ chol[0].T
    got0 = jnp.array([[emp_cov[0, 0], emp_cov[0, 2]], [emp_cov[2, 0], emp_cov[2, 2]]])
    assert jnp.allclose(got0, expected0, atol=0.05)
    # Cross-block correlation (index 0 vs index 1) must stay ~0.
    assert abs(float(emp_cov[0, 1])) < 0.03


def test_low_rank_and_block_kl_reduce_to_diagonal_prior_std():
    # With zero factor / zero off-diagonals, low-rank and block KL must equal the
    # diagonal KL for the SAME prior_std (not just prior_std=1).
    mean = {
        "stars": {"age": jnp.array([0.4, -0.2]), "metallicity": jnp.array([0.7, 0.1])}
    }
    log_std = {
        "stars": {"age": jnp.array([-0.3, 0.2]), "metallicity": jnp.array([0.0, -0.5])}
    }
    tau = 1.814
    flat_mean, _ = ravel_pytree(mean)

    lr = kl_low_rank_to_standard_normal(
        mean, log_std, jnp.zeros((flat_mean.shape[0], 2)), prior_std=tau
    )
    diag = kl_diag_gaussian_to_standard_normal(mean, log_std, prior_std=tau)
    assert float(lr) == pytest.approx(float(diag), rel=1e-5, abs=1e-6)

    idx = build_particle_block_index_map(
        mean, [("stars", "age"), ("stars", "metallicity")]
    )
    flat_log_std, _ = ravel_pytree(log_std)
    block_raw = jnp.zeros((2, 2, 2))
    for g in range(2):
        block_raw = block_raw.at[g, 0, 0].set(flat_log_std[idx[g, 0]])
        block_raw = block_raw.at[g, 1, 1].set(flat_log_std[idx[g, 1]])
    block = kl_block_to_standard_normal(mean, log_std, block_raw, idx, prior_std=tau)
    assert float(block) == pytest.approx(float(diag), rel=1e-5, abs=1e-6)
