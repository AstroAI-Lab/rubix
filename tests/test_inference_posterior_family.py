import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from rubix.inference.posterior_family import (
    init_low_rank_factor,
    kl_low_rank_to_standard_normal,
    low_rank_marginal_log_std,
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
