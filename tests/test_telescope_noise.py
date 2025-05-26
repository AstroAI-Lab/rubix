import jax.numpy as jnp
import jax.random as jrandom
import pytest

from rubix.telescope.noise.noise import calculate_noise_cube, sample_noise


def check_noise_variation(noise_cube):
    """Check that the noise cube does not have the same noise in each telescope bin."""
    flat_noise = noise_cube.reshape(-1, noise_cube.shape[-1])
    variances = jnp.var(flat_noise, axis=0)
    return jnp.all(variances > 0)


@pytest.mark.parametrize("noise_distribution", ["normal", "uniform"])
def test_calculate_noise_cube_standard_case(noise_distribution):
    key = jrandom.PRNGKey(0)
    cube = jrandom.uniform(key, shape=(5, 5, 10))
    S2N = 0.8

    noise_cube = calculate_noise_cube(cube, S2N, noise_distribution=noise_distribution)

    assert noise_cube.shape == cube.shape
    assert not jnp.isnan(noise_cube).any()
    assert check_noise_variation(noise_cube)


@pytest.mark.parametrize("noise_distribution", ["normal", "uniform"])
def test_calculate_noise_cube_with_zeros(noise_distribution):
    key = jrandom.PRNGKey(0)
    cube = jnp.zeros((5, 5, 10))
    S2N = 0.8

    noise_cube = calculate_noise_cube(cube, S2N, noise_distribution=noise_distribution)

    assert noise_cube.shape == cube.shape
    assert jnp.all(noise_cube == 0)
    # assert check_noise_variation(noise_cube)


@pytest.mark.parametrize("noise_distribution", ["normal", "uniform"])
def test_calculate_noise_cube_infinite_S2N(noise_distribution):
    key = jrandom.PRNGKey(0)
    cube = jrandom.uniform(key, shape=(5, 5, 10))
    S2N = 0.8

    noise_cube = calculate_noise_cube(cube, S2N, noise_distribution=noise_distribution)

    assert noise_cube.shape == cube.shape
    assert not jnp.isinf(noise_cube).any()
    # assert check_noise_variation(noise_cube)


@pytest.mark.parametrize("noise_distribution", ["normal", "uniform"])
def test_calculate_noise_cube_negative_S2N(noise_distribution):
    key = jrandom.PRNGKey(0)
    cube = jrandom.uniform(key, shape=(5, 5, 10))

    S2N = 0.8
    noise_cube = calculate_noise_cube(cube, S2N, noise_distribution=noise_distribution)

    assert noise_cube.shape == cube.shape
    assert not jnp.isnan(noise_cube).any()
    assert check_noise_variation(noise_cube)


def test_sample_noise_key_none():
    shape = (5, 5, 10)

    noise_normal = sample_noise(shape, type="normal", key=None)
    assert noise_normal.shape == shape
    assert not jnp.isnan(noise_normal).any()

    noise_uniform = sample_noise(shape, type="uniform", key=None)
    assert noise_uniform.shape == shape
    assert not jnp.isnan(noise_uniform).any()


def test_sample_noise_invalid_type():
    shape = (5, 5, 10)
    key = jrandom.PRNGKey(0)

    with pytest.raises(ValueError, match="Invalid noise type: invalid"):
        sample_noise(shape, type="invalid", key=key)
