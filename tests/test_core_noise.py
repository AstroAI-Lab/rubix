import jax.numpy as jnp
import pytest

from rubix.core.noise import build_post_aggregation_noise_fn, get_apply_noise


def test_no_noise_in_config():
    config = {"telescope": {}}
    with pytest.raises(
        ValueError, match="Noise information not provided in telescope config"
    ):
        get_apply_noise(config)


def test_no_signal_to_noise_in_noise_config():
    config = {"telescope": {"noise": {}}}
    with pytest.raises(
        ValueError, match="Signal to noise information not provided in noise config"
    ):
        get_apply_noise(config)


def test_no_noise_distribution_in_noise_config():
    config = {"telescope": {"noise": {"signal_to_noise": 10}}}
    with pytest.raises(ValueError):
        get_apply_noise(config)


def test_build_post_aggregation_noise_fn_requires_config_fields():
    with pytest.raises(ValueError, match="Noise information not provided"):
        build_post_aggregation_noise_fn({"telescope": {}})


def test_build_post_aggregation_noise_fn_adds_noise_with_key():
    config = {
        "telescope": {
            "noise": {
                "signal_to_noise": 2.0,
                "noise_distribution": "normal",
            }
        }
    }
    fn = build_post_aggregation_noise_fn(config)
    cube = jnp.ones((2, 2, 4))
    key = jnp.array([7, 11], dtype=jnp.uint32)
    noisy_1 = fn(cube, key)
    noisy_2 = fn(cube, key)

    assert noisy_1.shape == cube.shape
    assert jnp.allclose(noisy_1, noisy_2)
