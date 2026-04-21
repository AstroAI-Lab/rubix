import jax.numpy as jnp
import pytest

from rubix.inference import build_loss_from_config, build_loss_from_user_config


def test_build_loss_from_config_single_mse_matches_expected():
    cfg = {"kind": "mse", "normalize": True}
    loss_fn = build_loss_from_config(cfg)

    pred = jnp.array([[[2.0, 0.0]]])
    target = jnp.array([[[1.0, 2.0]]])
    value = loss_fn(pred, target)

    expected = (1.0**2 + (-2.0) ** 2) / 2.0
    assert jnp.allclose(value, expected)


def test_build_loss_from_config_combined_with_tensor_keys():
    cfg = {
        "kind": "combined",
        "terms": [
            {
                "kind": "gaussian_nll",
                "inv_variance_key": "ivar",
                "mask_key": "mask",
                "normalize": True,
            },
            {
                "kind": "huber",
                "delta": 1.0,
                "mask_key": "mask",
                "weight": 0.25,
                "normalize": True,
            },
        ],
    }

    tensors = {
        "ivar": jnp.array([[[4.0, 1.0]]]),
        "mask": jnp.array([[[1.0, 0.0]]]),
    }

    loss_fn = build_loss_from_config(cfg, tensors=tensors)
    pred = jnp.array([[[2.0, 0.0]]])
    target = jnp.array([[[1.0, 2.0]]])

    value = loss_fn(pred, target)

    gaussian = 0.5 * (4.0 - jnp.log(4.0))
    huber = 0.5
    expected = gaussian + 0.25 * huber
    assert jnp.allclose(value, expected)


def test_build_loss_from_config_rejects_missing_tensor_key():
    cfg = {"kind": "mse", "mask_key": "missing"}
    with pytest.raises(ValueError, match="mask key 'missing' not found"):
        build_loss_from_config(cfg, tensors={})


def test_build_loss_from_user_config_absent_objective_returns_none():
    cfg = {"pipeline": {"name": "calc_gradient"}}
    assert build_loss_from_user_config(cfg) is None


def test_build_loss_from_config_rejects_weights_length_mismatch():
    cfg = {
        "kind": "combined",
        "terms": [{"kind": "mse"}, {"kind": "huber"}],
        "weights": [0.5],
    }
    with pytest.raises(ValueError, match="'weights' length"):
        build_loss_from_config(cfg)


def test_build_loss_from_user_config_resolves_objective():
    cfg = {
        "inference": {
            "objective": {
                "kind": "huber",
                "delta": 0.5,
                "normalize": False,
            }
        }
    }
    loss_fn = build_loss_from_user_config(cfg)
    assert loss_fn is not None

    pred = jnp.array([[[3.0]]])
    target = jnp.array([[[1.0]]])
    value = loss_fn(pred, target)
    # delta=0.5, |r|=2 => 0.5*(2*delta - delta^2)=delta*(|r|-delta/2)=0.875
    assert jnp.allclose(value, 0.875)
