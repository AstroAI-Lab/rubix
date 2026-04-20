import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import apply_params, forward, loss, value_and_grad


class DummyPipeline:
    """Simple differentiable forward model for inference API tests."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        age_sum = jnp.sum(rubixdata.stars.age)
        metallicity_sum = jnp.sum(rubixdata.stars.metallicity)
        value = age_sum + 2.0 * metallicity_sum
        return jnp.reshape(value, (1, 1, 1))


class DummyPipelineWithNoiseCapture:
    """Dummy pipeline capturing the propagated noise key."""

    def __init__(self):
        self.seen_noise_key = None

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        self.seen_noise_key = rubixdata.noise_key
        return jnp.zeros((1, 1, 1))


def _make_rubix_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((2, 3)),
            velocity=jnp.zeros((2, 3)),
            mass=jnp.ones(2),
            age=jnp.array([1.0, 1.0]),
            metallicity=jnp.array([0.1, 0.1]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def test_apply_params_returns_new_rubix_data():
    static_data = _make_rubix_data()
    params = {
        "stars": {"age": jnp.array([2.0, 3.0]), "metallicity": jnp.array([0.2, 0.3])}
    }

    updated = apply_params(static_data, params)

    assert jnp.allclose(updated.stars.age, jnp.array([2.0, 3.0]))
    assert jnp.allclose(updated.stars.metallicity, jnp.array([0.2, 0.3]))
    assert jnp.allclose(static_data.stars.age, jnp.array([1.0, 1.0]))
    assert jnp.allclose(static_data.stars.metallicity, jnp.array([0.1, 0.1]))


def test_apply_params_raises_for_unknown_field():
    static_data = _make_rubix_data()
    params = {"stars": {"not_a_field": jnp.array([1.0, 2.0])}}

    with pytest.raises(ValueError, match="Unknown fields for 'stars'"):
        apply_params(static_data, params)


def test_loss_is_deterministic_for_same_inputs():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params = {
        "stars": {"age": jnp.array([2.0, 3.0]), "metallicity": jnp.array([0.2, 0.3])}
    }
    target = jnp.array([[[4.0]]])

    loss_1 = loss(pipeline, params, static_data, target)
    loss_2 = loss(pipeline, params, static_data, target)

    assert jnp.allclose(loss_1, loss_2)


def test_value_and_grad_matches_analytic_gradient():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params = {
        "stars": {"age": jnp.array([2.0, 3.0]), "metallicity": jnp.array([0.2, 0.3])}
    }
    target = jnp.array([[[4.0]]])

    value, grads = value_and_grad(pipeline, params, static_data, target)

    pred = (2.0 + 3.0) + 2.0 * (0.2 + 0.3)
    residual = pred - 4.0
    expected_grad_age = jnp.array([2.0 * residual, 2.0 * residual])
    expected_grad_metallicity = jnp.array([4.0 * residual, 4.0 * residual])

    assert jnp.allclose(value, residual**2)
    assert jnp.allclose(grads["stars"]["age"], expected_grad_age)
    assert jnp.allclose(grads["stars"]["metallicity"], expected_grad_metallicity)


def test_forward_propagates_noise_key():
    pipeline = DummyPipelineWithNoiseCapture()
    static_data = _make_rubix_data()
    params = {"stars": {"age": jnp.array([1.0, 1.0])}}
    noise_key = jnp.array([17, 42], dtype=jnp.uint32)

    _ = forward(
        pipeline=pipeline,
        params=params,
        static_data=static_data,
        noise_key=noise_key,
    )

    assert pipeline.seen_noise_key is not None
    assert jnp.array_equal(pipeline.seen_noise_key, noise_key)
