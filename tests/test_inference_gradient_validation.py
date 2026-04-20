import jax.numpy as jnp
import jax.random as jrandom
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    compare_gradients,
    finite_difference_grad,
    loss,
    value_and_grad,
)


class DeterministicDummyPipeline:
    """Simple deterministic forward model."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        age_sum = jnp.sum(rubixdata.stars.age)
        metallicity_sum = jnp.sum(rubixdata.stars.metallicity)
        prediction = age_sum + 2.0 * metallicity_sum
        return jnp.reshape(prediction, (1, 1, 1))


class StochasticDummyPipeline:
    """Forward model where outputs depend on a fixed PRNG key."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        age_sum = jnp.sum(rubixdata.stars.age)
        metallicity_sum = jnp.sum(rubixdata.stars.metallicity)
        base = age_sum + 2.0 * metallicity_sum

        key = rubixdata.noise_key
        if key is None:
            key = jrandom.PRNGKey(0)

        noise = 0.1 * jrandom.normal(key, ())
        prediction = base * (1.0 + noise)
        return jnp.reshape(prediction, (1, 1, 1))


def _make_rubix_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((2, 3)),
            velocity=jnp.zeros((2, 3)),
            mass=jnp.ones(2),
            age=jnp.array([1.0, 1.0]),
            metallicity=jnp.array([0.01, 0.02]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def _params_init():
    return {
        "stars": {
            "age": jnp.array([1.5, 2.0]),
            "metallicity": jnp.array([0.015, 0.025]),
        }
    }


def test_finite_difference_grad_matches_autodiff_deterministic():
    pipeline = DeterministicDummyPipeline()
    static_data = _make_rubix_data()
    params = _params_init()
    target = jnp.array([[[4.5]]])

    def loss_fn(current_params):
        return loss(
            pipeline=pipeline,
            params=current_params,
            static_data=static_data,
            target=target,
        )

    autodiff_grad = value_and_grad(
        pipeline=pipeline,
        params=params,
        static_data=static_data,
        target=target,
    )[1]
    fd_grad = finite_difference_grad(loss_fn, params, eps=1e-4)
    comparison = compare_gradients(autodiff_grad, fd_grad)

    assert comparison.max_abs_error < 1e-2
    assert comparison.relative_l2_error < 1e-3


def test_finite_difference_grad_matches_autodiff_fixed_stochastic_key():
    pipeline = StochasticDummyPipeline()
    static_data = _make_rubix_data()
    params = _params_init()
    target = jnp.array([[[4.5]]])
    noise_key = jrandom.PRNGKey(17)

    def loss_fn(current_params):
        return loss(
            pipeline=pipeline,
            params=current_params,
            static_data=static_data,
            target=target,
            noise_key=noise_key,
        )

    autodiff_grad = value_and_grad(
        pipeline=pipeline,
        params=params,
        static_data=static_data,
        target=target,
        noise_key=noise_key,
    )[1]
    fd_grad = finite_difference_grad(loss_fn, params, eps=1e-4)
    comparison = compare_gradients(autodiff_grad, fd_grad)

    assert comparison.max_abs_error < 1e-2
    assert comparison.relative_l2_error < 1e-3


def test_compare_gradients_is_zero_for_identical_gradients():
    gradient = {
        "stars": {
            "age": jnp.array([1.0, 2.0]),
            "metallicity": jnp.array([3.0]),
        }
    }

    comparison = compare_gradients(gradient, gradient)

    assert comparison.max_abs_error == 0.0
    assert comparison.l2_error == 0.0
    assert comparison.relative_l2_error == 0.0


def test_finite_difference_grad_rejects_non_positive_eps():
    params = _params_init()

    with pytest.raises(ValueError, match="eps must be strictly positive"):
        finite_difference_grad(lambda p: jnp.sum(p["stars"]["age"]), params, eps=0.0)


def test_finite_difference_grad_rejects_non_scalar_loss_fn():
    params = _params_init()

    with pytest.raises(ValueError, match="loss_fn must return a scalar"):
        finite_difference_grad(
            lambda p: jnp.array([jnp.sum(p["stars"]["age"])]),
            params,
            eps=1e-4,
        )


def test_compare_gradients_rejects_mismatched_shapes():
    auto = {"stars": {"age": jnp.array([1.0, 2.0])}}
    fd = {"stars": {"age": jnp.array([1.0])}}

    with pytest.raises(ValueError, match="must flatten to the same shape"):
        compare_gradients(auto, fd)
