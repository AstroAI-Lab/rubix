import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import combine_loss_fns, huber_data_loss, masked_gaussian_nll
from rubix.inference.optimize import optimize_params


class DummyPipeline:
    """Simple differentiable pipeline used for inference loss tests."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        value = jnp.sum(rubixdata.stars.age) + 2.0 * jnp.sum(
            rubixdata.stars.metallicity
        )
        return jnp.reshape(value, (1, 1, 1))


def _make_rubix_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
            age=jnp.array([1.0]),
            metallicity=jnp.array([0.1]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def test_masked_gaussian_nll_matches_expected_value():
    prediction = jnp.array([[[2.0, 1.0]]])
    target = jnp.array([[[1.0, 1.0]]])
    inv_variance = jnp.array([[[4.0, 1.0]]])
    mask = jnp.array([[[1.0, 0.0]]])

    value = masked_gaussian_nll(
        prediction=prediction,
        target=target,
        inv_variance=inv_variance,
        mask=mask,
        normalize=True,
    )

    # Active term: 0.5 * (1^2 * 4 - log(4))
    expected = 0.5 * (4.0 - jnp.log(4.0))
    assert jnp.allclose(value, expected)


def test_masked_gaussian_nll_rejects_ambiguous_uncertainty_inputs():
    prediction = jnp.ones((1, 1, 1))
    target = jnp.ones((1, 1, 1))
    sigma = jnp.ones((1, 1, 1))
    inv_variance = jnp.ones((1, 1, 1))

    with pytest.raises(ValueError, match="provide only one of sigma or inv_variance"):
        _ = masked_gaussian_nll(
            prediction=prediction,
            target=target,
            sigma=sigma,
            inv_variance=inv_variance,
        )


def test_huber_data_loss_rejects_non_positive_delta():
    prediction = jnp.ones((1, 1, 1))
    target = jnp.ones((1, 1, 1))

    with pytest.raises(ValueError, match="delta must be strictly positive"):
        huber_data_loss(prediction, target, delta=0.0)

    with pytest.raises(ValueError, match="delta must be strictly positive"):
        huber_data_loss(prediction, target, delta=-1.0)


def test_huber_data_loss_reduces_outlier_sensitivity_vs_quadratic():
    prediction = jnp.array([[[10.0]]])
    target = jnp.array([[[0.0]]])

    huber_value = huber_data_loss(prediction, target, delta=1.0, normalize=False)
    quadratic = 0.5 * (10.0**2)

    assert huber_value < quadratic
    assert jnp.allclose(huber_value, 9.5)


def test_combine_loss_fns_weighted_sum_matches_manual():
    prediction = jnp.array([[[3.0]]])
    target = jnp.array([[[1.0]]])

    l1 = lambda p, t: jnp.sum((p - t) ** 2)
    l2 = lambda p, t: jnp.sum(jnp.abs(p - t))
    combined = combine_loss_fns([l1, l2], weights=[0.5, 2.0])

    expected = 0.5 * 4.0 + 2.0 * 2.0
    assert jnp.allclose(combined(prediction, target), expected)


def test_composed_loss_works_with_optimize_params():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.0]), "metallicity": jnp.array([0.0])}}
    target = jnp.array([[[5.0]]])

    gaussian_term = lambda p, t: masked_gaussian_nll(
        p,
        t,
        sigma=jnp.ones_like(t),
        normalize=False,
    )
    robust_term = lambda p, t: huber_data_loss(p, t, delta=0.5, normalize=False)
    loss_fn = combine_loss_fns([gaussian_term, robust_term], weights=[1.0, 0.1])

    result = optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        loss_fn=loss_fn,
        learning_rate=0.1,
        max_steps=120,
        tol=1e-8,
    )

    assert result.loss_history[0] > result.loss_history[-1]
    assert jnp.allclose(result.best_loss, jnp.min(result.loss_history))
    assert result.final_loss < result.loss_history[0]
