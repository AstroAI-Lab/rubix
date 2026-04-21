import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    compute_residual_products,
    sample_posterior_predictive_cubes,
    summarize_masked_metrics,
    summarize_predictive_cube_samples,
)


class CubeScalePipeline:
    """Pipeline that scales a fixed cube template by stars.age[0]."""

    def __init__(self, template: jnp.ndarray):
        self.template = template

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        scale = rubixdata.stars.age[0]
        return scale * self.template


def _make_rubix_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
            age=jnp.array([1.0]),
            metallicity=jnp.array([0.01]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def test_sample_posterior_predictive_cubes_shape():
    pipeline = CubeScalePipeline(jnp.ones((2, 2, 3), dtype=jnp.float32))
    static_data = _make_rubix_data()
    mean = {"stars": {"age": jnp.array([1.5])}}
    log_std = {"stars": {"age": jnp.array([-3.0])}}

    samples = sample_posterior_predictive_cubes(
        pipeline=pipeline,
        posterior_mean_params=mean,
        posterior_log_std_params=log_std,
        static_data=static_data,
        num_samples=5,
        seed=7,
    )

    assert samples.shape == (5, 2, 2, 3)


def test_summarize_predictive_cube_samples_outputs_expected_keys():
    samples = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 2, 2)
    summary = summarize_predictive_cube_samples(samples)

    assert set(summary.keys()) == {"mean", "std", "p16", "p50", "p84"}
    assert summary["mean"].shape == (2, 2, 2)


def test_compute_residual_products_with_sigma_and_mask():
    prediction = jnp.array([[[2.0, 1.0]]])
    target = jnp.array([[[1.0, 2.0]]])
    sigma = jnp.array([[[0.5, 1.0]]])
    mask = jnp.array([[[1.0, 0.0]]])

    products = compute_residual_products(
        prediction=prediction,
        target=target,
        sigma=sigma,
        mask=mask,
    )

    assert set(products.keys()) == {
        "residual",
        "abs_residual",
        "chi2",
        "masked_residual",
        "masked_chi2",
    }
    assert jnp.allclose(products["chi2"], jnp.array([[[4.0, 1.0]]]))
    assert jnp.allclose(products["masked_chi2"], jnp.array([[[4.0, 0.0]]]))


def test_summarize_masked_metrics_matches_expected_values():
    prediction = jnp.array([[[2.0, 0.0]]])
    target = jnp.array([[[1.0, 2.0]]])
    mask = jnp.array([[[1.0, 0.0]]])

    metrics = summarize_masked_metrics(prediction, target, mask=mask)

    assert pytest.approx(metrics["mse"]) == 1.0
    assert pytest.approx(metrics["mae"]) == 1.0
