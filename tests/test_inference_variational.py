import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    build_age_metallicity_transforms,
    initialize_mean_field_params,
    kl_diag_gaussian_to_standard_normal,
    optimize_variational_posterior,
    sample_diag_gaussian,
)


class DummyPipeline:
    """Simple differentiable pipeline used for VI tests."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        age_sum = jnp.sum(rubixdata.stars.age)
        metallicity_sum = jnp.sum(rubixdata.stars.metallicity)
        value = age_sum + 2.0 * metallicity_sum
        return jnp.reshape(value, (1, 1, 1))


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


def test_kl_diag_gaussian_is_zero_for_standard_normal():
    mean = {"stars": {"age": jnp.array([0.0]), "metallicity": jnp.array([0.0])}}
    log_std = {"stars": {"age": jnp.array([0.0]), "metallicity": jnp.array([0.0])}}

    kl = kl_diag_gaussian_to_standard_normal(mean, log_std)

    assert jnp.allclose(kl, 0.0)


def test_initialize_and_sample_mean_field_params():
    params_init = {
        "stars": {
            "age": jnp.array([1.5]),
            "metallicity": jnp.array([0.02]),
        }
    }
    mean, log_std = initialize_mean_field_params(params_init, init_log_std=-1.5)
    sample = sample_diag_gaussian(
        mean,
        log_std,
        key=jnp.array([0, 7], dtype=jnp.uint32),
    )

    assert jnp.allclose(mean["stars"]["age"], params_init["stars"]["age"])
    assert jnp.allclose(
        mean["stars"]["metallicity"],
        params_init["stars"]["metallicity"],
    )
    assert jnp.allclose(log_std["stars"]["age"], jnp.array([-1.5]))
    assert jnp.allclose(log_std["stars"]["metallicity"], jnp.array([-1.5]))
    assert sample["stars"]["age"].shape == params_init["stars"]["age"].shape
    assert (
        sample["stars"]["metallicity"].shape
        == params_init["stars"]["metallicity"].shape
    )


def test_optimize_variational_posterior_improves_objective():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {
            "age": jnp.array([0.5]),
            "metallicity": jnp.array([0.001]),
        }
    }
    target = jnp.array([[[5.0]]])

    result = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=5e-2,
        max_steps=200,
        tol=1e-9,
        num_samples=4,
        beta_kl=1e-4,
        seed=11,
    )

    assert result.objective_history[0] > result.objective_history[-1]
    assert len(result.reconstruction_history) == len(result.objective_history)
    assert len(result.kl_history) == len(result.objective_history)
    assert result.steps_run <= 200


def test_optimize_variational_with_transforms_respects_bounds():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {
            "age": jnp.array([0.5]),
            "metallicity": jnp.array([0.001]),
        }
    }
    transforms = build_age_metallicity_transforms(
        age_lower=0.0,
        age_upper=20.0,
        metallicity_lower=0.0,
        metallicity_upper=0.05,
    )

    result = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=jnp.array([[[7.04]]]),
        learning_rate=5e-2,
        max_steps=200,
        tol=1e-9,
        num_samples=4,
        beta_kl=1e-4,
        transforms=transforms,
        seed=13,
    )

    mean_age = result.posterior_mean_constrained_params["stars"]["age"]
    mean_metallicity = result.posterior_mean_constrained_params["stars"]["metallicity"]

    assert jnp.all(mean_age > 0.0)
    assert jnp.all(mean_age < 20.0)
    assert jnp.all(mean_metallicity > 0.0)
    assert jnp.all(mean_metallicity < 0.05)


def test_optimize_variational_rejects_non_positive_num_samples():
    with pytest.raises(ValueError, match="num_samples must be strictly positive"):
        optimize_variational_posterior(
            pipeline=DummyPipeline(),
            params_init={
                "stars": {
                    "age": jnp.array([0.5]),
                    "metallicity": jnp.array([0.001]),
                }
            },
            static_data=_make_rubix_data(),
            target=jnp.array([[[5.0]]]),
            num_samples=0,
        )
