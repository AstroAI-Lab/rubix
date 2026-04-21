import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    build_age_metallicity_transforms,
    initialize_mean_field_params,
    kl_diag_gaussian_to_standard_normal,
    optimize_variational_ifu_cube,
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
    assert len(result.grad_norm_history) == len(result.objective_history)
    assert len(result.update_norm_history) == len(result.objective_history)
    assert result.best_step >= 0
    assert result.best_step < len(result.objective_history)
    assert result.best_objective == min(result.objective_history)
    assert jnp.isfinite(result.final_objective)
    assert jnp.isfinite(result.final_reconstruction)
    assert jnp.isfinite(result.final_kl)
    assert result.final_objective >= result.best_objective
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


def test_optimize_variational_ifu_cube_rejects_non_3d_target():
    with pytest.raises(ValueError, match="target must be a 3D IFU datacube"):
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init={
                "stars": {
                    "age": jnp.array([0.5]),
                    "metallicity": jnp.array([0.001]),
                }
            },
            static_data=_make_rubix_data(),
            target=jnp.ones((2, 2)),
        )


def test_optimize_variational_ifu_cube_rejects_invalid_huber_settings():
    with pytest.raises(ValueError, match="huber_weight must be non-negative"):
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init={
                "stars": {
                    "age": jnp.array([0.5]),
                    "metallicity": jnp.array([0.001]),
                }
            },
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 1, 1)),
            huber_weight=-0.1,
        )

    with pytest.raises(
        ValueError, match="huber_delta must be provided when huber_weight > 0"
    ):
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init={
                "stars": {
                    "age": jnp.array([0.5]),
                    "metallicity": jnp.array([0.001]),
                }
            },
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 1, 1)),
            huber_weight=0.2,
        )


def test_optimize_variational_ifu_cube_improves_objective():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {
            "age": jnp.array([0.5]),
            "metallicity": jnp.array([0.001]),
        }
    }
    target = jnp.array([[[5.0]]])
    sigma = jnp.ones_like(target) * 0.5
    mask = jnp.ones_like(target)

    result = optimize_variational_ifu_cube(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        sigma=sigma,
        mask=mask,
        huber_delta=0.2,
        huber_weight=0.1,
        learning_rate=5e-2,
        max_steps=120,
        tol=1e-9,
        num_samples=4,
        beta_kl=1e-4,
        seed=21,
    )

    assert result.objective_history[0] > result.objective_history[-1]
    assert result.best_objective <= result.objective_history[-1]
    assert result.final_objective <= result.objective_history[0]
