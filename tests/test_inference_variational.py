import warnings

import jax
import jax.numpy as jnp
import optax
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    build_age_metallicity_transforms,
    initialize_mean_field_params,
    kl_diag_gaussian_to_standard_normal,
    optimize_variational_ifu_cube,
    optimize_variational_posterior,
    sample_diag_gaussian,
    sample_low_rank_gaussian,
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


def test_kl_diag_gaussian_prior_std_matches_analytic():
    # q = N(0, I) over 2 latents; prior = N(0, tau^2 I). Per-dim KL is
    # 0.5 * (1/tau^2 - 1 + 2 log tau).
    mean = {"stars": {"age": jnp.array([0.0, 0.0])}}
    log_std = {"stars": {"age": jnp.array([0.0, 0.0])}}
    tau = 2.0
    kl = kl_diag_gaussian_to_standard_normal(mean, log_std, prior_std=tau)
    per_dim = 0.5 * (1.0 / tau**2 - 1.0 + 2.0 * jnp.log(tau))
    assert float(kl) == pytest.approx(float(2 * per_dim), rel=1e-6)

    # A wider prior penalizes an off-center mean less (the point of the fix):
    # this is what reduces the midpoint bias on calibrated runs.
    off_center = {"stars": {"age": jnp.array([2.0, -2.0])}}
    wide = kl_diag_gaussian_to_standard_normal(off_center, log_std, prior_std=tau)
    standard = kl_diag_gaussian_to_standard_normal(off_center, log_std)
    assert float(wide) < float(standard)


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
    # best_objective is the minimum EMA-smoothed objective, so it lies between
    # the raw minimum and the first (unsmoothed) objective.
    assert result.best_objective >= min(result.objective_history) - 1e-6
    assert result.best_objective <= result.objective_history[0] + 1e-6
    assert jnp.isfinite(result.final_objective)
    assert jnp.isfinite(result.final_reconstruction)
    assert jnp.isfinite(result.final_kl)
    assert result.steps_run <= 200


def test_optimize_variational_posterior_supports_lbfgs_extra_args():
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
        max_steps=5,
        tol=1e-12,
        num_samples=1,
        beta_kl=0.0,
        init_log_std=-12.0,
        optimizer=optax.lbfgs(),
        seed=11,
        # L-BFGS here is effectively deterministic; disable EMA smoothing so the
        # best objective tracks the rapidly converging raw value.
        best_selection_ema_decay=0.0,
    )

    assert result.objective_history[-1] < result.objective_history[0]
    assert result.best_objective < 1e-6


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


def test_optimize_variational_ifu_cube_warns_on_uncalibrated_units():
    params_init = {
        "stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.001])}
    }
    with pytest.warns(UserWarning, match="posterior widths are NOT"):
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init=params_init,
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 1, 1)),
            sigma=jnp.ones((1, 1, 1)),
            normalize_loss=True,
            beta_kl=1e-3,
            max_steps=1,
        )


def test_optimize_variational_ifu_cube_calibrated_default_is_quiet():
    params_init = {
        "stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.001])}
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init=params_init,
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 1, 1)),
            sigma=jnp.ones((1, 1, 1)),
            max_steps=1,
        )
    assert not any("posterior widths are NOT" in str(w.message) for w in caught)


def test_optimize_variational_ifu_cube_rejects_both_sigma_and_inv_variance():
    with pytest.raises(
        ValueError,
        match="only one of sigma or inv_variance may be provided, not both",
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
            sigma=jnp.ones((1, 1, 1)),
            inv_variance=jnp.ones((1, 1, 1)),
        )


def test_optimize_variational_ifu_cube_rejects_sigma_shape_mismatch():
    with pytest.raises(ValueError, match="sigma shape"):
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init={
                "stars": {
                    "age": jnp.array([0.5]),
                    "metallicity": jnp.array([0.001]),
                }
            },
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 2, 3)),
            sigma=jnp.ones((1, 2, 4)),
        )


def test_optimize_variational_ifu_cube_rejects_inv_variance_shape_mismatch():
    with pytest.raises(ValueError, match="inv_variance shape"):
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init={
                "stars": {
                    "age": jnp.array([0.5]),
                    "metallicity": jnp.array([0.001]),
                }
            },
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 2, 3)),
            inv_variance=jnp.ones((2, 2, 3)),
        )


def test_optimize_variational_ifu_cube_rejects_mask_shape_mismatch():
    with pytest.raises(ValueError, match="mask shape"):
        optimize_variational_ifu_cube(
            pipeline=DummyPipeline(),
            params_init={
                "stars": {
                    "age": jnp.array([0.5]),
                    "metallicity": jnp.array([0.001]),
                }
            },
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 2, 3)),
            mask=jnp.ones((1, 3, 3)),
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
    # best_objective is the smoothed (EMA) minimum, bounded by the raw minimum
    # and the first objective rather than the last raw value.
    assert result.best_objective >= min(result.objective_history) - 1e-6
    assert result.best_objective <= result.objective_history[0] + 1e-6
    assert result.final_objective <= result.objective_history[0]


def test_low_rank_posterior_returns_factor_and_marginal_widths():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.001])}
    }
    target = jnp.array([[[5.0]]])

    result = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=5e-2,
        max_steps=80,
        tol=1e-9,
        num_samples=8,
        beta_kl=1.0,
        seed=5,
        posterior_rank=1,
    )

    # Latent dimension is 2 (age + metallicity), rank 1.
    assert result.posterior_factor_params is not None
    assert result.posterior_factor_params.shape == (2, 1)
    # Reported (marginal) log-std is at least the diagonal init log-std.
    assert bool(jnp.all(result.posterior_log_std_params["stars"]["age"] >= -2.5))
    assert jnp.isfinite(result.final_kl)


def test_low_rank_posterior_captures_ridge_correlation():
    # DummyPipeline output = age + 2 * metallicity, so the likelihood only
    # constrains that linear combination -> an anti-correlated posterior ridge.
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.5])}}
    target = jnp.array([[[3.0]]])

    _, state = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=5e-2,
        max_steps=600,
        tol=1e-12,
        num_samples=16,
        beta_kl=1.0,
        seed=7,
        posterior_rank=1,
        return_state=True,
    )

    mean = state.variational_params["mean"]
    diag_log_std = state.variational_params["log_std"]
    factor = state.variational_params["factor"]

    keys = jax.random.split(jax.random.PRNGKey(0), 20000)
    samples = jax.vmap(
        lambda k: sample_low_rank_gaussian(mean, diag_log_std, factor, k)
    )(keys)
    age = samples["stars"]["age"][:, 0]
    met = samples["stars"]["metallicity"][:, 0]
    corr = float(jnp.corrcoef(age, met)[0, 1])
    # The ridge forces a clear negative age-metallicity correlation, which a
    # diagonal mean-field posterior cannot represent.
    assert corr < -0.2


def test_low_rank_posterior_resume_round_trips():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.001])}
    }
    target = jnp.array([[[4.0]]])

    _, state = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        max_steps=30,
        num_samples=4,
        beta_kl=1.0,
        seed=3,
        posterior_rank=2,
        return_state=True,
    )
    assert "factor" in state.variational_params

    # Resume ignores posterior_rank and continues the persisted low-rank family.
    result2 = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        max_steps=30,
        num_samples=4,
        beta_kl=1.0,
        state_init=state,
    )
    assert result2.posterior_factor_params is not None
    assert result2.posterior_factor_params.shape == (2, 2)
    assert result2.steps_run == 60


def test_optimize_variational_posterior_rejects_negative_rank():
    with pytest.raises(ValueError, match="posterior_rank must be non-negative"):
        optimize_variational_posterior(
            pipeline=DummyPipeline(),
            params_init={"stars": {"age": jnp.array([0.5])}},
            static_data=_make_rubix_data(),
            target=jnp.array([[[1.0]]]),
            max_steps=1,
            posterior_rank=-1,
        )


def test_best_selection_decay_zero_matches_raw_minimum():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.001])}
    }
    target = jnp.array([[[5.0]]])

    result = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=5e-2,
        max_steps=50,
        num_samples=2,
        beta_kl=1.0,
        seed=1,
        best_selection_ema_decay=0.0,
    )
    # With smoothing disabled, best equals the raw objective minimum.
    assert result.best_objective == pytest.approx(min(result.objective_history))


def test_best_selection_smoothing_is_between_min_and_first():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.001])}
    }
    target = jnp.array([[[5.0]]])

    smoothed = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=5e-2,
        max_steps=60,
        num_samples=2,
        beta_kl=1.0,
        seed=1,
        best_selection_ema_decay=0.9,
    )
    raw_min = min(smoothed.objective_history)
    # A smoothed best cannot dip below the raw minimum.
    assert smoothed.best_objective >= raw_min - 1e-6


def test_optimize_variational_posterior_rejects_bad_ema_decay():
    with pytest.raises(ValueError, match="best_selection_ema_decay"):
        optimize_variational_posterior(
            pipeline=DummyPipeline(),
            params_init={"stars": {"age": jnp.array([0.5])}},
            static_data=_make_rubix_data(),
            target=jnp.array([[[1.0]]]),
            max_steps=1,
            best_selection_ema_decay=1.0,
        )
