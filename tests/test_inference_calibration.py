import jax.numpy as jnp
import numpy as np
import pytest

from rubix.inference import (
    CalibrationSummary,
    sample_posterior_parameters,
    sbc_rank,
    summarize_calibration,
    summarize_parameter_calibration,
)
from rubix.inference.calibration import (
    central_interval_quantiles,
    interval_covers,
    normalized_error,
)
from rubix.inference.parameterization import SigmoidBounds


def _calibrated_samples(n_trials=500, n_samples=400, sigma=1.0, seed=0):
    """Return (samples, truths) where the truth is exchangeable with samples.

    Drawing the truth and the posterior samples from the same Gaussian makes the
    posterior calibrated by construction: central intervals cover at the nominal
    rate and SBC ranks are uniform.
    """
    rng = np.random.default_rng(seed)
    means = np.zeros(n_trials)
    truths = rng.normal(means, sigma)
    samples = rng.normal(means[:, None], sigma, size=(n_trials, n_samples))
    return jnp.asarray(samples), jnp.asarray(truths)


def test_central_interval_quantiles_edges():
    lo, hi = central_interval_quantiles(0.9)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(95.0)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_central_interval_quantiles_rejects_out_of_range(level):
    with pytest.raises(ValueError, match="level must lie strictly"):
        central_interval_quantiles(level)


def test_interval_covers_and_normalized_error_basic():
    samples = jnp.linspace(0.0, 100.0, 101)  # median 50, symmetric
    assert bool(interval_covers(samples, jnp.asarray(50.0), 0.5))
    assert not bool(interval_covers(samples, jnp.asarray(1.0), 0.5))
    z = normalized_error(samples, jnp.asarray(50.0))
    assert abs(float(z)) < 1e-6


def test_normalized_error_collapsed_posterior_is_nan():
    samples = jnp.ones((1, 8))
    z = normalized_error(samples, jnp.asarray([1.0]))
    assert bool(jnp.isnan(z[0]))


def test_sbc_rank_counts_samples_below_truth():
    samples = jnp.asarray([[0.0, 1.0, 2.0, 3.0]])
    assert int(sbc_rank(samples, jnp.asarray([2.5]))[0]) == 3
    assert int(sbc_rank(samples, jnp.asarray([-1.0]))[0]) == 0


def test_calibrated_posterior_recovers_nominal_coverage():
    samples, truths = _calibrated_samples()
    summary = summarize_calibration(samples, truths)

    assert isinstance(summary, CalibrationSummary)
    for nominal, empirical in zip(summary.nominal_coverage, summary.empirical_coverage):
        assert abs(empirical - nominal) < 0.06

    assert abs(summary.mean_z) < 0.15
    assert 0.85 < summary.rms_z < 1.15
    # Uniform SBC ranks -> reduced chi-square near 1.
    assert summary.sbc_reduced_chi2 < 3.0


def test_underdispersed_posterior_is_flagged():
    rng = np.random.default_rng(1)
    n_trials, n_samples = 500, 400
    means = np.zeros(n_trials)
    truths = rng.normal(means, 1.0)
    # Posterior far too narrow relative to the spread of truths.
    samples = jnp.asarray(rng.normal(means[:, None], 0.2, size=(n_trials, n_samples)))
    summary = summarize_calibration(samples, jnp.asarray(truths))

    # 90% interval should badly under-cover.
    idx_90 = summary.levels.index(0.9)
    assert summary.empirical_coverage[idx_90] < 0.6
    assert summary.rms_z > 2.0
    # Non-uniform (U-shaped) ranks -> large reduced chi-square.
    assert summary.sbc_reduced_chi2 > 5.0


def test_summarize_calibration_rejects_bad_shapes():
    with pytest.raises(ValueError, match="shape"):
        summarize_calibration(jnp.ones((3, 3, 3)), jnp.ones(3))
    with pytest.raises(ValueError, match="truths must have shape"):
        summarize_calibration(jnp.ones((3, 4)), jnp.ones(2))


def test_sample_posterior_parameters_respects_transforms_and_shape():
    mean = {"stars": {"metallicity": jnp.zeros((2,))}}
    log_std = {"stars": {"metallicity": jnp.full((2,), -1.0)}}
    transforms = {"stars": {"metallicity": SigmoidBounds(lower=0.0, upper=0.05)}}

    samples = sample_posterior_parameters(
        posterior_mean_params=mean,
        posterior_log_std_params=log_std,
        num_samples=64,
        transforms=transforms,
        seed=3,
    )
    leaf = samples["stars"]["metallicity"]
    assert leaf.shape == (64, 2)
    assert bool(jnp.all(leaf > 0.0)) and bool(jnp.all(leaf < 0.05))


def test_sample_posterior_parameters_rejects_nonpositive():
    with pytest.raises(ValueError, match="num_samples must be strictly positive"):
        sample_posterior_parameters(
            posterior_mean_params={"stars": {"age": jnp.zeros((1,))}},
            posterior_log_std_params={"stars": {"age": jnp.zeros((1,))}},
            num_samples=0,
        )


def test_summarize_parameter_calibration_pools_components():
    # Two trials, each with a 3-component age vector, 200 samples.
    rng = np.random.default_rng(2)
    n_trials, n_samples, n_comp = 200, 300, 3
    means = np.zeros((n_trials, n_comp))
    truths = rng.normal(means, 1.0)
    samples = rng.normal(means[:, None, :], 1.0, size=(n_trials, n_samples, n_comp))
    report = summarize_parameter_calibration(
        sample_sets={"stars": {"age": jnp.asarray(samples)}},
        truths={"stars": {"age": jnp.asarray(truths)}},
    )
    summary = report["stars"]["age"]
    assert summary["n_trials"] == n_trials * n_comp
    idx_68 = summary["levels"].index(0.68)
    assert abs(summary["empirical_coverage"][idx_68] - 0.68) < 0.06


def test_summarize_parameter_calibration_rejects_missing_truth():
    with pytest.raises(ValueError, match="truths missing"):
        summarize_parameter_calibration(
            sample_sets={"stars": {"age": jnp.ones((2, 4))}},
            truths={"stars": {}},
        )


def test_joint_credible_coverage_calibrated_and_underdispersed():
    from rubix.inference import joint_credible_coverage

    rng = np.random.default_rng(0)
    n_trials, n_samples = 400, 500
    # Correlated 2D Gaussian; truth exchangeable with samples -> calibrated.
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    chol = np.linalg.cholesky(cov)
    truths = (chol @ rng.normal(size=(2, n_trials))).T  # (T, 2)
    samples = np.einsum("ij,tsj->tsi", chol, rng.normal(size=(n_trials, n_samples, 2)))
    rep = joint_credible_coverage(jnp.asarray(samples), jnp.asarray(truths))
    idx90 = rep["levels"].index(0.9)
    assert abs(rep["empirical_coverage"][idx90] - 0.9) < 0.06

    # Under-dispersed joint posterior (half width) -> joint under-coverage.
    narrow = 0.5 * samples
    rep2 = joint_credible_coverage(jnp.asarray(narrow), jnp.asarray(truths))
    assert rep2["empirical_coverage"][idx90] < 0.7


def test_joint_credible_coverage_rejects_bad_shapes():
    from rubix.inference import joint_credible_coverage

    with pytest.raises(ValueError, match="n_trials, n_samples, n_dim"):
        joint_credible_coverage(jnp.ones((3, 4)), jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="truths must have shape"):
        joint_credible_coverage(jnp.ones((3, 4, 2)), jnp.ones((3, 3)))
