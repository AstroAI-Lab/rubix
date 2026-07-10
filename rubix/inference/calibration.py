"""Posterior calibration diagnostics for variational inference.

This module answers the Phase 4 question of the VI science validation plan:
*are the posterior summaries calibrated, not just numerically stable?*

Two complementary diagnostics are provided over repeated synthetic experiments
(each with a known truth):

- **Central credible-interval coverage.** For a well-calibrated posterior, the
  nominal ``level`` central interval should contain the truth a fraction
  ``level`` of the time across replications.
- **Simulation-based calibration (SBC) rank statistics.** For each replication
  the rank of the truth within the posterior samples is computed. If the
  posterior is calibrated (and truths are drawn from the prior), the ranks are
  uniform on ``{0, ..., num_samples}``; systematic deviations reveal bias
  (skewed ranks) or under/over-dispersion (∪- or ∩-shaped rank histograms).

All statistics operate on posterior *parameter samples in constrained
(physical) space*, so they remain correct under the nonlinear parameter
transforms used by VI. Use :func:`sample_posterior_parameters` to draw those
samples from a :class:`~rubix.inference.variational.VariationalResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
from beartype.typing import Any

from .parameterization import TransformTree, apply_transforms
from .variational import sample_diag_gaussian

ParamsTree = Mapping[str, Mapping[str, Any]]

DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.68, 0.9, 0.95)


def sample_posterior_parameters(
    posterior_mean_params: ParamsTree,
    posterior_log_std_params: ParamsTree,
    num_samples: int,
    transforms: Optional[TransformTree] = None,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Draw parameter samples from a diagonal Gaussian posterior.

    Samples are drawn in unconstrained space and mapped to constrained
    (physical) space via ``transforms`` so downstream coverage/SBC statistics
    are computed in the space where the truth lives.

    Args:
        posterior_mean_params (ParamsTree): Posterior mean in unconstrained
            space (``VariationalResult.posterior_mean_params``).
        posterior_log_std_params (ParamsTree): Posterior log-std in
            unconstrained space (``VariationalResult.posterior_log_std_params``).
        num_samples (int): Number of posterior draws.
        transforms (Optional[TransformTree], optional): Transform tree mapping
            unconstrained latents to constrained parameters. Defaults to
            ``None`` (identity).
        seed (int, optional): PRNG seed. Defaults to 0.

    Raises:
        ValueError: If ``num_samples`` is not strictly positive.

    Returns:
        dict[str, dict[str, Any]]: Parameter pytree whose leaves carry a leading
        sample axis of length ``num_samples``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be strictly positive")

    sample_keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)

    def _single_sample(key):
        sampled_unconstrained = sample_diag_gaussian(
            mean=posterior_mean_params,
            log_std=posterior_log_std_params,
            key=key,
        )
        if transforms is None:
            return sampled_unconstrained
        return apply_transforms(
            params=sampled_unconstrained,
            transforms=transforms,
            direction="forward",
        )

    return jax.lax.map(_single_sample, sample_keys)


def central_interval_quantiles(level: float) -> tuple[float, float]:
    """Return the (low, high) percentile edges of a central interval.

    Args:
        level (float): Central mass in ``(0, 1)`` (e.g. ``0.9`` for a 90%
            interval).

    Raises:
        ValueError: If ``level`` is not strictly inside ``(0, 1)``.

    Returns:
        tuple[float, float]: Percentile edges in ``[0, 100]``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly in (0, 1)")
    return 100.0 * (1.0 - level) / 2.0, 100.0 * (1.0 + level) / 2.0


def interval_covers(
    samples: jnp.ndarray, truth: jnp.ndarray, level: float
) -> jnp.ndarray:
    """Return whether the truth lies inside the central credible interval.

    Args:
        samples (jnp.ndarray): Posterior samples with the sample axis last.
        truth (jnp.ndarray): Truth broadcastable to ``samples`` without the
            sample axis.
        level (float): Central interval mass in ``(0, 1)``.

    Returns:
        jnp.ndarray: Boolean array with the sample axis removed.
    """
    lo_q, hi_q = central_interval_quantiles(level)
    lo = jnp.percentile(samples, lo_q, axis=-1)
    hi = jnp.percentile(samples, hi_q, axis=-1)
    return (truth >= lo) & (truth <= hi)


def normalized_error(samples: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
    """Return the standardized error ``(posterior_mean - truth) / posterior_std``.

    Under a calibrated Gaussian posterior this is distributed as ``N(0, 1)``.
    Zero-width (collapsed) posteriors yield ``nan`` rather than dividing by zero.

    Args:
        samples (jnp.ndarray): Posterior samples with the sample axis last.
        truth (jnp.ndarray): Truth broadcastable to the reduced shape.

    Returns:
        jnp.ndarray: Standardized errors with the sample axis removed.
    """
    mean = jnp.mean(samples, axis=-1)
    std = jnp.std(samples, axis=-1)
    safe_std = jnp.where(std > 0.0, std, jnp.nan)
    return (mean - truth) / safe_std


def sbc_rank(samples: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
    """Return the SBC rank of the truth within the posterior samples.

    The rank is the number of samples strictly less than the truth, an integer
    in ``{0, ..., num_samples}``.

    Args:
        samples (jnp.ndarray): Posterior samples with the sample axis last.
        truth (jnp.ndarray): Truth broadcastable to the reduced shape.

    Returns:
        jnp.ndarray: Integer ranks with the sample axis removed.
    """
    return jnp.sum(samples < truth[..., None], axis=-1)


def joint_credible_coverage(
    samples: jnp.ndarray,
    truths: jnp.ndarray,
    levels: Sequence[float] = DEFAULT_LEVELS,
    jitter: float = 1e-12,
) -> dict[str, Any]:
    """Empirical coverage of *joint* highest-density credible regions.

    For each replication the joint credible region is the Mahalanobis ellipsoid
    of the posterior samples: the truth is covered at ``level`` if its squared
    Mahalanobis distance to the posterior mean is below the ``level`` empirical
    quantile of the samples' own squared Mahalanobis distances. This exercises
    the *correlation* structure of the posterior (e.g. an age--metallicity
    covariance), which per-parameter marginal coverage ignores.

    Args:
        samples (jnp.ndarray): Posterior samples shaped ``(n_trials, n_samples,
            n_dim)`` (the joint block of coupled parameters).
        truths (jnp.ndarray): Ground-truth values shaped ``(n_trials, n_dim)``.
        levels (Sequence[float], optional): Nominal joint-region levels. Defaults
            to :data:`DEFAULT_LEVELS`.
        jitter (float, optional): Diagonal regularization added to each posterior
            covariance for numerical stability. Defaults to 1e-12.

    Raises:
        ValueError: If array shapes are inconsistent or no levels are given.

    Returns:
        dict[str, Any]: ``{"n_trials", "n_dim", "levels", "nominal_coverage",
        "empirical_coverage"}``.
    """
    samples = jnp.asarray(samples)
    truths = jnp.asarray(truths)
    if samples.ndim != 3:
        raise ValueError("samples must have shape (n_trials, n_samples, n_dim)")
    if truths.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("truths must have shape (n_trials, n_dim)")
    if len(levels) == 0:
        raise ValueError("levels must contain at least one entry")

    n_dim = int(samples.shape[2])
    level_arr = jnp.asarray([float(x) for x in levels])

    def trial_coverage(trial_samples, trial_truth):
        mean = jnp.mean(trial_samples, axis=0)
        cov = jnp.cov(trial_samples, rowvar=False)
        cov = jnp.atleast_2d(cov) + jitter * jnp.eye(n_dim, dtype=cov.dtype)
        cov_inv = jnp.linalg.inv(cov)
        centered = trial_samples - mean
        d2_samples = jnp.einsum("si,ij,sj->s", centered, cov_inv, centered)
        truth_centered = trial_truth - mean
        d2_truth = truth_centered @ cov_inv @ truth_centered
        thresholds = jnp.percentile(d2_samples, level_arr * 100.0)
        return (d2_truth <= thresholds).astype(jnp.float32)

    covered = jax.vmap(trial_coverage)(samples, truths)  # (n_trials, n_levels)
    empirical = [float(x) for x in jnp.mean(covered, axis=0)]
    return {
        "n_trials": int(samples.shape[0]),
        "n_dim": n_dim,
        "levels": [float(x) for x in levels],
        "nominal_coverage": [float(x) for x in levels],
        "empirical_coverage": empirical,
    }


@dataclass
class CalibrationSummary:
    """Aggregated calibration diagnostics over repeated experiments."""

    n_trials: int
    n_samples: int
    levels: tuple[float, ...]
    nominal_coverage: tuple[float, ...]
    empirical_coverage: tuple[float, ...]
    mean_z: float
    rms_z: float
    sbc_num_bins: int
    sbc_rank_counts: tuple[int, ...]
    sbc_reduced_chi2: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict view of the summary."""
        return {
            "n_trials": self.n_trials,
            "n_samples": self.n_samples,
            "levels": list(self.levels),
            "nominal_coverage": list(self.nominal_coverage),
            "empirical_coverage": list(self.empirical_coverage),
            "coverage_error": [
                emp - nom
                for emp, nom in zip(self.empirical_coverage, self.nominal_coverage)
            ],
            "mean_z": self.mean_z,
            "rms_z": self.rms_z,
            "sbc_num_bins": self.sbc_num_bins,
            "sbc_rank_counts": list(self.sbc_rank_counts),
            "sbc_reduced_chi2": self.sbc_reduced_chi2,
        }


def _default_sbc_bins(n_samples: int, n_trials: int) -> int:
    """Pick a reasonable SBC histogram bin count.

    Targets roughly five or more expected counts per bin, caps the count at 20,
    and never exceeds the number of possible ranks (``n_samples + 1``). Bins are
    equal width over the rank range; for the usual ``n_samples`` this leaves a
    negligible ``O(1/n_samples)`` unevenness in per-bin rank counts.
    """
    num_ranks = n_samples + 1
    by_expected_count = max(1, n_trials // 5)
    return max(1, min(20, num_ranks, by_expected_count))


def sbc_rank_histogram(
    ranks: jnp.ndarray, n_samples: int, num_bins: int
) -> tuple[list[int], float]:
    """Bin SBC ranks and return counts plus a reduced chi-square uniformity stat.

    Args:
        ranks (jnp.ndarray): Integer ranks in ``{0, ..., n_samples}``.
        n_samples (int): Number of posterior samples per trial.
        num_bins (int): Number of equal-width histogram bins over the rank range.

    Raises:
        ValueError: If ``num_bins`` is not strictly positive.

    Returns:
        tuple[list[int], float]: Per-bin counts and the reduced chi-square
        (``chi2 / dof``) against a uniform expectation; ``~1`` indicates
        uniform ranks, large values indicate miscalibration.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be strictly positive")

    ranks_arr = jnp.asarray(ranks).ravel()
    n_trials = int(ranks_arr.shape[0])
    edges = jnp.linspace(0.0, float(n_samples + 1), num_bins + 1)
    counts = jnp.histogram(ranks_arr.astype(jnp.float32), bins=edges)[0]
    counts_list = [int(c) for c in counts]

    expected = n_trials / num_bins
    if expected <= 0.0 or num_bins == 1:
        return counts_list, float("nan")
    chi2 = float(jnp.sum((counts - expected) ** 2 / expected))
    dof = max(num_bins - 1, 1)
    return counts_list, chi2 / dof


def summarize_calibration(
    samples: jnp.ndarray,
    truths: jnp.ndarray,
    levels: Sequence[float] = DEFAULT_LEVELS,
    sbc_num_bins: Optional[int] = None,
) -> CalibrationSummary:
    """Summarize coverage and SBC diagnostics for one scalar quantity.

    Args:
        samples (jnp.ndarray): Posterior samples shaped ``(n_trials, n_samples)``
            (one row of samples per replication).
        truths (jnp.ndarray): Ground-truth values shaped ``(n_trials,)``.
        levels (Sequence[float], optional): Nominal central-interval levels.
            Defaults to :data:`DEFAULT_LEVELS`.
        sbc_num_bins (Optional[int], optional): SBC histogram bin count. Defaults
            to an automatically chosen divisor of ``n_samples + 1``.

    Raises:
        ValueError: If array shapes are inconsistent or no levels are given.

    Returns:
        CalibrationSummary: Aggregated calibration diagnostics.
    """
    samples = jnp.asarray(samples)
    truths = jnp.asarray(truths)

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_trials, n_samples)")
    if truths.ndim != 1 or truths.shape[0] != samples.shape[0]:
        raise ValueError("truths must have shape (n_trials,) matching samples")
    if len(levels) == 0:
        raise ValueError("levels must contain at least one entry")

    n_trials, n_samples = int(samples.shape[0]), int(samples.shape[1])

    empirical = []
    for level in levels:
        covered = interval_covers(samples, truths, level)
        empirical.append(float(jnp.mean(covered.astype(jnp.float32))))

    z = normalized_error(samples, truths)
    finite_z = z[jnp.isfinite(z)]
    if finite_z.size == 0:
        mean_z = float("nan")
        rms_z = float("nan")
    else:
        mean_z = float(jnp.mean(finite_z))
        rms_z = float(jnp.sqrt(jnp.mean(finite_z**2)))

    ranks = sbc_rank(samples, truths)
    num_bins = (
        _default_sbc_bins(n_samples, n_trials)
        if sbc_num_bins is None
        else int(sbc_num_bins)
    )
    counts, reduced_chi2 = sbc_rank_histogram(ranks, n_samples, num_bins)

    return CalibrationSummary(
        n_trials=n_trials,
        n_samples=n_samples,
        levels=tuple(float(x) for x in levels),
        nominal_coverage=tuple(float(x) for x in levels),
        empirical_coverage=tuple(empirical),
        mean_z=mean_z,
        rms_z=rms_z,
        sbc_num_bins=num_bins,
        sbc_rank_counts=tuple(counts),
        sbc_reduced_chi2=reduced_chi2,
    )


def summarize_parameter_calibration(
    sample_sets: ParamsTree,
    truths: ParamsTree,
    levels: Sequence[float] = DEFAULT_LEVELS,
    sbc_num_bins: Optional[int] = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Summarize calibration per parameter leaf, pooling over vector components.

    Each leaf of ``sample_sets`` has shape ``(n_trials, n_samples, *param_shape)``
    and the matching ``truths`` leaf has shape ``(n_trials, *param_shape)``. All
    ``(trial, component)`` pairs are pooled into independent calibration trials.

    Args:
        sample_sets (ParamsTree): Posterior samples per parameter leaf.
        truths (ParamsTree): Ground-truth values per parameter leaf.
        levels (Sequence[float], optional): Nominal interval levels. Defaults to
            :data:`DEFAULT_LEVELS`.
        sbc_num_bins (Optional[int], optional): SBC bin count. Defaults to auto.

    Raises:
        ValueError: If a truth leaf is missing or shapes are inconsistent.

    Returns:
        dict[str, dict[str, dict[str, Any]]]: ``component -> field -> summary``
        where each summary is :meth:`CalibrationSummary.to_dict`.
    """
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for component, fields in sample_sets.items():
        if component not in truths:
            raise ValueError(f"truths missing component '{component}'")
        report[component] = {}
        for field, leaf in fields.items():
            if field not in truths[component]:
                raise ValueError(f"truths missing field '{component}.{field}'")
            leaf_arr = jnp.asarray(leaf)
            truth_arr = jnp.asarray(truths[component][field])
            if leaf_arr.ndim < 2:
                raise ValueError(
                    f"'{component}.{field}' samples must have shape "
                    "(n_trials, n_samples, *param_shape)"
                )
            n_trials, n_samples = int(leaf_arr.shape[0]), int(leaf_arr.shape[1])
            if truth_arr.shape[0] != n_trials:
                raise ValueError(f"'{component}.{field}' truth n_trials mismatch")
            # Move the sample axis last and pool trials with components:
            # (n_trials, n_samples, *p) -> (n_trials * prod(p), n_samples).
            moved = jnp.moveaxis(leaf_arr, 1, -1)
            pooled_samples = moved.reshape(-1, n_samples)
            pooled_truths = truth_arr.reshape(-1)
            summary = summarize_calibration(
                pooled_samples,
                pooled_truths,
                levels=levels,
                sbc_num_bins=sbc_num_bins,
            )
            report[component][field] = summary.to_dict()
    return report
