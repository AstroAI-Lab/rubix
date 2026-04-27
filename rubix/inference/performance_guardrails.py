from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from rubix.utils import read_yaml

from .benchmark import IFUCubeBenchmarkResult
from .vi_benchmark import VIBenchmarkResult


@dataclass(frozen=True)
class RuntimeThresholds:
    """Thresholds for runtime regression checks."""

    max_mean_runtime_s: Optional[float] = None
    max_median_runtime_s: Optional[float] = None


@dataclass(frozen=True)
class OptimizationObjectiveThresholds:
    """Thresholds for optimization loss quality checks."""

    max_final_loss: Optional[float] = None
    max_best_loss: Optional[float] = None


@dataclass(frozen=True)
class VIObjectiveThresholds:
    """Thresholds for variational inference objective quality checks."""

    max_final_objective: Optional[float] = None
    max_best_objective: Optional[float] = None


@dataclass(frozen=True)
class PerformanceCheckResult:
    """Outcome of a performance guardrail check."""

    passed: bool
    message: str
    failed_conditions: tuple[str, ...] = ()


def _as_optional_float(value: Any) -> Optional[float]:
    """Convert supported scalar value to optional float."""
    if value is None:
        return None
    return float(value)


def load_guardrail_threshold_profile(
    config_path: Union[str, Path],
    profile_name: str,
    mode: str,
) -> tuple[
    RuntimeThresholds,
    Union[OptimizationObjectiveThresholds, VIObjectiveThresholds],
]:
    """Load runtime/objective thresholds from a versioned YAML profile.

    Args:
        config_path (Union[str, Path]): Guardrail threshold config path.
        profile_name (str): Profile key under ``profiles``.
        mode (str): One of ``'optimization'`` or ``'variational'``.

    Raises:
        ValueError: If config schema, profile, or mode entries are invalid.

    Returns:
        tuple[RuntimeThresholds, Union[OptimizationObjectiveThresholds,
        VIObjectiveThresholds]]: Runtime and objective thresholds.
    """
    if mode not in {"optimization", "variational"}:
        raise ValueError("mode must be 'optimization' or 'variational'")

    cfg = read_yaml(str(config_path))
    profiles = cfg.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("guardrail config must contain a 'profiles' mapping")

    profile = profiles.get(profile_name)
    if not isinstance(profile, Mapping):
        raise ValueError(
            f"profile '{profile_name}' not found in guardrail config {config_path}"
        )

    mode_cfg = profile.get(mode)
    if not isinstance(mode_cfg, Mapping):
        raise ValueError(f"profile '{profile_name}' must define a '{mode}' mapping")

    runtime_cfg = mode_cfg.get("runtime") or {}
    if not isinstance(runtime_cfg, Mapping):
        raise ValueError(f"profile '{profile_name}' {mode}.runtime must be a mapping")

    runtime = RuntimeThresholds(
        max_mean_runtime_s=_as_optional_float(runtime_cfg.get("max_mean_runtime_s")),
        max_median_runtime_s=_as_optional_float(
            runtime_cfg.get("max_median_runtime_s")
        ),
    )

    objective_cfg = mode_cfg.get("objective") or {}
    if not isinstance(objective_cfg, Mapping):
        raise ValueError(f"profile '{profile_name}' {mode}.objective must be a mapping")

    if mode == "optimization":
        objective = OptimizationObjectiveThresholds(
            max_final_loss=_as_optional_float(objective_cfg.get("max_final_loss")),
            max_best_loss=_as_optional_float(objective_cfg.get("max_best_loss")),
        )
    else:
        objective = VIObjectiveThresholds(
            max_final_objective=_as_optional_float(
                objective_cfg.get("max_final_objective")
            ),
            max_best_objective=_as_optional_float(
                objective_cfg.get("max_best_objective")
            ),
        )

    return runtime, objective


def _check_runtime(
    mean_runtime_s: float,
    median_runtime_s: float,
    runtime_thresholds: RuntimeThresholds,
) -> list[str]:
    errors: list[str] = []
    if runtime_thresholds.max_mean_runtime_s is not None:
        if mean_runtime_s > runtime_thresholds.max_mean_runtime_s:
            errors.append(
                f"mean runtime {mean_runtime_s:.6f}s exceeds "
                f"threshold {runtime_thresholds.max_mean_runtime_s:.6f}s"
            )

    if runtime_thresholds.max_median_runtime_s is not None:
        if median_runtime_s > runtime_thresholds.max_median_runtime_s:
            errors.append(
                f"median runtime {median_runtime_s:.6f}s exceeds "
                f"threshold {runtime_thresholds.max_median_runtime_s:.6f}s"
            )

    return errors


def check_ifu_optimization_guardrails(
    result: IFUCubeBenchmarkResult,
    runtime_thresholds: RuntimeThresholds,
    objective_thresholds: OptimizationObjectiveThresholds,
) -> PerformanceCheckResult:
    """Check optimization benchmark result against runtime/objective thresholds.

    Args:
        result (IFUCubeBenchmarkResult): Optimization benchmark result.
        runtime_thresholds (RuntimeThresholds): Runtime limits.
        objective_thresholds (OptimizationObjectiveThresholds): Loss limits.

    Returns:
        PerformanceCheckResult: Pass/fail status and explanatory message.
    """
    errors = _check_runtime(
        result.mean_runtime_s,
        result.median_runtime_s,
        runtime_thresholds,
    )

    if objective_thresholds.max_final_loss is not None:
        if result.final_loss > objective_thresholds.max_final_loss:
            errors.append(
                f"final loss {result.final_loss:.6e} exceeds "
                f"threshold {objective_thresholds.max_final_loss:.6e}"
            )

    if objective_thresholds.max_best_loss is not None:
        if result.best_loss > objective_thresholds.max_best_loss:
            errors.append(
                f"best loss {result.best_loss:.6e} exceeds "
                f"threshold {objective_thresholds.max_best_loss:.6e}"
            )

    if errors:
        return PerformanceCheckResult(
            passed=False,
            message="; ".join(errors),
            failed_conditions=tuple(errors),
        )

    return PerformanceCheckResult(
        passed=True,
        message="optimization benchmark satisfies configured thresholds",
        failed_conditions=(),
    )


def check_vi_guardrails(
    result: VIBenchmarkResult,
    runtime_thresholds: RuntimeThresholds,
    objective_thresholds: VIObjectiveThresholds,
) -> PerformanceCheckResult:
    """Check VI benchmark result against runtime/objective thresholds.

    Args:
        result (VIBenchmarkResult): VI benchmark result.
        runtime_thresholds (RuntimeThresholds): Runtime limits.
        objective_thresholds (VIObjectiveThresholds): Objective limits.

    Returns:
        PerformanceCheckResult: Pass/fail status and explanatory message.
    """
    errors = _check_runtime(
        result.mean_runtime_s,
        result.median_runtime_s,
        runtime_thresholds,
    )

    if objective_thresholds.max_final_objective is not None:
        if result.final_objective > objective_thresholds.max_final_objective:
            errors.append(
                f"final objective {result.final_objective:.6e} exceeds "
                f"threshold {objective_thresholds.max_final_objective:.6e}"
            )

    if objective_thresholds.max_best_objective is not None:
        if result.best_objective > objective_thresholds.max_best_objective:
            errors.append(
                f"best objective {result.best_objective:.6e} exceeds "
                f"threshold {objective_thresholds.max_best_objective:.6e}"
            )

    if errors:
        return PerformanceCheckResult(
            passed=False,
            message="; ".join(errors),
            failed_conditions=tuple(errors),
        )

    return PerformanceCheckResult(
        passed=True,
        message="VI benchmark satisfies configured thresholds",
        failed_conditions=(),
    )
