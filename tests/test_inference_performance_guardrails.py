from rubix.inference.benchmark import IFUCubeBenchmarkResult
from rubix.inference.performance_guardrails import (
    ObjectiveThresholds,
    RuntimeThresholds,
    check_ifu_optimization_guardrails,
    check_vi_guardrails,
)
from rubix.inference.vi_benchmark import VIBenchmarkResult


def _make_opt_result(mean_runtime=1.0, final_loss=1e-4, best_loss=1e-4):
    return IFUCubeBenchmarkResult(
        repeats=3,
        warmup=True,
        runtimes_s=[mean_runtime, mean_runtime, mean_runtime],
        mean_runtime_s=mean_runtime,
        median_runtime_s=mean_runtime,
        min_runtime_s=mean_runtime,
        max_runtime_s=mean_runtime,
        steps_run=10,
        final_loss=final_loss,
        best_loss=best_loss,
        target_nbytes=1024,
        mask_nbytes=0,
        weights_nbytes=0,
        estimated_objective_working_set_nbytes=4096,
    )


def _make_vi_result(mean_runtime=1.0, final_obj=1e-3, best_obj=8e-4):
    return VIBenchmarkResult(
        repeats=3,
        warmup=True,
        runtimes_s=[mean_runtime, mean_runtime, mean_runtime],
        mean_runtime_s=mean_runtime,
        median_runtime_s=mean_runtime,
        min_runtime_s=mean_runtime,
        max_runtime_s=mean_runtime,
        steps_run=12,
        final_objective=final_obj,
        best_objective=best_obj,
        final_reconstruction=final_obj,
        final_kl=1e-5,
        target_nbytes=1024,
    )


def test_check_ifu_optimization_guardrails_passes_within_limits():
    result = _make_opt_result()
    runtime = RuntimeThresholds(max_mean_runtime_s=2.0, max_median_runtime_s=2.0)
    objective = ObjectiveThresholds(max_final_loss=1e-3, max_best_loss=1e-3)

    check = check_ifu_optimization_guardrails(result, runtime, objective)
    assert check.passed is True


def test_check_ifu_optimization_guardrails_fails_on_runtime_and_loss():
    result = _make_opt_result(mean_runtime=3.0, final_loss=1e-1, best_loss=1e-2)
    runtime = RuntimeThresholds(max_mean_runtime_s=2.0, max_median_runtime_s=2.0)
    objective = ObjectiveThresholds(max_final_loss=1e-3, max_best_loss=1e-3)

    check = check_ifu_optimization_guardrails(result, runtime, objective)
    assert check.passed is False
    assert "mean runtime" in check.message
    assert "final loss" in check.message


def test_check_vi_guardrails_passes_within_limits():
    result = _make_vi_result()
    runtime = RuntimeThresholds(max_mean_runtime_s=2.0, max_median_runtime_s=2.0)
    objective = ObjectiveThresholds(max_final_objective=2e-3, max_best_objective=2e-3)

    check = check_vi_guardrails(result, runtime, objective)
    assert check.passed is True


def test_check_vi_guardrails_fails_on_objective():
    result = _make_vi_result(final_obj=5e-2, best_obj=4e-2)
    runtime = RuntimeThresholds(max_mean_runtime_s=2.0, max_median_runtime_s=2.0)
    objective = ObjectiveThresholds(max_final_objective=1e-3, max_best_objective=1e-3)

    check = check_vi_guardrails(result, runtime, objective)
    assert check.passed is False
    assert "final objective" in check.message
