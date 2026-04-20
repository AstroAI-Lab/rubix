"""Inference helpers for gradient-based modeling workflows."""

from .api import apply_params, forward, loss, value_and_grad
from .benchmark import (
    IFUCubeBenchmarkResult,
    benchmark_callable,
    benchmark_ifu_cube_optimization,
    benchmark_result_to_dict,
    estimate_array_nbytes,
)
from .modes import get_pipeline_name_for_mode, make_inference_pipeline
from .optimize import OptimizationResult, optimize_params
from .parameterization import (
    IdentityTransform,
    ParameterTransform,
    SigmoidBounds,
    SoftplusLowerBound,
    apply_transforms,
    build_age_metallicity_transforms,
    inverse_transforms,
)
from .validation import GradientComparison, compare_gradients, finite_difference_grad
from .variational import (
    VariationalResult,
    initialize_mean_field_params,
    kl_diag_gaussian_to_standard_normal,
    optimize_variational_posterior,
    sample_diag_gaussian,
)

__all__ = [
    "IdentityTransform",
    "ParameterTransform",
    "SigmoidBounds",
    "SoftplusLowerBound",
    "GradientComparison",
    "IFUCubeBenchmarkResult",
    "OptimizationResult",
    "VariationalResult",
    "apply_params",
    "apply_transforms",
    "build_age_metallicity_transforms",
    "benchmark_callable",
    "benchmark_ifu_cube_optimization",
    "benchmark_result_to_dict",
    "compare_gradients",
    "estimate_array_nbytes",
    "finite_difference_grad",
    "forward",
    "get_pipeline_name_for_mode",
    "inverse_transforms",
    "loss",
    "make_inference_pipeline",
    "initialize_mean_field_params",
    "kl_diag_gaussian_to_standard_normal",
    "optimize_variational_posterior",
    "optimize_params",
    "sample_diag_gaussian",
    "value_and_grad",
]
