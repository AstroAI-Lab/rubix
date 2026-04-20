"""Inference helpers for gradient-based modeling workflows."""

from .api import apply_params, forward, loss, value_and_grad
from .modes import get_pipeline_name_for_mode, make_inference_pipeline
from .objectives import build_ifu_cube_loss, masked_weighted_mse
from .optimize import OptimizationResult, optimize_ifu_cube, optimize_params
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
    "OptimizationResult",
    "VariationalResult",
    "apply_params",
    "apply_transforms",
    "build_age_metallicity_transforms",
    "compare_gradients",
    "finite_difference_grad",
    "forward",
    "get_pipeline_name_for_mode",
    "build_ifu_cube_loss",
    "inverse_transforms",
    "loss",
    "make_inference_pipeline",
    "masked_weighted_mse",
    "initialize_mean_field_params",
    "kl_diag_gaussian_to_standard_normal",
    "optimize_ifu_cube",
    "optimize_variational_posterior",
    "optimize_params",
    "sample_diag_gaussian",
    "value_and_grad",
]
