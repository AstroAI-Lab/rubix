"""Inference helpers for gradient-based modeling workflows."""

from .api import apply_params, forward, loss, value_and_grad
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

__all__ = [
    "IdentityTransform",
    "ParameterTransform",
    "SigmoidBounds",
    "SoftplusLowerBound",
    "GradientComparison",
    "OptimizationResult",
    "apply_params",
    "apply_transforms",
    "build_age_metallicity_transforms",
    "compare_gradients",
    "finite_difference_grad",
    "forward",
    "get_pipeline_name_for_mode",
    "inverse_transforms",
    "loss",
    "make_inference_pipeline",
    "optimize_params",
    "value_and_grad",
]
