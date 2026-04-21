from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rubix.core.data import Galaxy, GasData, RubixData, StarsData

from .optimize import optimize_ifu_cube
from .posterior_predictive import (
    compute_residual_products,
    sample_posterior_predictive_cubes,
    summarize_masked_metrics,
    summarize_predictive_cube_samples,
)
from .variational import optimize_variational_ifu_cube


class SyntheticScalePipeline:
    """Simple synthetic IFU pipeline used by the science recipe workflow."""

    def __init__(self, template: jnp.ndarray):
        self.template = template

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        scale = rubixdata.stars.age[0]
        return scale * self.template


def _make_static_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
            age=jnp.array([0.0]),
            metallicity=jnp.array([0.01]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def _to_jsonable(value: Any) -> Any:
    """Convert nested JAX/NumPy containers into JSON-serializable values."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, jnp.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_synthetic_science_recipe(
    cube_shape: tuple[int, int, int] = (4, 4, 16),
    target_scale: float = 1.7,
    optimize_steps: int = 120,
    vi_steps: int = 120,
    num_vi_samples: int = 4,
    num_posterior_draws: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a compact end-to-end synthetic science workflow.

    This workflow performs deterministic optimization, variational inference,
    posterior predictive sampling, and residual/metric summarization.

    Args:
        cube_shape (tuple[int, int, int], optional): Synthetic IFU cube shape
            ``(nx, ny, nw)``. Defaults to ``(4, 4, 16)``.
        target_scale (float, optional): Multiplicative scale for the synthetic
            target cube. Defaults to 1.7.
        optimize_steps (int, optional): Maximum deterministic optimization
            steps. Defaults to 120.
        vi_steps (int, optional): Maximum variational optimization steps.
            Defaults to 120.
        num_vi_samples (int, optional): Monte Carlo samples per VI step.
            Defaults to 4.
        num_posterior_draws (int, optional): Number of posterior predictive
            cube draws. Defaults to 8.
        seed (int, optional): Random seed for VI and predictive sampling.
            Defaults to 0.

    Returns:
        dict[str, Any]: Workflow outputs and diagnostic summaries.
    """
    template = jnp.ones(cube_shape, dtype=jnp.float32)
    target = target_scale * template

    pipeline = SyntheticScalePipeline(template)
    static_data = _make_static_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}

    opt_result = optimize_ifu_cube(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=0.1,
        max_steps=optimize_steps,
        tol=1e-8,
    )

    vi_result = optimize_variational_ifu_cube(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        sigma=jnp.ones_like(target),
        learning_rate=5e-2,
        max_steps=vi_steps,
        tol=1e-8,
        num_samples=num_vi_samples,
        beta_kl=1e-4,
        seed=seed,
    )

    predictive_samples = sample_posterior_predictive_cubes(
        pipeline=pipeline,
        posterior_mean_params=vi_result.posterior_mean_params,
        posterior_log_std_params=vi_result.posterior_log_std_params,
        static_data=static_data,
        num_samples=num_posterior_draws,
        seed=seed + 1,
    )
    predictive_summary = summarize_predictive_cube_samples(predictive_samples)
    residual_products = compute_residual_products(
        prediction=predictive_summary["mean"],
        target=target,
    )
    metrics = summarize_masked_metrics(
        prediction=predictive_summary["mean"],
        target=target,
    )

    return {
        "config": {
            "cube_shape": cube_shape,
            "target_scale": target_scale,
            "optimize_steps": optimize_steps,
            "vi_steps": vi_steps,
            "num_vi_samples": num_vi_samples,
            "num_posterior_draws": num_posterior_draws,
            "seed": seed,
        },
        "optimization": {
            "final_loss": opt_result.final_loss,
            "best_loss": opt_result.best_loss,
            "steps_run": opt_result.steps_run,
            "converged": opt_result.converged,
        },
        "variational": {
            "final_objective": vi_result.final_objective,
            "best_objective": vi_result.best_objective,
            "steps_run": vi_result.steps_run,
            "converged": vi_result.converged,
        },
        "predictive_summary": predictive_summary,
        "residual_products": residual_products,
        "metrics": metrics,
    }


def save_science_recipe_outputs(
    outputs: dict[str, Any],
    output_dir: str,
) -> None:
    """Persist workflow outputs to JSON and NPZ files.

    Args:
        outputs (dict[str, Any]): Outputs from
            :func:`run_synthetic_science_recipe`.
        output_dir (str): Destination directory.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": outputs["config"],
        "optimization": outputs["optimization"],
        "variational": outputs["variational"],
        "metrics": outputs["metrics"],
    }
    json_summary = _to_jsonable(summary)
    (out_dir / "summary.json").write_text(
        json.dumps(json_summary, indent=2), encoding="utf-8"
    )

    predictive_np = {k: np.asarray(v) for k, v in outputs["predictive_summary"].items()}
    residual_np = {k: np.asarray(v) for k, v in outputs["residual_products"].items()}

    np.savez(out_dir / "predictive_summary.npz", **predictive_np)
    np.savez(out_dir / "residual_products.npz", **residual_np)
