#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import jax.numpy as jnp

from rubix.config.config import PARENT_DIR
from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    OptimizationObjectiveThresholds,
    RuntimeThresholds,
    benchmark_ifu_cube_optimization,
    benchmark_result_to_dict,
    check_ifu_optimization_guardrails,
    load_guardrail_threshold_profile,
)
from rubix.utils import read_yaml


class _SyntheticPipeline:
    """Small synthetic pipeline for guardrail smoke checks."""

    def __init__(self, template: jnp.ndarray):
        self.template = template

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        return rubixdata.stars.age[0] * self.template


def _make_data() -> RubixData:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optimization benchmark and assert runtime/loss guardrails."
    )
    parser.add_argument(
        "--threshold-config",
        type=str,
        default=str(Path(PARENT_DIR) / "guardrail_thresholds.yml"),
    )
    parser.add_argument("--profile", type=str, default="small_cube_default")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--max-mean-runtime-s", type=float, default=None)
    parser.add_argument("--max-median-runtime-s", type=float, default=None)
    parser.add_argument("--max-final-loss", type=float, default=None)
    parser.add_argument("--max-best-loss", type=float, default=None)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def _resolve_with_profile(
    override: Optional[float],
    profile_mapping: Mapping[str, Any],
    key: str,
) -> Optional[float]:
    """Resolve value from CLI override or profile mapping."""
    if override is not None:
        return override
    value = profile_mapping.get(key)
    return None if value is None else float(value)


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.threshold_config)
    if not isinstance(cfg, Mapping):
        raise SystemExit(
            f"threshold config {args.threshold_config} must contain a top-level mapping"
        )
    profiles = cfg.get("profiles") or {}
    if not isinstance(profiles, Mapping):
        raise SystemExit(
            f"threshold config {args.threshold_config} field 'profiles' must be a mapping"
        )
    profile = profiles.get(args.profile)
    if not isinstance(profile, Mapping):
        raise SystemExit(
            f"profile '{args.profile}' not found in {args.threshold_config}"
        )
    benchmark_profile = profile.get("benchmark") or {}
    if not isinstance(benchmark_profile, Mapping):
        raise SystemExit(f"profile '{args.profile}' benchmark section must be mapping")

    nx = int(_resolve_with_profile(args.nx, benchmark_profile, "nx") or 8)
    ny = int(_resolve_with_profile(args.ny, benchmark_profile, "ny") or 8)
    nw = int(_resolve_with_profile(args.nw, benchmark_profile, "nw") or 64)
    max_steps = int(
        _resolve_with_profile(args.max_steps, benchmark_profile, "max_steps") or 120
    )
    repeats = int(
        _resolve_with_profile(args.repeats, benchmark_profile, "repeats") or 2
    )

    cube = jnp.ones((nx, ny, nw), dtype=jnp.float32)
    target = 1.5 * cube

    benchmark_result = benchmark_ifu_cube_optimization(
        pipeline=_SyntheticPipeline(cube),
        params_init={"stars": {"age": jnp.array([0.2])}},
        static_data=_make_data(),
        target=target,
        learning_rate=0.1,
        max_steps=max_steps,
        tol=1e-8,
        repeats=repeats,
        warmup=True,
    )

    runtime_thresholds, objective_thresholds = load_guardrail_threshold_profile(
        config_path=args.threshold_config,
        profile_name=args.profile,
        mode="optimization",
    )
    runtime_thresholds = RuntimeThresholds(
        max_mean_runtime_s=_resolve_with_profile(
            args.max_mean_runtime_s,
            {"max_mean_runtime_s": runtime_thresholds.max_mean_runtime_s},
            "max_mean_runtime_s",
        ),
        max_median_runtime_s=_resolve_with_profile(
            args.max_median_runtime_s,
            {"max_median_runtime_s": runtime_thresholds.max_median_runtime_s},
            "max_median_runtime_s",
        ),
    )
    objective_thresholds = OptimizationObjectiveThresholds(
        max_final_loss=_resolve_with_profile(
            args.max_final_loss,
            {"max_final_loss": objective_thresholds.max_final_loss},
            "max_final_loss",
        ),
        max_best_loss=_resolve_with_profile(
            args.max_best_loss,
            {"max_best_loss": objective_thresholds.max_best_loss},
            "max_best_loss",
        ),
    )

    check = check_ifu_optimization_guardrails(
        benchmark_result,
        runtime_thresholds,
        objective_thresholds,
    )

    payload = {
        "profile": {
            "name": args.profile,
            "threshold_config": args.threshold_config,
            "benchmark": {
                "nx": nx,
                "ny": ny,
                "nw": nw,
                "max_steps": max_steps,
                "repeats": repeats,
            },
        },
        "benchmark": benchmark_result_to_dict(benchmark_result),
        "guardrail": {
            "passed": check.passed,
            "message": check.message,
            "failed_conditions": list(check.failed_conditions),
        },
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not check.passed:
        raise SystemExit(check.message)


if __name__ == "__main__":
    main()
