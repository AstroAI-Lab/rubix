#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    OptimizationObjectiveThresholds,
    RuntimeThresholds,
    benchmark_ifu_cube_optimization,
    benchmark_result_to_dict,
    check_ifu_optimization_guardrails,
)


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
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nw", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-mean-runtime-s", type=float, default=3.0)
    parser.add_argument("--max-median-runtime-s", type=float, default=3.0)
    parser.add_argument("--max-final-loss", type=float, default=1e-3)
    parser.add_argument("--max-best-loss", type=float, default=1e-3)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cube = jnp.ones((args.nx, args.ny, args.nw), dtype=jnp.float32)
    target = 1.5 * cube

    benchmark_result = benchmark_ifu_cube_optimization(
        pipeline=_SyntheticPipeline(cube),
        params_init={"stars": {"age": jnp.array([0.2])}},
        static_data=_make_data(),
        target=target,
        learning_rate=0.1,
        max_steps=args.max_steps,
        tol=1e-8,
        repeats=args.repeats,
        warmup=True,
    )

    runtime_thresholds = RuntimeThresholds(
        max_mean_runtime_s=args.max_mean_runtime_s,
        max_median_runtime_s=args.max_median_runtime_s,
    )
    objective_thresholds = OptimizationObjectiveThresholds(
        max_final_loss=args.max_final_loss,
        max_best_loss=args.max_best_loss,
    )

    check = check_ifu_optimization_guardrails(
        benchmark_result,
        runtime_thresholds,
        objective_thresholds,
    )

    payload = {
        "benchmark": benchmark_result_to_dict(benchmark_result),
        "guardrail": {
            "passed": check.passed,
            "message": check.message,
            "failed_conditions": check.failed_conditions,
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
