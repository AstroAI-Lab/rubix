#!/usr/bin/env python
import argparse
import json

import jax.numpy as jnp

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference.benchmark import (
    benchmark_ifu_cube_optimization,
    benchmark_result_to_dict,
)


class SyntheticScalePipeline:
    """Simple synthetic pipeline for benchmarking optimizer overhead."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full-IFU cube optimization runtime and memory diagnostics.",
    )
    parser.add_argument("--nx", type=int, default=25, help="Cube x-size")
    parser.add_argument("--ny", type=int, default=25, help="Cube y-size")
    parser.add_argument("--nw", type=int, default=128, help="Spectral bins")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats")
    parser.add_argument("--max-steps", type=int, default=100, help="Optimizer steps")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-2, help="Adam learning rate"
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable untimed warmup run",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Benchmark with a central-region voxel mask",
    )
    parser.add_argument(
        "--use-weights",
        action="store_true",
        help="Benchmark with simple inverse-scale-like weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cube_shape = (args.nx, args.ny, args.nw)
    template = jnp.ones(cube_shape, dtype=jnp.float32)
    target_scale = 1.7
    target = target_scale * template

    mask = None
    if args.use_mask:
        mask = jnp.zeros(cube_shape, dtype=jnp.float32)
        x0 = args.nx // 4
        x1 = 3 * args.nx // 4
        y0 = args.ny // 4
        y1 = 3 * args.ny // 4
        mask = mask.at[x0:x1, y0:y1, :].set(1.0)

    weights = None
    if args.use_weights:
        spectral_axis = jnp.linspace(1.0, 2.0, args.nw, dtype=jnp.float32)
        weights = jnp.broadcast_to(1.0 / spectral_axis, cube_shape)

    pipeline = SyntheticScalePipeline(template)
    static_data = _make_static_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}

    result = benchmark_ifu_cube_optimization(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        mask=mask,
        weights=weights,
        normalize_loss=True,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        tol=args.tol,
        repeats=args.repeats,
        warmup=not args.no_warmup,
    )

    print(json.dumps(benchmark_result_to_dict(result), indent=2))


if __name__ == "__main__":
    main()
