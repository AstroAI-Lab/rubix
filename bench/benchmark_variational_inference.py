#!/usr/bin/env python
import argparse
import json

import jax.numpy as jnp

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference.vi_benchmark import (
    benchmark_variational_inference,
    vi_benchmark_result_to_dict,
)


class SyntheticScalePipeline:
    """Simple synthetic pipeline for VI benchmark runs."""

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
        description="Benchmark full-IFU variational inference runtime and diagnostics.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=25,
        help="Number of spatial pixels along the x axis.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=25,
        help="Number of spatial pixels along the y axis.",
    )
    parser.add_argument(
        "--nw",
        type=int,
        default=128,
        help="Number of wavelength bins in the synthetic IFU cube.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of benchmark repetitions to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum number of optimization steps per run.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of variational samples used per optimization step.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-2,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--beta-kl",
        type=float,
        default=1e-3,
        help="Weight applied to the KL-divergence term.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for early stopping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible benchmark runs.",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Apply a central spatial mask to the synthetic target cube.",
    )
    parser.add_argument(
        "--use-huber",
        action="store_true",
        help="Add a Huber loss term to the Gaussian loss.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.2,
        help="Delta threshold for the Huber loss.",
    )
    parser.add_argument(
        "--huber-weight",
        type=float,
        default=0.1,
        help="Weight assigned to the Huber loss term.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable any warmup run before timing benchmark repetitions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cube_shape = (args.nx, args.ny, args.nw)
    template = jnp.ones(cube_shape, dtype=jnp.float32)
    target = 1.7 * template

    mask = None
    if args.use_mask:
        mask = jnp.zeros(cube_shape, dtype=jnp.float32)
        x0 = args.nx // 4
        x1 = 3 * args.nx // 4
        y0 = args.ny // 4
        y1 = 3 * args.ny // 4
        mask = mask.at[x0:x1, y0:y1, :].set(1.0)

    pipeline = SyntheticScalePipeline(template)
    static_data = _make_static_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}

    result = benchmark_variational_inference(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        sigma=jnp.ones_like(target),
        mask=mask,
        huber_delta=args.huber_delta if args.use_huber else None,
        huber_weight=args.huber_weight if args.use_huber else 0.0,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        tol=args.tol,
        num_samples=args.num_samples,
        beta_kl=args.beta_kl,
        seed=args.seed,
        repeats=args.repeats,
        warmup=not args.no_warmup,
    )

    print(json.dumps(vi_benchmark_result_to_dict(result), indent=2))


if __name__ == "__main__":
    main()
