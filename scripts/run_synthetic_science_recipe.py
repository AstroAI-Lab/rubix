#!/usr/bin/env python
import argparse

from rubix.inference.workflows import (
    run_synthetic_science_recipe,
    save_science_recipe_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic end-to-end science workflow and save outputs."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/science_recipe")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nw", type=int, default=16)
    parser.add_argument("--target-scale", type=float, default=1.7)
    parser.add_argument("--optimize-steps", type=int, default=120)
    parser.add_argument("--vi-steps", type=int, default=120)
    parser.add_argument("--num-vi-samples", type=int, default=4)
    parser.add_argument("--num-posterior-draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_synthetic_science_recipe(
        cube_shape=(args.nx, args.ny, args.nw),
        target_scale=args.target_scale,
        optimize_steps=args.optimize_steps,
        vi_steps=args.vi_steps,
        num_vi_samples=args.num_vi_samples,
        num_posterior_draws=args.num_posterior_draws,
        seed=args.seed,
    )
    save_science_recipe_outputs(outputs, args.output_dir)


if __name__ == "__main__":
    main()
