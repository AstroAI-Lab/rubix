#!/usr/bin/env python
import argparse

from rubix.inference.experiment import run_ifu_experiment, save_ifu_experiment_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a production IFU inference experiment from YAML config."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config.run.output_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_ifu_experiment(args.config)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = outputs["config"]["run"]["output_dir"]

    save_ifu_experiment_outputs(outputs, output_dir)


if __name__ == "__main__":
    main()
