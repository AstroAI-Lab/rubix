#!/usr/bin/env python
import argparse
import json

from rubix.inference.experiment import (
    run_ifu_experiment,
    save_ifu_experiment_outputs,
    validate_ifu_experiment_inputs,
)


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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config/inputs only and exit without running inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        report = validate_ifu_experiment_inputs(args.config)
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
        return

    outputs = run_ifu_experiment(args.config)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = outputs["config"]["run"]["output_dir"]

    save_ifu_experiment_outputs(outputs, output_dir)


if __name__ == "__main__":
    main()
