#!/usr/bin/env python
import argparse
import json

from rubix.inference.experiment import validate_ifu_experiment_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate IFU experiment config and data tensors before expensive runs."
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_ifu_experiment_inputs(args.config)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
