#!/usr/bin/env python
import argparse
import json

from rubix.inference.experiment import run_ifu_experiment_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IFU workflow sequence: validate -> smoke -> full."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default=None,
        help="Optional root directory for validate/smoke/full outputs.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validation phase.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke-only phase.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full optimization/VI phase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ifu_experiment_sequence(
        config=args.config,
        run_validate=not args.skip_validate,
        run_smoke=not args.skip_smoke,
        run_full=not args.skip_full,
        output_root_dir=args.output_root_dir,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
