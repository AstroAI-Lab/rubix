#!/usr/bin/env python
import argparse
import json

from rubix.inference.experiment import (
    compare_science_run_to_baseline,
    create_science_run_baseline,
)


def parse_tolerances(items: list[str]) -> dict[str, float]:
    tolerances: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid tolerance '{item}'; expected format key=value")
        key, value = item.split("=", 1)
        tolerances[key.strip()] = float(value)
    return tolerances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or compare small-science run baselines."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create", help="Create baseline snapshot from saved run output."
    )
    create_parser.add_argument("--output-dir", required=True, type=str)
    create_parser.add_argument("--baseline-path", required=True, type=str)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare saved run output to baseline snapshot."
    )
    compare_parser.add_argument("--output-dir", required=True, type=str)
    compare_parser.add_argument("--baseline-path", required=True, type=str)
    compare_parser.add_argument(
        "--tolerance",
        action="append",
        default=[],
        help=(
            "Tolerance override in key=value format, e.g. --tolerance mse=1e-4 "
            "--tolerance final_objective=1e-3"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "create":
        baseline = create_science_run_baseline(
            output_dir=args.output_dir,
            baseline_path=args.baseline_path,
        )
        print(json.dumps(baseline, indent=2))
        return

    tolerances = parse_tolerances(args.tolerance)
    result = compare_science_run_to_baseline(
        output_dir=args.output_dir,
        baseline_path=args.baseline_path,
        tolerances=tolerances,
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
