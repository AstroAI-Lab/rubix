#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from rubix.inference.experiment import (
    compare_science_run_to_baseline,
    create_science_run_baseline,
    run_ifu_experiment_sequence,
)


def parse_tolerances(items: list[str]) -> dict[str, float]:
    """Parse CLI tolerances in key=value form."""
    tolerances: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid tolerance '{item}'; expected format key=value")
        key, value = item.split("=", 1)
        tolerances[key.strip()] = float(value)
    return tolerances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full small-science cycle: validate->smoke->full, then "
            "baseline create/compare."
        )
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-root-dir", required=True, type=str)
    parser.add_argument(
        "--baseline-path",
        required=True,
        type=str,
        help="Baseline JSON path for create/compare.",
    )
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create/update baseline from the produced full run outputs.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare produced full run against baseline and fail on regression.",
    )
    parser.add_argument(
        "--tolerance",
        action="append",
        default=[],
        help="Tolerance in key=value format, e.g. --tolerance mse=1e-4",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validation phase.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke phase.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full phase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tolerances = parse_tolerances(args.tolerance)

    sequence = run_ifu_experiment_sequence(
        config=args.config,
        output_root_dir=args.output_root_dir,
        run_validate=not args.skip_validate,
        run_smoke=not args.skip_smoke,
        run_full=not args.skip_full,
        include_outputs=False,
    )

    full_meta = sequence.get("full")
    full_output_dir = None if full_meta is None else full_meta.get("output_dir")

    baseline_result = None
    compare_result = None

    if args.create_baseline:
        if full_output_dir is None:
            raise SystemExit("cannot create baseline without full phase outputs")
        baseline_result = create_science_run_baseline(
            output_dir=str(full_output_dir),
            baseline_path=args.baseline_path,
        )

    if args.compare_baseline:
        if full_output_dir is None:
            raise SystemExit("cannot compare baseline without full phase outputs")
        baseline_path = Path(args.baseline_path)
        if not baseline_path.exists():
            raise SystemExit(
                f"baseline file does not exist for comparison: {baseline_path}"
            )
        compare_result = compare_science_run_to_baseline(
            output_dir=str(full_output_dir),
            baseline_path=args.baseline_path,
            tolerances=tolerances,
        )
        if not compare_result["passed"]:
            print(json.dumps(compare_result, indent=2))
            raise SystemExit(1)

    payload = {
        "sequence": sequence,
        "baseline_created": baseline_result is not None,
        "baseline_compared": compare_result is not None,
        "compare_result": compare_result,
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
