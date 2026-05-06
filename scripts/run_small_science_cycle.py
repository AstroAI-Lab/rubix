#!/usr/bin/env python
import argparse
import json
import sys
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
        required=False,
        default=None,
        type=str,
        help="Baseline JSON path for create/compare. Required when --create-baseline or --compare-baseline is set.",
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

    if (args.create_baseline or args.compare_baseline) and not args.baseline_path:
        raise SystemExit(
            "--baseline-path is required when using --create-baseline or --compare-baseline"
        )

    try:
        sequence = run_ifu_experiment_sequence(
            config=args.config,
            output_root_dir=args.output_root_dir,
            run_validate=not args.skip_validate,
            run_smoke=not args.skip_smoke,
            run_full=not args.skip_full,
            include_outputs=False,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    full_meta = sequence.get("full")
    full_output_dir = None if full_meta is None else full_meta.get("output_dir")

    baseline_result = None
    compare_result = None

    if args.create_baseline:
        if full_output_dir is None:
            raise SystemExit("cannot create baseline without full phase outputs")
        baseline_result = create_science_run_baseline(
            output_dir=full_output_dir,
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
        tolerances = parse_tolerances(args.tolerance)
        compare_result = compare_science_run_to_baseline(
            output_dir=full_output_dir,
            baseline_path=args.baseline_path,
            tolerances=tolerances,
        )
        if not compare_result["passed"]:
            print(json.dumps(compare_result, indent=2), file=sys.stderr)
            raise SystemExit(1)

    # Print a lightweight, machine-readable summary; detailed per-phase
    # artifacts (summary.json, validate_report.json, etc.) are on disk.
    summary = {
        "output_root_dir": sequence.get("output_root_dir"),
        "validate_ok": (sequence.get("validate") or {}).get("ok"),
        "smoke_output_dir": (sequence.get("smoke") or {}).get("output_dir"),
        "full_output_dir": (sequence.get("full") or {}).get("output_dir"),
        "baseline_created": baseline_result is not None,
        "baseline_compared": compare_result is not None,
        "compare_passed": (
            compare_result["passed"] if compare_result is not None else None
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
