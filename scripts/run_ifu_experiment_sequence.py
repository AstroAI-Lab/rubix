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
    # Print a lightweight summary; per-phase details are in the on-disk files
    # (validate_report.json, smoke/summary.json, full/summary.json).
    summary = {
        "output_root_dir": result.get("output_root_dir"),
        "validate_ok": (result.get("validate") or {}).get("ok"),
        "smoke_output_dir": (result.get("smoke") or {}).get("output_dir"),
        "full_output_dir": (result.get("full") or {}).get("output_dir"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
