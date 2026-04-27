#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from rubix.inference.experiment import generate_ifu_experiment_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact JSON report from saved IFU experiment outputs."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional custom path for report JSON (defaults to <output-dir>/science_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_ifu_experiment_report(args.output_dir)

    if args.report_path is None:
        report_path = Path(args.output_dir) / "science_report.json"
    else:
        report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
