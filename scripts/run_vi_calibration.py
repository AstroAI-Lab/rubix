#!/usr/bin/env python
"""Compute posterior calibration diagnostics across repeated VI runs.

This is the Phase 4 calibration gate of ``docs/vi_science_validation_plan.md``:
it turns a collection of ``science_cycle_outputs.npz`` files (one per seed /
replication, each carrying posterior parameter samples and the known truth) into
credible-interval coverage and simulation-based-calibration (SBC) statistics.

Each input file must contain ``true_{age,metallicity,vz}`` (shape ``(P,)``) and
``post_samples_{age,metallicity,vz}`` (shape ``(S, P)``), produced by
``scripts/run_realistic_synthetic_vi_cycle.py``. Every ``(file, particle)`` pair
becomes one calibration trial; all files must share the posterior sample count
``S`` (the particle count ``P`` may differ between files).

Interpretation:

- Empirical coverage close to the nominal level ⇒ calibrated intervals.
  Consistently *below* nominal ⇒ over-confident (too-narrow) posteriors.
- ``rms_z`` near 1 and ``mean_z`` near 0 ⇒ well-scaled Gaussian errors.
- ``sbc_reduced_chi2`` near 1 ⇒ uniform SBC ranks. Large values flag bias
  (skewed ranks) or dispersion errors (∪- / ∩-shaped rank histograms).

Note: for SBC to be strictly valid the truths must vary across replications
(e.g. a sampled population per seed). With a fixed deterministic truth this
still measures coverage of that truth under repeated noise/init realizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rubix.inference import summarize_calibration

_PARAMS = (
    ("age", "true_age", "post_samples_age"),
    ("metallicity", "true_metallicity", "post_samples_metallicity"),
    ("vz", "true_vz", "post_samples_vz"),
)


def _resolve_npz(path: Path) -> Path:
    """Return the npz path, expanding a directory to its science-cycle output."""
    if path.is_dir():
        candidate = path / "science_cycle_outputs.npz"
        if not candidate.exists():
            raise FileNotFoundError(
                f"{path} does not contain science_cycle_outputs.npz"
            )
        return candidate
    return path


def _pool_parameter(
    files: list[np.lib.npyio.NpzFile],
    truth_key: str,
    sample_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool (S, P) samples and (P,) truths across files into (T, S) and (T,).

    Args:
        files (list[np.lib.npyio.NpzFile]): Loaded npz files.
        truth_key (str): Key of the truth array within each file.
        sample_key (str): Key of the posterior-sample array within each file.

    Raises:
        KeyError: If a required key is missing from any file.
        ValueError: If the posterior sample count differs across files.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pooled ``(T, S)`` samples and ``(T,)``
        truths, where ``T`` is the total number of pooled trials.
    """
    samples_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    n_samples: int | None = None
    for f in files:
        if truth_key not in f or sample_key not in f:
            raise KeyError(
                f"file missing '{truth_key}' or '{sample_key}'; regenerate with "
                "an updated run_realistic_synthetic_vi_cycle.py"
            )
        truth = np.asarray(f[truth_key]).reshape(-1)  # (P,)
        samples = np.asarray(f[sample_key])  # (S, P)
        if samples.ndim != 2 or samples.shape[1] != truth.shape[0]:
            raise ValueError(
                f"'{sample_key}' shape {samples.shape} incompatible with "
                f"'{truth_key}' shape {truth.shape}"
            )
        if n_samples is None:
            n_samples = samples.shape[0]
        elif samples.shape[0] != n_samples:
            raise ValueError(
                "all files must share the posterior sample count S; got "
                f"{samples.shape[0]} and {n_samples}"
            )
        samples_blocks.append(samples.T)  # (P, S) -> P trials
        truth_blocks.append(truth)
    return np.concatenate(samples_blocks, axis=0), np.concatenate(truth_blocks)


def compute_calibration_report(
    input_paths: list[Path],
    levels: tuple[float, ...],
    sbc_bins: int | None,
) -> dict:
    """Build the full calibration report dict from input npz paths."""
    resolved = [_resolve_npz(Path(p)) for p in input_paths]
    files = [np.load(p) for p in resolved]
    try:
        parameters: dict[str, dict] = {}
        for name, truth_key, sample_key in _PARAMS:
            samples, truths = _pool_parameter(files, truth_key, sample_key)
            summary = summarize_calibration(
                jnp.asarray(samples),
                jnp.asarray(truths),
                levels=levels,
                sbc_num_bins=sbc_bins,
            )
            parameters[name] = summary.to_dict()
    finally:
        for f in files:
            f.close()

    return {
        "inputs": [str(p) for p in resolved],
        "n_files": len(resolved),
        "levels": list(levels),
        "parameters": parameters,
    }


def _print_summary(report: dict) -> None:
    print(f"\nCalibration over {report['n_files']} run(s):")
    levels = report["levels"]
    header = "  param        trials  " + "  ".join(f"cov@{lv:g}" for lv in levels)
    header += "   rms_z  sbc_rchi2"
    print(header)
    for name, s in report["parameters"].items():
        cov = "  ".join(f"{c:6.3f}" for c in s["empirical_coverage"])
        print(
            f"  {name:<11} {s['n_trials']:>6}  {cov}   "
            f"{s['rms_z']:5.2f}   {s['sbc_reduced_chi2']:7.2f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="npz files or directories containing science_cycle_outputs.npz",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write calibration_summary.json",
    )
    parser.add_argument(
        "--levels",
        default="0.5,0.68,0.9,0.95",
        help="Comma-separated nominal central-interval levels in (0, 1)",
    )
    parser.add_argument(
        "--sbc-bins",
        type=int,
        default=None,
        help="SBC rank histogram bins (default: auto)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    levels = tuple(float(x) for x in args.levels.split(",") if x.strip())
    report = compute_calibration_report(
        [Path(p) for p in args.inputs], levels, args.sbc_bins
    )
    _print_summary(report)
    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
