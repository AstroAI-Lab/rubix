import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "run_vi_calibration.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_vi_calibration", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(path: Path, seed: int, n_particles=4, n_samples=200, sigma=1.0):
    """Write a science-cycle-like npz with a calibrated Gaussian posterior."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, size=n_particles)
    truths = rng.normal(means, sigma)
    # (S, P) posterior samples, matching the science-cycle npz layout.
    samples = rng.normal(means[None, :], sigma, size=(n_samples, n_particles))
    np.savez(
        path,
        true_age=truths,
        true_metallicity=truths * 1e-3,
        true_vz=truths * 10.0,
        post_samples_age=samples,
        post_samples_metallicity=samples * 1e-3,
        post_samples_vz=samples * 10.0,
    )


def test_compute_calibration_report_calibrated(tmp_path):
    module = _load_module()
    paths = []
    for seed in range(40):
        p = tmp_path / f"run_{seed}.npz"
        _write_run(p, seed)
        paths.append(p)

    report = module.compute_calibration_report(
        paths, levels=(0.5, 0.68, 0.9, 0.95), sbc_bins=None
    )
    assert report["n_files"] == 40
    age = report["parameters"]["age"]
    assert age["n_trials"] == 40 * 4
    idx_90 = age["levels"].index(0.9)
    # Calibrated construction -> coverage near nominal within sampling noise.
    assert abs(age["empirical_coverage"][idx_90] - 0.9) < 0.1
    assert age["rms_z"] < 1.4


def test_pool_parameter_rejects_mismatched_sample_count(tmp_path):
    module = _load_module()
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    _write_run(a, 0, n_samples=100)
    _write_run(b, 1, n_samples=120)
    files = [np.load(a), np.load(b)]
    try:
        with pytest.raises(ValueError, match="share the posterior sample count"):
            module._pool_parameter(files, "true_age", "post_samples_age")
    finally:
        for f in files:
            f.close()


def test_resolve_npz_expands_directory(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "run0"
    run_dir.mkdir()
    _write_run(run_dir / "science_cycle_outputs.npz", 0)
    assert module._resolve_npz(run_dir).name == "science_cycle_outputs.npz"
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        module._resolve_npz(empty_dir)


def test_cli_writes_summary_json(tmp_path):
    for seed in range(6):
        _write_run(tmp_path / f"run_{seed}.npz", seed)
    out = tmp_path / "calibration_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            *[str(tmp_path / f"run_{seed}.npz") for seed in range(6)],
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert set(payload["parameters"]) == {"age", "metallicity", "vz"}
    assert payload["n_files"] == 6
