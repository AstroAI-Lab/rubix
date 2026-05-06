import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rubix.inference.experiment import (
    compare_science_run_to_baseline,
    create_science_run_baseline,
    run_ifu_experiment_sequence,
)

from tests._helpers import PreparedSyntheticPipeline, write_experiment_config

_SCRIPT = Path(__file__).parent.parent / "scripts" / "run_small_science_cycle.py"


def test_small_science_cycle_sequence_create_compare(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    config_path = write_experiment_config(
        tmp_path, str(target_path), str(tmp_path / "ckpt"),
        max_steps=40, checkpoint_interval_steps=20, num_draws=4,
    )

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    sequence = run_ifu_experiment_sequence(
        config=str(config_path),
        pipeline_factory=pipeline_factory,
        output_root_dir=str(tmp_path / "sequence"),
    )
    full_output_dir = sequence["full"]["output_dir"]
    baseline_path = tmp_path / "baseline.json"

    create_science_run_baseline(full_output_dir, str(baseline_path))
    assert baseline_path.exists()

    compare_result = compare_science_run_to_baseline(
        output_dir=full_output_dir,
        baseline_path=str(baseline_path),
        tolerances={"mse": 1e-6, "final_objective": 1e-6},
    )
    assert compare_result["passed"] is True


def test_cli_missing_baseline_path_exits(tmp_path):
    """CLI exits with a non-zero code when --create-baseline is given without --baseline-path."""
    # The guard fires before config loading, so the config file need not exist.
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--config",
            str(tmp_path / "experiment.yml"),
            "--output-root-dir",
            str(tmp_path / "out"),
            "--create-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--baseline-path" in result.stderr


def test_cli_sequence_only_exits_0(tmp_path):
    """CLI exits 0 and prints a JSON summary when all phases are skipped and no baseline ops."""
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    config_path = write_experiment_config(
        tmp_path, str(target_path), str(tmp_path / "ckpt"),
        max_steps=40, checkpoint_interval_steps=20, num_draws=4,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--config",
            str(config_path),
            "--output-root-dir",
            str(tmp_path / "out"),
            "--skip-validate",
            "--skip-smoke",
            "--skip-full",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert "validate_ok" in summary
    assert "full_output_dir" in summary
    assert summary["baseline_created"] is False
    assert summary["baseline_compared"] is False
