from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference.experiment import (
    compare_science_run_to_baseline,
    create_science_run_baseline,
    run_ifu_experiment_sequence,
)


class PreparedSyntheticPipeline:
    """Synthetic pipeline for sequence/baseline cycle tests."""

    def __init__(self, template: jnp.ndarray):
        self.template = template

    def prepare_data(self) -> RubixData:
        return RubixData(
            galaxy=Galaxy(),
            stars=StarsData(
                coords=jnp.zeros((1, 3)),
                velocity=jnp.zeros((1, 3)),
                mass=jnp.ones(1),
                age=jnp.array([0.2]),
                metallicity=jnp.array([0.01]),
            ),
            gas=GasData(
                coords=jnp.zeros((1, 3)),
                velocity=jnp.zeros((1, 3)),
                mass=jnp.ones(1),
            ),
        )

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        return rubixdata.stars.age[0] * self.template


def _write_config(tmp_path: Path, target_path: str, checkpoint_dir: str) -> Path:
    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
            "seed": 0,
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": checkpoint_dir,
            "params_init_overrides": {"stars": {"age": [0.2]}},
        },
        "data": {
            "target_path": target_path,
            "mask_path": str(tmp_path / "mask.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {
            "enabled": True,
            "learning_rate": 0.1,
            "max_steps": 20,
            "tol": 1e-8,
            "checkpoint_interval_steps": 10,
        },
        "variational": {
            "enabled": True,
            "learning_rate": 0.05,
            "max_steps": 20,
            "tol": 1e-8,
            "num_samples": 2,
            "beta_kl": 1e-4,
            "checkpoint_interval_steps": 10,
        },
        "predictive": {
            "enabled": True,
            "num_draws": 3,
        },
    }

    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")
    config_path = tmp_path / "experiment.yml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def test_small_science_cycle_sequence_create_compare(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    config_path = _write_config(tmp_path, str(target_path), str(tmp_path / "ckpt"))

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
