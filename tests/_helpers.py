"""Shared test helpers for IFU experiment tests."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import yaml

from rubix.core.data import Galaxy, GasData, RubixData, StarsData


class PreparedSyntheticPipeline:
    """Synthetic pipeline implementing prepare_data + run_sharded."""

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
        scale = rubixdata.stars.age[0]
        return scale * self.template


def write_experiment_config(
    tmp_path: Path,
    target_path: str,
    checkpoint_dir: str,
    *,
    max_steps: int = 40,
    checkpoint_interval_steps: int = 20,
    num_draws: int = 4,
) -> Path:
    """Write a minimal experiment YAML config for tests.

    Args:
        tmp_path (Path): Temporary directory for config and auxiliary files.
        target_path (str): Path to the target data ``.npy`` file.
        checkpoint_dir (str): Directory where checkpoints will be written.
        max_steps (int): Number of steps for both optimization and variational
            phases.
        checkpoint_interval_steps (int): Steps between checkpoint saves (applies
            to both optimization and variational phases).
        num_draws (int): Number of posterior-predictive draws.

    Returns:
        Path to the written experiment YAML file.
    """
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
            "max_steps": max_steps,
            "tol": 1e-8,
            "checkpoint_interval_steps": checkpoint_interval_steps,
        },
        "variational": {
            "enabled": True,
            "learning_rate": 0.05,
            "max_steps": max_steps,
            "tol": 1e-8,
            "num_samples": 2,
            "beta_kl": 1e-4,
            "checkpoint_interval_steps": checkpoint_interval_steps,
        },
        "predictive": {
            "enabled": True,
            "num_draws": num_draws,
        },
    }

    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")
    config_path = tmp_path / "experiment.yml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path
