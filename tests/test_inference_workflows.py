import json

import numpy as np

from rubix.inference.workflows import (
    run_synthetic_science_recipe,
    save_science_recipe_outputs,
)


def test_run_synthetic_science_recipe_returns_expected_structure():
    outputs = run_synthetic_science_recipe(
        cube_shape=(2, 2, 4),
        target_scale=1.5,
        optimize_steps=20,
        vi_steps=20,
        num_vi_samples=2,
        num_posterior_draws=4,
        seed=0,
    )

    assert "config" in outputs
    assert "optimization" in outputs
    assert "variational" in outputs
    assert "predictive_summary" in outputs
    assert "residual_products" in outputs
    assert "metrics" in outputs

    assert outputs["predictive_summary"]["mean"].shape == (2, 2, 4)
    assert outputs["residual_products"]["residual"].shape == (2, 2, 4)
    assert outputs["metrics"]["mse"] >= 0.0
    assert outputs["metrics"]["mae"] >= 0.0


def test_save_science_recipe_outputs_writes_json_and_npz(tmp_path):
    outputs = run_synthetic_science_recipe(
        cube_shape=(2, 2, 4),
        target_scale=1.2,
        optimize_steps=10,
        vi_steps=10,
        num_vi_samples=2,
        num_posterior_draws=3,
        seed=1,
    )

    save_science_recipe_outputs(outputs, str(tmp_path))

    summary_path = tmp_path / "summary.json"
    predictive_path = tmp_path / "predictive_summary.npz"
    residual_path = tmp_path / "residual_products.npz"

    assert summary_path.exists()
    assert predictive_path.exists()
    assert residual_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["config"]["cube_shape"] == [2, 2, 4]
    assert "final_loss" in summary["optimization"]
    assert "final_objective" in summary["variational"]

    predictive = np.load(predictive_path)
    residual = np.load(residual_path)

    assert predictive["mean"].shape == (2, 2, 4)
    assert residual["residual"].shape == (2, 2, 4)
