from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference.checkpoint import (
    make_optimization_checkpoint,
    save_checkpoint,
)
from rubix.inference.experiment import (
    generate_ifu_experiment_report,
    normalize_experiment_config,
    run_ifu_experiment,
    save_ifu_experiment_outputs,
    validate_ifu_experiment_inputs,
)
from rubix.inference.optimize import OptimizationResult, OptimizationState


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


class PreparedNaNPipeline(PreparedSyntheticPipeline):
    """Synthetic pipeline that returns NaNs to force stage failure artifacts."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        return jnp.full_like(self.template, jnp.nan)


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
            "max_steps": 40,
            "tol": 1e-8,
            "checkpoint_interval_steps": 20,
        },
        "variational": {
            "enabled": True,
            "learning_rate": 0.05,
            "max_steps": 40,
            "tol": 1e-8,
            "num_samples": 2,
            "beta_kl": 1e-4,
            "checkpoint_interval_steps": 20,
        },
        "predictive": {
            "enabled": True,
            "num_draws": 4,
        },
    }

    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")
    config_path = tmp_path / "experiment.yml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def test_normalize_experiment_config_applies_defaults():
    normalized = normalize_experiment_config(
        {
            "run": {"rubix_config_path": "conf.yml"},
            "data": {"target_path": "target.npy"},
        }
    )

    assert normalized["run"]["mode"] == "deterministic"
    assert normalized["optimization"]["enabled"] is True
    assert normalized["variational"]["enabled"] is True
    assert normalized["predictive"]["num_draws"] == 16
    assert normalized["optimization"]["checkpoint_interval_steps"] == 80
    assert normalized["variational"]["checkpoint_interval_steps"] == 80
    assert normalized["run"]["checkpoint_policy"] == "standard"


def test_run_ifu_experiment_and_save_outputs(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target = 1.5 * cube
    mask = np.ones_like(target)
    ivar = np.ones_like(target)

    target_path = tmp_path / "target.npy"
    np.save(target_path, target)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "ivar.npy", ivar)

    checkpoint_dir = str(tmp_path / "checkpoints")
    config_path = _write_config(tmp_path, str(target_path), checkpoint_dir)

    def pipeline_factory(_cfg, mode):
        assert mode == "deterministic"
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    outputs = run_ifu_experiment(str(config_path), pipeline_factory=pipeline_factory)

    assert outputs["stages"]["optimization"]["status"] == "completed"
    assert outputs["stages"]["variational"]["status"] == "completed"
    assert outputs["run_metadata"]["config_hash_sha256"] is not None
    assert outputs["run_metadata"]["started_at_utc"] is not None
    assert outputs["run_metadata"]["finished_at_utc"] is not None
    assert outputs["run_metadata"]["run_duration_s"] is not None
    assert outputs["metrics"]["mse"] >= 0.0
    assert outputs["predictive_summary"]["mean"].shape == (2, 2, 4)

    checkpoint_files = list((tmp_path / "checkpoints").glob("*.pkl"))
    assert len(checkpoint_files) >= 2

    save_ifu_experiment_outputs(outputs, str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "summary.json").exists()
    assert (tmp_path / "saved" / "predictive_summary.npz").exists()
    assert (tmp_path / "saved" / "residual_products.npz").exists()
    assert (tmp_path / "saved" / "science_products.npz").exists()
    assert (tmp_path / "saved" / "science_metrics.csv").exists()

    report = generate_ifu_experiment_report(str(tmp_path / "saved"))
    assert "summary" in report
    assert "artifacts" in report
    assert "science_products" in report["artifacts"]
    assert "posterior_mean_cube" in report["artifacts"]["science_products"]


def test_run_ifu_experiment_generates_failure_artifacts(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")

    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
            "checkpoint_dir": str(tmp_path / "ckpt"),
        },
        "data": {
            "target_path": str(target_path),
            "mask_path": str(tmp_path / "mask.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {"enabled": True, "max_steps": 3, "learning_rate": 0.1},
        "variational": {"enabled": False},
        "predictive": {"enabled": False},
    }

    def pipeline_factory(_cfg, _mode):
        return PreparedNaNPipeline(jnp.asarray(cube))

    outputs = run_ifu_experiment(cfg, pipeline_factory=pipeline_factory)
    assert outputs["stages"]["optimization"]["status"] == "failed"
    assert outputs["failure_artifacts"] is not None
    assert "optimization" in outputs["failure_artifacts"]["failed_stages"]

    save_ifu_experiment_outputs(outputs, str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "failure_report.json").exists()


def test_normalize_experiment_config_rejects_invalid_checkpoint_interval():
    try:
        normalize_experiment_config(
            {
                "run": {"rubix_config_path": "conf.yml"},
                "data": {"target_path": "target.npy"},
                "optimization": {"checkpoint_interval_steps": 0},
            }
        )
    except ValueError as exc:
        assert "checkpoint_interval_steps" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive checkpoint interval")


def test_run_ifu_experiment_rejects_mismatched_resume_checkpoint_kind(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)

    (tmp_path / "rubix_user.yml").write_text("pipeline:\\n  name: calc_gradient\\n")
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))

    opt_result = OptimizationResult(
        params={"stars": {"age": jnp.array([0.2])}},
        best_params={"stars": {"age": jnp.array([0.2])}},
        loss_history=[1.0],
        grad_norm_history=[1.0],
        best_loss=1.0,
        final_loss=1.0,
        steps_run=1,
        converged=False,
    )
    opt_state = OptimizationState(
        trainable_params={"stars": {"age": jnp.array([0.2])}},
        opt_state=None,
        best_trainable_params={"stars": {"age": jnp.array([0.2])}},
        best_loss=1.0,
    )
    checkpoint_path = tmp_path / "opt.pkl"
    save_checkpoint(
        str(checkpoint_path),
        make_optimization_checkpoint(opt_result, opt_state),
    )

    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
        },
        "data": {
            "target_path": str(target_path),
            "mask_path": str(tmp_path / "mask.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {"enabled": False},
        "variational": {
            "enabled": True,
            "resume_checkpoint": str(checkpoint_path),
        },
        "predictive": {"enabled": False},
    }

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    try:
        run_ifu_experiment(cfg, pipeline_factory=pipeline_factory)
    except ValueError as exc:
        assert "variational.resume_checkpoint" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched checkpoint kind")


def test_validate_ifu_experiment_inputs_reports_ok(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    config_path = _write_config(tmp_path, str(target_path), str(tmp_path / "ckpt"))

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    report = validate_ifu_experiment_inputs(
        str(config_path),
        pipeline_factory=pipeline_factory,
    )
    assert report["ok"] is True
    assert report["errors"] == []
    assert tuple(report["shapes"]["target"]) == (2, 2, 4)


def test_validate_ifu_experiment_inputs_detects_shape_mismatch(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones((2, 2, 5), dtype=np.float32))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    config_path = _write_config(tmp_path, str(target_path), str(tmp_path / "ckpt"))

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    report = validate_ifu_experiment_inputs(
        str(config_path),
        pipeline_factory=pipeline_factory,
    )
    assert report["ok"] is False
    assert any("mask shape" in error for error in report["errors"])


def test_validate_ifu_experiment_inputs_detects_bad_objective_kind(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")

    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
            "objective": {"kind": "unsupported_kind"},
        },
        "data": {
            "target_path": str(target_path),
            "mask_path": str(tmp_path / "mask.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {"enabled": False},
        "variational": {"enabled": False},
        "predictive": {"enabled": False},
    }

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    report = validate_ifu_experiment_inputs(cfg, pipeline_factory=pipeline_factory)
    assert report["ok"] is False
    assert any("objective_error" in error for error in report["errors"])


def test_validate_ifu_experiment_inputs_detects_missing_tensor_key(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")

    # mask is loaded (so tensors["mask"] is present), but mask_key references a
    # key that does not exist in the tensors mapping → objective_error expected
    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
            "objective": {"kind": "mse", "mask_key": "nonexistent_key"},
        },
        "data": {
            "target_path": str(target_path),
            "mask_path": str(tmp_path / "mask.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {"enabled": False},
        "variational": {"enabled": False},
        "predictive": {"enabled": False},
    }

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    report = validate_ifu_experiment_inputs(cfg, pipeline_factory=pipeline_factory)
    assert report["ok"] is False
    assert any("objective_error" in error for error in report["errors"])


def test_validate_ifu_experiment_inputs_valid_objective_passes(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "mask.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")

    # Use mask_key pointing at the actual loaded mask tensor
    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
            "objective": {"kind": "mse", "mask_key": "mask"},
        },
        "data": {
            "target_path": str(target_path),
            "mask_path": str(tmp_path / "mask.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {"enabled": False},
        "variational": {"enabled": False},
        "predictive": {"enabled": False},
    }

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    report = validate_ifu_experiment_inputs(cfg, pipeline_factory=pipeline_factory)
    assert report["ok"] is True
    assert report["errors"] == []


def test_smoke_only_rejects_both_sigma_and_inv_variance(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target_path = tmp_path / "target.npy"
    np.save(target_path, cube)
    np.save(tmp_path / "sigma.npy", np.ones_like(cube))
    np.save(tmp_path / "ivar.npy", np.ones_like(cube))
    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")

    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
            "smoke_only": True,
        },
        "data": {
            "target_path": str(target_path),
            "sigma_path": str(tmp_path / "sigma.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {"enabled": False},
        "variational": {"enabled": False},
        "predictive": {"enabled": False},
    }

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    try:
        run_ifu_experiment(cfg, pipeline_factory=pipeline_factory)
    except ValueError as exc:
        assert "smoke_only" in str(exc)
        assert "sigma" in str(exc)
        assert "inv_variance" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when both sigma and inv_variance provided in smoke_only mode"
        )


def test_run_ifu_experiment_smoke_only_computes_metrics(tmp_path):
    cube = np.ones((2, 2, 4), dtype=np.float32)
    target = 1.5 * cube
    target_path = tmp_path / "target.npy"
    np.save(target_path, target)
    np.save(tmp_path / "mask.npy", np.ones_like(target))
    np.save(tmp_path / "ivar.npy", np.ones_like(target))

    cfg = {
        "run": {
            "rubix_config_path": str(tmp_path / "rubix_user.yml"),
            "mode": "deterministic",
            "smoke_only": True,
        },
        "data": {
            "target_path": str(target_path),
            "mask_path": str(tmp_path / "mask.npy"),
            "inv_variance_path": str(tmp_path / "ivar.npy"),
        },
        "optimization": {"enabled": False},
        "variational": {"enabled": False},
        "predictive": {"enabled": False},
    }
    (tmp_path / "rubix_user.yml").write_text("pipeline:\n  name: calc_gradient\n")

    def pipeline_factory(_cfg, _mode):
        return PreparedSyntheticPipeline(jnp.asarray(cube))

    outputs = run_ifu_experiment(cfg, pipeline_factory=pipeline_factory)
    assert outputs["stages"]["optimization"]["reason"] == "smoke_only"
    assert outputs["stages"]["variational"]["reason"] == "smoke_only"
    assert outputs["predictive_summary"] is None
    assert outputs["metrics"] is not None
    assert outputs["residual_products"] is not None

    save_ifu_experiment_outputs(outputs, str(tmp_path / "saved_smoke"))
    with np.load(tmp_path / "saved_smoke" / "science_products.npz") as science_products:
        assert "residual_cube" in science_products.files
        assert "chi2_cube" in science_products.files


def test_generate_ifu_experiment_report_requires_summary(tmp_path):
    try:
        generate_ifu_experiment_report(str(tmp_path))
    except FileNotFoundError as exc:
        assert "summary.json" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when summary.json is missing")
