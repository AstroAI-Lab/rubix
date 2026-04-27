from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from rubix.core.data import RubixData
from rubix.utils import get_config, read_yaml

from .api import forward
from .checkpoint import (
    load_checkpoint,
    make_optimization_checkpoint,
    make_variational_checkpoint,
    save_checkpoint,
)
from .modes import InferenceMode, make_inference_pipeline
from .objective_config import build_loss_from_config
from .optimize import (
    OptimizationResult,
    OptimizationState,
    optimize_ifu_cube,
    optimize_params,
)
from .posterior_predictive import (
    compute_residual_products,
    sample_posterior_predictive_cubes,
    summarize_masked_metrics,
    summarize_predictive_cube_samples,
)
from .variational import (
    VariationalResult,
    VariationalState,
    optimize_variational_ifu_cube,
    optimize_variational_posterior,
)
from .workflows import _to_jsonable

ExperimentConfigInput = Union[str, Mapping[str, Any]]
PipelineFactory = Callable[[dict[str, Any], InferenceMode], Any]


class _PreparedPipeline:
    """Container for prepared pipeline and static data."""

    def __init__(self, pipeline: Any, static_data: RubixData):
        self.pipeline = pipeline
        self.static_data = static_data


def _load_array(path: str, key: Optional[str] = None) -> jnp.ndarray:
    """Load an array from ``.npy`` or ``.npz`` and return JAX array.

    Args:
        path (str): Array file path.
        key (Optional[str], optional): Required key for multi-array ``.npz`` files.
            Defaults to ``None``.

    Raises:
        ValueError: If the file extension is unsupported, key resolution fails,
            or the loaded value is not array-like.

    Returns:
        jnp.ndarray: Loaded array.
    """
    array_path = Path(path)
    suffix = array_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(array_path)
    elif suffix == ".npz":
        with np.load(array_path) as npz:
            if key is not None:
                if key not in npz.files:
                    raise ValueError(
                        f"key '{key}' not found in {path}; keys={npz.files}"
                    )
                arr = npz[key]
            elif len(npz.files) == 1:
                arr = npz[npz.files[0]]
            else:
                raise ValueError(
                    f"npz file {path} contains multiple arrays {npz.files}; provide a key"
                )
    else:
        raise ValueError(f"unsupported array format for {path}; expected .npy or .npz")

    return jnp.asarray(arr)


def _default_params_init(static_data: RubixData) -> dict[str, dict[str, Any]]:
    """Build default parameter initialization from present RubixData fields.

    Args:
        static_data (RubixData): Prepared static pipeline data.

    Raises:
        ValueError: If no supported default parameter fields are available.

    Returns:
        dict[str, dict[str, Any]]: Nested parameter tree for inference.
    """
    params: dict[str, dict[str, Any]] = {}

    if static_data.stars is not None:
        stars_updates: dict[str, Any] = {}
        if static_data.stars.age is not None:
            stars_updates["age"] = jnp.asarray(static_data.stars.age)
        if static_data.stars.metallicity is not None:
            stars_updates["metallicity"] = jnp.asarray(static_data.stars.metallicity)
        if len(stars_updates) > 0:
            params["stars"] = stars_updates

    if static_data.gas is not None:
        gas_updates: dict[str, Any] = {}
        if static_data.gas.metallicity is not None:
            gas_updates["metallicity"] = jnp.asarray(static_data.gas.metallicity)
        if len(gas_updates) > 0:
            params["gas"] = gas_updates

    if len(params) == 0:
        raise ValueError(
            "could not infer params_init from static_data; provide params_init_overrides"
        )

    return params


def _apply_params_overrides(
    params_init: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply nested parameter overrides to ``params_init``.

    Args:
        params_init (Mapping[str, Mapping[str, Any]]): Baseline initialization.
        overrides (Mapping[str, Mapping[str, Any]]): Nested override mapping.

    Raises:
        ValueError: If overrides are not nested mappings.

    Returns:
        dict[str, dict[str, Any]]: Updated initialization tree.
    """
    merged = {component: dict(fields) for component, fields in params_init.items()}
    for component, fields in overrides.items():
        if not isinstance(fields, Mapping):
            raise ValueError(f"params_init_overrides[{component!r}] must be a mapping")
        component_dict = merged.setdefault(component, {})
        for field, value in fields.items():
            component_dict[field] = jnp.asarray(value)
    return merged


def _is_finite_scalar(value: float) -> bool:
    """Return ``True`` when scalar value is finite.

    Args:
        value (float): Scalar to evaluate.

    Returns:
        bool: ``True`` if finite.
    """
    return bool(np.isfinite(value))


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_config_hash(config: Mapping[str, Any]) -> str:
    """Return stable SHA256 hash for a normalized experiment config."""
    json_text = json.dumps(_to_jsonable(dict(config)), sort_keys=True)
    return hashlib.sha256(json_text.encode("utf-8")).hexdigest()


def _get_git_commit_sha() -> Optional[str]:
    """Return current git commit SHA if available, else ``None``."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:  # pragma: no cover - environment-dependent
        return None
    value = output.strip()
    return value if value else None


def _default_checkpoint_interval(max_steps: int) -> int:
    """Return standard checkpoint cadence for a stage."""
    return max(1, min(200, max_steps // 5 if max_steps >= 5 else max_steps))


def _resolve_resume_checkpoint_path(
    resume_checkpoint: Optional[str],
    checkpoint_dir: Optional[str],
    stage: str,
) -> Optional[str]:
    """Resolve explicit or ``latest`` stage checkpoint path.

    Args:
        resume_checkpoint (Optional[str]): Configured checkpoint path or
            ``'latest'`` sentinel.
        checkpoint_dir (Optional[str]): Stage checkpoint directory.
        stage (str): Stage name.

    Raises:
        ValueError: If ``resume_checkpoint`` is ``'latest'`` but
            ``checkpoint_dir`` is unavailable.

    Returns:
        Optional[str]: Resolved checkpoint path if available.
    """
    if not isinstance(resume_checkpoint, str) or len(resume_checkpoint) == 0:
        return None

    if resume_checkpoint != "latest":
        return resume_checkpoint

    if checkpoint_dir is None:
        raise ValueError(
            f"{stage}.resume_checkpoint is 'latest' but run.checkpoint_dir is not set"
        )

    pattern = f"{stage}_chunk_*.pkl"
    candidates = sorted(Path(checkpoint_dir).glob(pattern))
    if len(candidates) == 0:
        return None
    return str(candidates[-1])


def normalize_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a production IFU experiment config mapping.

    Expected sections are ``run``, ``data``, ``optimization``, ``variational``,
    and ``predictive``.

    Args:
        config (Mapping[str, Any]): Raw user mapping.

    Raises:
        ValueError: If required keys are missing or malformed.

    Returns:
        dict[str, Any]: Normalized config with defaults.
    """
    run_cfg = dict(config.get("run") or {})
    data_cfg = dict(config.get("data") or {})
    opt_cfg = dict(config.get("optimization") or {})
    vi_cfg = dict(config.get("variational") or {})
    pred_cfg = dict(config.get("predictive") or {})

    rubix_config_path = run_cfg.get("rubix_config_path")
    target_path = data_cfg.get("target_path")

    if not isinstance(rubix_config_path, str) or len(rubix_config_path.strip()) == 0:
        raise ValueError("run.rubix_config_path must be a non-empty string")

    if not isinstance(target_path, str) or len(target_path.strip()) == 0:
        raise ValueError("data.target_path must be a non-empty string")

    mode = run_cfg.get("mode", "deterministic")
    if mode not in {"deterministic", "stochastic"}:
        raise ValueError("run.mode must be 'deterministic' or 'stochastic'")

    normalized = {
        "run": {
            "rubix_config_path": rubix_config_path,
            "mode": mode,
            "smoke_only": bool(run_cfg.get("smoke_only", False)),
            "auto_resume_latest": bool(run_cfg.get("auto_resume_latest", False)),
            "fail_on_stage_failure": bool(run_cfg.get("fail_on_stage_failure", False)),
            "checkpoint_policy": str(run_cfg.get("checkpoint_policy", "standard")),
            "seed": int(run_cfg.get("seed", 0)),
            "output_dir": run_cfg.get("output_dir", "outputs/ifu_science"),
            "checkpoint_dir": run_cfg.get("checkpoint_dir"),
            "params_init_overrides": dict(run_cfg.get("params_init_overrides") or {}),
            "objective": run_cfg.get("objective"),
            "noise_seed": run_cfg.get("noise_seed"),
        },
        "data": {
            "target_path": target_path,
            "target_key": data_cfg.get("target_key"),
            "mask_path": data_cfg.get("mask_path"),
            "mask_key": data_cfg.get("mask_key"),
            "weights_path": data_cfg.get("weights_path"),
            "weights_key": data_cfg.get("weights_key"),
            "sigma_path": data_cfg.get("sigma_path"),
            "sigma_key": data_cfg.get("sigma_key"),
            "inv_variance_path": data_cfg.get("inv_variance_path"),
            "inv_variance_key": data_cfg.get("inv_variance_key"),
        },
        "optimization": {
            "enabled": bool(opt_cfg.get("enabled", True)),
            "learning_rate": float(opt_cfg.get("learning_rate", 1e-3)),
            "max_steps": int(opt_cfg.get("max_steps", 400)),
            "tol": float(opt_cfg.get("tol", 1e-6)),
            "normalize_loss": bool(opt_cfg.get("normalize_loss", True)),
            "checkpoint_interval_steps": opt_cfg.get("checkpoint_interval_steps"),
            "resume_checkpoint": opt_cfg.get("resume_checkpoint"),
        },
        "variational": {
            "enabled": bool(vi_cfg.get("enabled", True)),
            "learning_rate": float(vi_cfg.get("learning_rate", 5e-3)),
            "max_steps": int(vi_cfg.get("max_steps", 400)),
            "tol": float(vi_cfg.get("tol", 1e-6)),
            "num_samples": int(vi_cfg.get("num_samples", 4)),
            "beta_kl": float(vi_cfg.get("beta_kl", 1e-3)),
            "init_log_std": float(vi_cfg.get("init_log_std", -2.0)),
            "normalize_loss": bool(vi_cfg.get("normalize_loss", True)),
            "huber_delta": vi_cfg.get("huber_delta"),
            "huber_weight": float(vi_cfg.get("huber_weight", 0.0)),
            "checkpoint_interval_steps": vi_cfg.get("checkpoint_interval_steps"),
            "resume_checkpoint": vi_cfg.get("resume_checkpoint"),
        },
        "predictive": {
            "enabled": bool(pred_cfg.get("enabled", True)),
            "num_draws": int(pred_cfg.get("num_draws", 16)),
        },
    }

    for stage_name in ["optimization", "variational"]:
        interval = normalized[stage_name]["checkpoint_interval_steps"]
        if interval is not None:
            interval_int = int(interval)
            if interval_int <= 0:
                raise ValueError(
                    f"{stage_name}.checkpoint_interval_steps must be positive if provided"
                )
            normalized[stage_name]["checkpoint_interval_steps"] = interval_int
        elif normalized["run"]["checkpoint_policy"] == "standard":
            normalized[stage_name]["checkpoint_interval_steps"] = (
                _default_checkpoint_interval(int(normalized[stage_name]["max_steps"]))
            )

    if normalized["run"]["checkpoint_policy"] not in {"standard", "manual"}:
        raise ValueError("run.checkpoint_policy must be 'standard' or 'manual'")

    return normalized


def validate_ifu_experiment_inputs(
    config: ExperimentConfigInput,
    pipeline_factory: PipelineFactory = make_inference_pipeline,
) -> dict[str, Any]:
    """Validate IFU experiment inputs and return a machine-readable report.

    Validation covers config normalization, tensor loading, tensor shape/value
    consistency, and default parameter initialization against prepared static
    Rubix data.

    Args:
        config (ExperimentConfigInput): Mapping or YAML path following the
            experiment schema consumed by :func:`normalize_experiment_config`.
        pipeline_factory (PipelineFactory, optional): Pipeline builder used to
            instantiate and prepare a Rubix pipeline. Defaults to
            :func:`make_inference_pipeline`.

    Returns:
        dict[str, Any]: Validation report with keys ``ok``, ``errors``,
        ``warnings``, ``config``, and ``shapes``.
    """
    report: dict[str, Any] = {
        "ok": False,
        "errors": [],
        "warnings": [],
        "config": None,
        "shapes": {},
    }

    try:
        raw_cfg = read_yaml(config) if isinstance(config, str) else dict(config)
        cfg = normalize_experiment_config(raw_cfg)
        report["config"] = cfg
    except Exception as exc:  # pragma: no cover - exercised indirectly
        report["errors"].append(f"config_error: {exc}")
        return report

    try:
        prepared = _prepare_pipeline(
            rubix_config_path=str(cfg["run"]["rubix_config_path"]),
            mode=cfg["run"]["mode"],
            pipeline_factory=pipeline_factory,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime setup
        report["errors"].append(f"pipeline_error: {exc}")
        return report

    try:
        tensors = _resolve_data_tensors(cfg["data"])
    except Exception as exc:
        report["errors"].append(f"data_error: {exc}")
        return report

    target = tensors["target"]
    mask = tensors["mask"]
    weights = tensors["weights"]
    sigma = tensors["sigma"]
    inv_variance = tensors["inv_variance"]

    report["shapes"] = {
        "target": tuple(target.shape),
        "mask": None if mask is None else tuple(mask.shape),
        "weights": None if weights is None else tuple(weights.shape),
        "sigma": None if sigma is None else tuple(sigma.shape),
        "inv_variance": None if inv_variance is None else tuple(inv_variance.shape),
    }

    if target.ndim != 3:
        report["errors"].append("target must be 3D")

    if sigma is not None and inv_variance is not None:
        report["errors"].append("only one of sigma or inv_variance may be provided")

    if not bool(jnp.all(jnp.isfinite(target))):
        report["errors"].append("target contains non-finite values")

    for name, tensor in {
        "mask": mask,
        "weights": weights,
        "sigma": sigma,
        "inv_variance": inv_variance,
    }.items():
        if tensor is None:
            continue
        if tensor.shape != target.shape:
            report["errors"].append(
                f"{name} shape {tuple(tensor.shape)} does not match target shape {tuple(target.shape)}"
            )
        if not bool(jnp.all(jnp.isfinite(tensor))):
            report["errors"].append(f"{name} contains non-finite values")

    if mask is not None and not bool(jnp.all(mask >= 0)):
        report["errors"].append("mask must be non-negative")

    if weights is not None and not bool(jnp.all(weights >= 0)):
        report["errors"].append("weights must be non-negative")

    if sigma is not None and not bool(jnp.all(sigma > 0)):
        report["errors"].append("sigma must be strictly positive")

    if inv_variance is not None and not bool(jnp.all(inv_variance >= 0)):
        report["errors"].append("inv_variance must be non-negative")

    try:
        params_init = _default_params_init(prepared.static_data)
        _apply_params_overrides(params_init, cfg["run"]["params_init_overrides"])
    except Exception as exc:
        report["errors"].append(f"params_error: {exc}")

    objective_cfg = cfg["run"].get("objective")
    if isinstance(objective_cfg, Mapping):
        try:
            build_loss_from_config(
                objective_config=objective_cfg,
                tensors={
                    "mask": mask,
                    "weights": weights,
                    "sigma": sigma,
                    "inv_variance": inv_variance,
                },
            )
        except Exception as exc:
            report["errors"].append(f"objective_error: {exc}")

    report["ok"] = len(report["errors"]) == 0
    return report


def _prepare_pipeline(
    rubix_config_path: str,
    mode: InferenceMode,
    pipeline_factory: PipelineFactory,
) -> _PreparedPipeline:
    """Build inference pipeline and static data from Rubix config.

    Args:
        rubix_config_path (str): Path to Rubix user config yaml.
        mode (InferenceMode): Deterministic or stochastic mode.
        pipeline_factory (PipelineFactory): Builder callable.

    Returns:
        _PreparedPipeline: Prepared container with pipeline and static data.
    """
    user_config = get_config(rubix_config_path)
    pipeline = pipeline_factory(user_config, mode=mode)
    static_data = pipeline.prepare_data()
    return _PreparedPipeline(pipeline=pipeline, static_data=static_data)


def _resolve_data_tensors(
    data_cfg: Mapping[str, Any],
) -> dict[str, Optional[jnp.ndarray]]:
    """Load target and optional cube-side tensors from configured paths.

    Args:
        data_cfg (Mapping[str, Any]): Normalized ``data`` config section.

    Returns:
        dict[str, Optional[jnp.ndarray]]: Loaded tensors.
    """

    def maybe_load(path_key: str, key_key: str) -> Optional[jnp.ndarray]:
        path_value = data_cfg.get(path_key)
        if path_value is None:
            return None
        return _load_array(path=str(path_value), key=data_cfg.get(key_key))

    target = _load_array(
        path=str(data_cfg["target_path"]),
        key=data_cfg.get("target_key"),
    )

    mask = maybe_load("mask_path", "mask_key")
    weights = maybe_load("weights_path", "weights_key")
    sigma = maybe_load("sigma_path", "sigma_key")
    inv_variance = maybe_load("inv_variance_path", "inv_variance_key")

    return {
        "target": target,
        "mask": mask,
        "weights": weights,
        "sigma": sigma,
        "inv_variance": inv_variance,
    }


def _checkpoint_path(directory: str, stage: str, chunk_idx: int) -> str:
    """Build checkpoint path for a given stage chunk.

    Args:
        directory (str): Checkpoint directory.
        stage (str): Stage name.
        chunk_idx (int): Chunk index.

    Returns:
        str: Checkpoint file path.
    """
    return str(Path(directory) / f"{stage}_chunk_{chunk_idx:04d}.pkl")


def _chunk_idx_from_path(path: str) -> int:
    """Extract the chunk index encoded in a checkpoint filename.

    Checkpoint filenames follow the pattern ``{stage}_chunk_{idx:04d}.pkl``.
    If the index cannot be parsed, 0 is returned so the caller's ``+= 1``
    logic still produces a safe (though non-contiguous) index.

    Args:
        path (str): Checkpoint file path.

    Returns:
        int: Parsed chunk index, or 0 on parse failure.
    """
    stem = Path(path).stem  # e.g. "optimization_chunk_0003"
    parts = stem.rsplit("_", 1)
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def _run_optimization_stage(
    pipeline: Any,
    params_init: Mapping[str, Mapping[str, Any]],
    static_data: RubixData,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    weights: Optional[jnp.ndarray],
    noise_key: Optional[jnp.ndarray],
    objective_loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]],
    stage_cfg: Mapping[str, Any],
    checkpoint_dir: Optional[str],
    auto_resume_latest: bool = False,
) -> tuple[Optional[OptimizationResult], Optional[OptimizationState], dict[str, Any]]:
    """Run optimization stage with optional checkpoint cadence and resume.

    Args:
        pipeline (Any): Inference pipeline with ``run_sharded``.
        params_init (Mapping[str, Mapping[str, Any]]): Initial params.
        static_data (RubixData): Static data tree.
        target (jnp.ndarray): Target IFU cube.
        mask (Optional[jnp.ndarray]): Optional mask.
        weights (Optional[jnp.ndarray]): Optional weights.
        noise_key (Optional[jnp.ndarray]): Optional stochastic key.
        objective_loss_fn (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Optional custom objective callable.
        stage_cfg (Mapping[str, Any]): Normalized optimization config.
        checkpoint_dir (Optional[str]): Optional checkpoint output directory.
        auto_resume_latest (bool, optional): If ``True`` and no explicit
            resume checkpoint is provided, attempt to resume from the latest
            stage checkpoint in ``checkpoint_dir``. Defaults to ``False``.

    Raises:
        ValueError: If ``resume_checkpoint`` exists but is not an optimization
            checkpoint payload.

    Returns:
        tuple[Optional[OptimizationResult], Optional[OptimizationState], dict[str, Any]]:
            Latest result, latest state, and stage metadata.
    """
    if not stage_cfg["enabled"]:
        return None, None, {"status": "skipped", "reason": "disabled"}

    stage_t0 = time.perf_counter()
    remaining = int(stage_cfg["max_steps"])
    interval = stage_cfg["checkpoint_interval_steps"]
    if interval is None:
        interval = remaining

    state_init: Optional[OptimizationState] = None
    current_params = {
        component: dict(fields) for component, fields in params_init.items()
    }
    latest_result: Optional[OptimizationResult] = None
    latest_state: Optional[OptimizationState] = None
    chunk_idx = 0
    steps_completed = 0

    resume_value = stage_cfg.get("resume_checkpoint")
    if auto_resume_latest and not resume_value and checkpoint_dir is not None:
        resume_value = "latest"
    resolved_resume = _resolve_resume_checkpoint_path(
        resume_checkpoint=resume_value,
        checkpoint_dir=checkpoint_dir,
        stage="optimization",
    )
    if resolved_resume is not None:
        payload = load_checkpoint(resolved_resume)
        if payload.get("kind") != "optimization":
            raise ValueError(
                "optimization.resume_checkpoint must reference optimization"
            )
        latest_result = payload["result"]
        latest_state = payload["state"]
        state_init = latest_state
        current_params = latest_result.params
        # Use cumulative steps_completed from checkpoint metadata; fall back to
        # per-chunk steps_run only if metadata is absent (legacy checkpoints).
        steps_completed = int(
            payload.get("metadata", {}).get(
                "steps_completed", latest_result.steps_run
            )
        )
        remaining = max(0, remaining - steps_completed)
        # Restore chunk_idx from the resumed filename so subsequent saves do
        # not overwrite earlier chunk files.
        chunk_idx = _chunk_idx_from_path(resolved_resume)

    while remaining > 0:
        run_steps = min(remaining, interval)
        chunk_idx += 1

        if objective_loss_fn is None:
            result, state = optimize_ifu_cube(
                pipeline=pipeline,
                params_init=current_params,
                static_data=static_data,
                target=target,
                mask=mask,
                weights=weights,
                normalize_loss=bool(stage_cfg["normalize_loss"]),
                learning_rate=float(stage_cfg["learning_rate"]),
                max_steps=int(run_steps),
                tol=float(stage_cfg["tol"]),
                noise_key=noise_key,
                state_init=state_init,
                return_state=True,
            )
        else:
            result, state = optimize_params(
                pipeline=pipeline,
                params_init=current_params,
                static_data=static_data,
                target=target,
                learning_rate=float(stage_cfg["learning_rate"]),
                max_steps=int(run_steps),
                tol=float(stage_cfg["tol"]),
                loss_fn=objective_loss_fn,
                noise_key=noise_key,
                state_init=state_init,
                return_state=True,
            )

        latest_result = result
        latest_state = state
        current_params = result.params
        state_init = state

        steps_completed += int(result.steps_run)
        remaining -= int(result.steps_run)

        if checkpoint_dir is not None:
            ckpt_payload = make_optimization_checkpoint(
                result=result,
                state=state,
                learning_rate=float(stage_cfg["learning_rate"]),
                tol=float(stage_cfg["tol"]),
                metadata={"stage": "optimization", "steps_completed": steps_completed},
            )
            save_checkpoint(
                _checkpoint_path(checkpoint_dir, "optimization", chunk_idx),
                ckpt_payload,
            )

        if not _is_finite_scalar(result.final_loss):
            return (
                latest_result,
                latest_state,
                {
                    "status": "failed",
                    "reason": "non_finite_final_loss",
                    "steps_completed": steps_completed,
                },
            )

        if result.converged or result.steps_run < run_steps:
            break

    if latest_result is None:
        return None, None, {"status": "skipped", "reason": "no_steps"}

    return (
        latest_result,
        latest_state,
        {
            "status": "completed",
            "converged": bool(latest_result.converged),
            "steps_completed": steps_completed,
            "final_loss": float(latest_result.final_loss),
            "duration_s": float(time.perf_counter() - stage_t0),
            "resume_checkpoint_used": resolved_resume,
        },
    )


def _run_variational_stage(
    pipeline: Any,
    params_init: Mapping[str, Mapping[str, Any]],
    static_data: RubixData,
    target: jnp.ndarray,
    sigma: Optional[jnp.ndarray],
    inv_variance: Optional[jnp.ndarray],
    mask: Optional[jnp.ndarray],
    noise_key: Optional[jnp.ndarray],
    objective_loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]],
    stage_cfg: Mapping[str, Any],
    checkpoint_dir: Optional[str],
    seed: int,
    auto_resume_latest: bool = False,
) -> tuple[Optional[VariationalResult], Optional[VariationalState], dict[str, Any]]:
    """Run VI stage with optional checkpoint cadence and resume.

    Args:
        pipeline (Any): Inference pipeline with ``run_sharded``.
        params_init (Mapping[str, Mapping[str, Any]]): Initial constrained params.
        static_data (RubixData): Static data tree.
        target (jnp.ndarray): Target IFU cube.
        sigma (Optional[jnp.ndarray]): Optional sigma map.
        inv_variance (Optional[jnp.ndarray]): Optional inverse variance map.
        mask (Optional[jnp.ndarray]): Optional mask.
        noise_key (Optional[jnp.ndarray]): Optional stochastic key.
        objective_loss_fn (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Optional custom objective callable.
        stage_cfg (Mapping[str, Any]): Normalized variational config.
        checkpoint_dir (Optional[str]): Optional checkpoint output directory.
        seed (int): Random seed.
        auto_resume_latest (bool, optional): If ``True`` and no explicit
            resume checkpoint is provided, attempt to resume from the latest
            stage checkpoint in ``checkpoint_dir``. Defaults to ``False``.

    Raises:
        ValueError: If ``resume_checkpoint`` exists but is not a variational
            checkpoint payload.

    Returns:
        tuple[Optional[VariationalResult], Optional[VariationalState], dict[str, Any]]:
            Latest result, latest state, and stage metadata.
    """
    if not stage_cfg["enabled"]:
        return None, None, {"status": "skipped", "reason": "disabled"}

    stage_t0 = time.perf_counter()
    remaining = int(stage_cfg["max_steps"])
    interval = stage_cfg["checkpoint_interval_steps"]
    if interval is None:
        interval = remaining

    state_init: Optional[VariationalState] = None
    current_params = {
        component: dict(fields) for component, fields in params_init.items()
    }
    latest_result: Optional[VariationalResult] = None
    latest_state: Optional[VariationalState] = None
    chunk_idx = 0
    steps_completed = 0

    resume_value = stage_cfg.get("resume_checkpoint")
    if auto_resume_latest and not resume_value and checkpoint_dir is not None:
        resume_value = "latest"
    resolved_resume = _resolve_resume_checkpoint_path(
        resume_checkpoint=resume_value,
        checkpoint_dir=checkpoint_dir,
        stage="variational",
    )
    if resolved_resume is not None:
        payload = load_checkpoint(resolved_resume)
        if payload.get("kind") != "variational":
            raise ValueError("variational.resume_checkpoint must reference variational")
        latest_result = payload["result"]
        latest_state = payload["state"]
        state_init = latest_state
        current_params = latest_result.posterior_mean_constrained_params
        # Use cumulative steps_completed from checkpoint metadata; fall back to
        # per-chunk steps_run only if metadata is absent (legacy checkpoints).
        steps_completed = int(
            payload.get("metadata", {}).get(
                "steps_completed", latest_result.steps_run
            )
        )
        remaining = max(0, remaining - steps_completed)
        # Restore chunk_idx from the resumed filename so subsequent saves do
        # not overwrite earlier chunk files.
        chunk_idx = _chunk_idx_from_path(resolved_resume)

    while remaining > 0:
        run_steps = min(remaining, interval)
        chunk_idx += 1

        if objective_loss_fn is None:
            result, state = optimize_variational_ifu_cube(
                pipeline=pipeline,
                params_init=current_params,
                static_data=static_data,
                target=target,
                sigma=sigma,
                inv_variance=inv_variance,
                mask=mask,
                normalize_loss=bool(stage_cfg["normalize_loss"]),
                huber_delta=stage_cfg["huber_delta"],
                huber_weight=float(stage_cfg["huber_weight"]),
                learning_rate=float(stage_cfg["learning_rate"]),
                max_steps=int(run_steps),
                tol=float(stage_cfg["tol"]),
                num_samples=int(stage_cfg["num_samples"]),
                beta_kl=float(stage_cfg["beta_kl"]),
                init_log_std=float(stage_cfg["init_log_std"]),
                noise_key=noise_key,
                seed=int(seed),
                state_init=state_init,
                return_state=True,
            )
        else:
            result, state = optimize_variational_posterior(
                pipeline=pipeline,
                params_init=current_params,
                static_data=static_data,
                target=target,
                learning_rate=float(stage_cfg["learning_rate"]),
                max_steps=int(run_steps),
                tol=float(stage_cfg["tol"]),
                num_samples=int(stage_cfg["num_samples"]),
                beta_kl=float(stage_cfg["beta_kl"]),
                init_log_std=float(stage_cfg["init_log_std"]),
                loss_fn=objective_loss_fn,
                noise_key=noise_key,
                seed=int(seed),
                state_init=state_init,
                return_state=True,
            )

        latest_result = result
        latest_state = state
        current_params = result.posterior_mean_constrained_params
        state_init = state

        steps_completed += int(result.steps_run)
        remaining -= int(result.steps_run)

        if checkpoint_dir is not None:
            ckpt_payload = make_variational_checkpoint(
                result=result,
                state=state,
                learning_rate=float(stage_cfg["learning_rate"]),
                tol=float(stage_cfg["tol"]),
                num_samples=int(stage_cfg["num_samples"]),
                beta_kl=float(stage_cfg["beta_kl"]),
                init_log_std=float(stage_cfg["init_log_std"]),
                seed=int(seed),
                metadata={"stage": "variational", "steps_completed": steps_completed},
            )
            save_checkpoint(
                _checkpoint_path(checkpoint_dir, "variational", chunk_idx), ckpt_payload
            )

        if not _is_finite_scalar(result.final_objective):
            return (
                latest_result,
                latest_state,
                {
                    "status": "failed",
                    "reason": "non_finite_final_objective",
                    "steps_completed": steps_completed,
                },
            )

        if result.converged or result.steps_run < run_steps:
            break

    if latest_result is None:
        return None, None, {"status": "skipped", "reason": "no_steps"}

    return (
        latest_result,
        latest_state,
        {
            "status": "completed",
            "converged": bool(latest_result.converged),
            "steps_completed": steps_completed,
            "final_objective": float(latest_result.final_objective),
            "duration_s": float(time.perf_counter() - stage_t0),
            "resume_checkpoint_used": resolved_resume,
        },
    )


def run_ifu_experiment(
    config: ExperimentConfigInput,
    pipeline_factory: PipelineFactory = make_inference_pipeline,
) -> dict[str, Any]:
    """Run a production IFU inference experiment from config.

    This orchestration supports deterministic/stochastic pipeline mode, objective
    config selection, optimization and VI stages, posterior predictive outputs,
    periodic checkpoints, and resume from stage checkpoints.

    Args:
        config (ExperimentConfigInput): Mapping or YAML path following the
            experiment schema consumed by :func:`normalize_experiment_config`.
        pipeline_factory (PipelineFactory, optional): Pipeline builder used to
            instantiate and prepare a Rubix pipeline. Defaults to
            :func:`make_inference_pipeline`.

    Raises:
        ValueError: If incompatible smoke-only uncertainty settings are
            provided (both ``sigma`` and ``inv_variance``).
        RuntimeError: If stage failures occur and
            ``run.fail_on_stage_failure`` is enabled.

    Returns:
        dict[str, Any]: Structured outputs including stage summaries, optional
        predictive outputs, residual maps, and scalar metrics.
    """
    run_started_at = _utc_now_iso()
    run_t0 = time.perf_counter()
    raw_cfg = read_yaml(config) if isinstance(config, str) else dict(config)
    cfg = normalize_experiment_config(raw_cfg)

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]

    prepared = _prepare_pipeline(
        rubix_config_path=str(run_cfg["rubix_config_path"]),
        mode=run_cfg["mode"],
        pipeline_factory=pipeline_factory,
    )

    tensors = _resolve_data_tensors(data_cfg)
    target = tensors["target"]
    mask = tensors["mask"]
    weights = tensors["weights"]
    sigma = tensors["sigma"]
    inv_variance = tensors["inv_variance"]

    params_init = _default_params_init(prepared.static_data)
    params_init = _apply_params_overrides(params_init, run_cfg["params_init_overrides"])

    objective_cfg = run_cfg.get("objective")
    objective_loss_fn = None
    if isinstance(objective_cfg, Mapping):
        objective_loss_fn = build_loss_from_config(
            objective_config=objective_cfg,
            tensors={
                "mask": mask,
                "weights": weights,
                "sigma": sigma,
                "inv_variance": inv_variance,
            },
        )

    noise_key = None
    if run_cfg["mode"] == "stochastic":
        noise_seed = run_cfg.get("noise_seed")
        if noise_seed is None:
            noise_seed = run_cfg["seed"]
        noise_key = jax.random.PRNGKey(int(noise_seed))

    checkpoint_dir = run_cfg.get("checkpoint_dir")
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    smoke_only = bool(run_cfg["smoke_only"])
    auto_resume_latest = bool(run_cfg["auto_resume_latest"])

    if smoke_only:
        opt_result = None
        vi_result = None
        opt_status = {"status": "skipped", "reason": "smoke_only"}
        vi_status = {"status": "skipped", "reason": "smoke_only"}
    else:
        opt_result, _, opt_status = _run_optimization_stage(
            pipeline=prepared.pipeline,
            params_init=params_init,
            static_data=prepared.static_data,
            target=target,
            mask=mask,
            weights=weights,
            noise_key=noise_key,
            objective_loss_fn=objective_loss_fn,
            stage_cfg=cfg["optimization"],
            checkpoint_dir=checkpoint_dir,
            auto_resume_latest=auto_resume_latest,
        )

        vi_params_init = params_init
        if opt_result is not None and opt_status.get("status") == "completed":
            vi_params_init = opt_result.params

        vi_result, _, vi_status = _run_variational_stage(
            pipeline=prepared.pipeline,
            params_init=vi_params_init,
            static_data=prepared.static_data,
            target=target,
            sigma=sigma,
            inv_variance=inv_variance,
            mask=mask,
            noise_key=noise_key,
            objective_loss_fn=objective_loss_fn,
            stage_cfg=cfg["variational"],
            checkpoint_dir=checkpoint_dir,
            seed=int(run_cfg["seed"]),
            auto_resume_latest=auto_resume_latest,
        )

    failed_stages = {
        name: stage
        for name, stage in {
            "optimization": opt_status,
            "variational": vi_status,
        }.items()
        if stage.get("status") == "failed"
    }

    outputs: dict[str, Any] = {
        "config": cfg,
        "run_metadata": {
            "started_at_utc": run_started_at,
            "finished_at_utc": None,
            "run_duration_s": None,
            "git_commit_sha": _get_git_commit_sha(),
            "config_hash_sha256": _stable_config_hash(cfg),
        },
        "stages": {
            "optimization": opt_status,
            "variational": vi_status,
        },
        "failure_artifacts": None,
        "optimization": None,
        "variational": None,
        "predictive_summary": None,
        "residual_products": None,
        "metrics": None,
    }

    if opt_result is not None:
        outputs["optimization"] = {
            "final_loss": float(opt_result.final_loss),
            "best_loss": float(opt_result.best_loss),
            "steps_run": int(opt_result.steps_run),
            "converged": bool(opt_result.converged),
        }

    if vi_result is not None:
        outputs["variational"] = {
            "final_objective": float(vi_result.final_objective),
            "best_objective": float(vi_result.best_objective),
            "steps_run": int(vi_result.steps_run),
            "converged": bool(vi_result.converged),
            "final_reconstruction": float(vi_result.final_reconstruction),
            "final_kl": float(vi_result.final_kl),
        }

    if smoke_only:
        if sigma is not None and inv_variance is not None:
            raise ValueError(
                "smoke_only: provide only one of data.sigma_path or"
                " data.inv_variance_path, not both"
            )
        smoke_prediction = forward(
            pipeline=prepared.pipeline,
            params=params_init,
            static_data=prepared.static_data,
            noise_key=noise_key,
        )
        residual_products = compute_residual_products(
            prediction=smoke_prediction,
            target=target,
            sigma=sigma,
            inv_variance=inv_variance,
            mask=mask,
        )
        metrics = summarize_masked_metrics(
            prediction=smoke_prediction,
            target=target,
            mask=mask,
            loss_fn=objective_loss_fn,
        )
        outputs["residual_products"] = residual_products
        outputs["metrics"] = metrics
    elif bool(cfg["predictive"]["enabled"]) and vi_result is not None:
        predictive_samples = sample_posterior_predictive_cubes(
            pipeline=prepared.pipeline,
            posterior_mean_params=vi_result.posterior_mean_params,
            posterior_log_std_params=vi_result.posterior_log_std_params,
            static_data=prepared.static_data,
            num_samples=int(cfg["predictive"]["num_draws"]),
            noise_key=noise_key,
            seed=int(run_cfg["seed"]) + 1,
        )
        predictive_summary = summarize_predictive_cube_samples(predictive_samples)
        residual_products = compute_residual_products(
            prediction=predictive_summary["mean"],
            target=target,
            sigma=sigma,
            inv_variance=inv_variance,
            mask=mask,
        )
        metrics = summarize_masked_metrics(
            prediction=predictive_summary["mean"],
            target=target,
            mask=mask,
            loss_fn=objective_loss_fn,
        )

        outputs["predictive_summary"] = predictive_summary
        outputs["residual_products"] = residual_products
        outputs["metrics"] = metrics

    if len(failed_stages) > 0:
        outputs["failure_artifacts"] = {
            "failed_stages": failed_stages,
            "guidance": (
                "Inspect stage status reasons and resume from checkpoints after "
                "adjusting learning rate / objective settings."
            ),
        }

    outputs["run_metadata"]["finished_at_utc"] = _utc_now_iso()
    outputs["run_metadata"]["run_duration_s"] = float(time.perf_counter() - run_t0)

    if len(failed_stages) > 0 and bool(run_cfg["fail_on_stage_failure"]):
        raise RuntimeError(
            f"IFU experiment failed stages: {sorted(failed_stages.keys())}"
        )

    return outputs


def save_ifu_experiment_outputs(outputs: Mapping[str, Any], output_dir: str) -> None:
    """Persist production IFU experiment outputs to JSON and NPZ artifacts.

    Args:
        outputs (Mapping[str, Any]): Outputs returned by
            :func:`run_ifu_experiment`.
        output_dir (str): Destination output directory.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": outputs.get("config"),
        "run_metadata": outputs.get("run_metadata"),
        "stages": outputs.get("stages"),
        "optimization": outputs.get("optimization"),
        "variational": outputs.get("variational"),
        "metrics": outputs.get("metrics"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2),
        encoding="utf-8",
    )

    predictive_summary = outputs.get("predictive_summary")
    if isinstance(predictive_summary, Mapping):
        np.savez(
            out_dir / "predictive_summary.npz",
            **{k: np.asarray(v) for k, v in predictive_summary.items()},
        )

    residual_products = outputs.get("residual_products")
    if isinstance(residual_products, Mapping):
        np.savez(
            out_dir / "residual_products.npz",
            **{k: np.asarray(v) for k, v in residual_products.items()},
        )

    failure_artifacts = outputs.get("failure_artifacts")
    if isinstance(failure_artifacts, Mapping):
        (out_dir / "failure_report.json").write_text(
            json.dumps(_to_jsonable(dict(failure_artifacts)), indent=2),
            encoding="utf-8",
        )
