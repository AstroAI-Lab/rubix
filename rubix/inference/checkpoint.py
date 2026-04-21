import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

import jax.numpy as jnp
import optax
from beartype.typing import Union

from rubix.core.data import RubixData

from .api import LossFn
from .optimize import OptimizationResult, OptimizationState, optimize_params
from .parameterization import TransformTree
from .variational import (
    VariationalResult,
    VariationalState,
    optimize_variational_posterior,
)

CheckpointPath = Union[str, Path]


def save_checkpoint(path: CheckpointPath, payload: Mapping[str, Any]) -> None:
    """Save a checkpoint payload to disk via pickle.

    Args:
        path (CheckpointPath): Destination checkpoint path.
        payload (Mapping[str, Any]): Serializable checkpoint mapping.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as f:
        pickle.dump(dict(payload), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: CheckpointPath) -> dict[str, Any]:
    """Load a checkpoint payload from disk.

    .. warning::
        This function uses :mod:`pickle`, which can execute arbitrary code when
        loading data from untrusted files.  Only load checkpoints from sources
        you trust.

    Args:
        path (CheckpointPath): Checkpoint path.

    Raises:
        ValueError: If the loaded payload is not a dictionary.

    Returns:
        dict[str, Any]: Loaded checkpoint mapping.
    """
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload


def make_optimization_checkpoint(
    result: OptimizationResult,
    state: OptimizationState,
    transforms: Optional[TransformTree] = None,
    learning_rate: float = 1e-3,
    tol: float = 1e-6,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a checkpoint payload for optimization resume.

    The supplied configuration (``transforms``, ``learning_rate``, ``tol``) is
    stored in the checkpoint so that :func:`resume_optimization_from_checkpoint`
    can fall back to these values when they are not re-supplied by the caller,
    making resumes reproducible without requiring the caller to re-specify every
    hyperparameter.

    Args:
        result (OptimizationResult): Result from the completed optimization run.
        state (OptimizationState): Internal state from the completed run.
        transforms (Optional[TransformTree], optional): Transform tree used
            during the run.  Stored so that :func:`resume_optimization_from_checkpoint`
            can default to it on resume.  Defaults to ``None``.
        learning_rate (float, optional): Learning rate used during the run.
            Defaults to 1e-3.
        tol (float, optional): Convergence tolerance used during the run.
            Defaults to 1e-6.
        metadata (Optional[Mapping[str, Any]], optional): Arbitrary extra
            metadata to embed in the checkpoint.  Defaults to ``None``.

    Returns:
        dict[str, Any]: Serializable checkpoint mapping.
    """
    config = {
        "transforms": transforms,
        "learning_rate": learning_rate,
        "tol": tol,
    }
    return {
        "kind": "optimization",
        "result": result,
        "state": state,
        "config": config,
        "metadata": dict(metadata or {}),
    }


def make_variational_checkpoint(
    result: VariationalResult,
    state: VariationalState,
    transforms: Optional[TransformTree] = None,
    learning_rate: float = 5e-3,
    tol: float = 1e-6,
    num_samples: int = 4,
    beta_kl: float = 1e-3,
    init_log_std: float = -2.0,
    seed: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a checkpoint payload for variational inference resume.

    The supplied configuration is stored so that
    :func:`resume_variational_from_checkpoint` can fall back to these values
    when they are not re-supplied by the caller.

    Args:
        result (VariationalResult): Result from the completed VI run.
        state (VariationalState): Internal state from the completed run.
        transforms (Optional[TransformTree], optional): Transform tree used
            during the run.  Defaults to ``None``.
        learning_rate (float, optional): Learning rate used during the run.
            Defaults to 5e-3.
        tol (float, optional): Convergence tolerance used during the run.
            Defaults to 1e-6.
        num_samples (int, optional): MC samples per step used during the run.
            Defaults to 4.
        beta_kl (float, optional): KL weight used during the run.
            Defaults to 1e-3.
        init_log_std (float, optional): Initial posterior log-std used at the
            start of the run.  Defaults to -2.0.
        seed (int, optional): Random seed used at the start of the run.
            Defaults to 0.
        metadata (Optional[Mapping[str, Any]], optional): Arbitrary extra
            metadata.  Defaults to ``None``.

    Returns:
        dict[str, Any]: Serializable checkpoint mapping.
    """
    config = {
        "transforms": transforms,
        "learning_rate": learning_rate,
        "tol": tol,
        "num_samples": num_samples,
        "beta_kl": beta_kl,
        "init_log_std": init_log_std,
        "seed": seed,
    }
    return {
        "kind": "variational",
        "result": result,
        "state": state,
        "config": config,
        "metadata": dict(metadata or {}),
    }


def resume_optimization_from_checkpoint(
    checkpoint: Mapping[str, Any],
    pipeline: Any,
    static_data: RubixData,
    target: jnp.ndarray,
    learning_rate: Optional[float] = None,
    max_steps: int = 500,
    tol: Optional[float] = None,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> tuple[OptimizationResult, OptimizationState]:
    """Resume gradient optimization from an optimization checkpoint.

    Hyperparameters (``learning_rate``, ``tol``, ``transforms``) that are not
    explicitly provided default to the values stored in the checkpoint by
    :func:`make_optimization_checkpoint`, so the resumed run uses the same
    configuration as the original without requiring the caller to re-specify
    every hyperparameter.

    Args:
        checkpoint (Mapping[str, Any]): Checkpoint payload returned by
            :func:`load_checkpoint`.
        pipeline (Any): Pipeline-like object consumed by
            :func:`rubix.inference.loss`.
        static_data (RubixData): Baseline RubixData passed to the forward
            model.
        target (jnp.ndarray): Target datacube or statistic.
        learning_rate (Optional[float], optional): Override for the learning
            rate.  Defaults to ``None`` (use value stored in the checkpoint).
        max_steps (int, optional): Maximum additional optimization steps.
            Defaults to 500.
        tol (Optional[float], optional): Override for the convergence
            tolerance.  Defaults to ``None`` (use value stored in the
            checkpoint).
        loss_fn (Optional[LossFn], optional): Optional custom loss function.
            Defaults to ``None`` (sum-of-squares).
        noise_key (Optional[jnp.ndarray], optional): Optional key for
            stochastic pipelines.  Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Override for the
            transform tree.  Defaults to ``None`` (use value stored in the
            checkpoint).
        optimizer (Optional[optax.GradientTransformation], optional): Custom
            Optax optimizer.  Defaults to ``None`` (Adam).

    Raises:
        ValueError: If the checkpoint kind is not ``'optimization'`` or the
            checkpoint is missing valid result/state fields.

    Returns:
        tuple[OptimizationResult, OptimizationState]: Resumed result and
            updated internal state.
    """
    if checkpoint.get("kind") != "optimization":
        raise ValueError("checkpoint kind must be 'optimization'")

    result = checkpoint.get("result")
    state = checkpoint.get("state")
    if not isinstance(result, OptimizationResult) or not isinstance(
        state, OptimizationState
    ):
        raise ValueError("optimization checkpoint is missing valid result/state")

    stored_config: dict[str, Any] = checkpoint.get("config") or {}
    resolved_learning_rate: float = (
        learning_rate if learning_rate is not None else stored_config.get("learning_rate", 1e-3)
    )
    resolved_tol: float = (
        tol if tol is not None else stored_config.get("tol", 1e-6)
    )
    resolved_transforms: Optional[TransformTree] = (
        transforms if transforms is not None else stored_config.get("transforms")
    )

    resumed = optimize_params(
        pipeline=pipeline,
        params_init=result.params,
        static_data=static_data,
        target=target,
        learning_rate=resolved_learning_rate,
        max_steps=max_steps,
        tol=resolved_tol,
        loss_fn=loss_fn,
        noise_key=noise_key,
        transforms=resolved_transforms,
        optimizer=optimizer,
        state_init=state,
        return_state=True,
    )
    return resumed


def resume_variational_from_checkpoint(
    checkpoint: Mapping[str, Any],
    pipeline: Any,
    static_data: RubixData,
    target: jnp.ndarray,
    learning_rate: Optional[float] = None,
    max_steps: int = 500,
    tol: Optional[float] = None,
    num_samples: Optional[int] = None,
    beta_kl: Optional[float] = None,
    init_log_std: Optional[float] = None,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    seed: Optional[int] = None,
) -> tuple[VariationalResult, VariationalState]:
    """Resume variational inference from a variational checkpoint.

    Hyperparameters (``learning_rate``, ``tol``, ``num_samples``, ``beta_kl``,
    ``init_log_std``, ``seed``, ``transforms``) that are not explicitly
    provided default to the values stored in the checkpoint by
    :func:`make_variational_checkpoint`, so the resumed run uses the same
    configuration as the original without requiring the caller to re-specify
    every hyperparameter.

    Args:
        checkpoint (Mapping[str, Any]): Checkpoint payload returned by
            :func:`load_checkpoint`.
        pipeline (Any): Pipeline-like object consumed by
            :func:`rubix.inference.loss`.
        static_data (RubixData): Baseline RubixData passed to the forward
            model.
        target (jnp.ndarray): Target datacube or statistic.
        learning_rate (Optional[float], optional): Override for the learning
            rate.  Defaults to ``None`` (use value stored in the checkpoint).
        max_steps (int, optional): Maximum additional VI steps.
            Defaults to 500.
        tol (Optional[float], optional): Override for the convergence
            tolerance.  Defaults to ``None`` (use value stored in the
            checkpoint).
        num_samples (Optional[int], optional): Override for the number of MC
            samples per step.  Defaults to ``None`` (use checkpoint value).
        beta_kl (Optional[float], optional): Override for the KL weight.
            Defaults to ``None`` (use checkpoint value).
        init_log_std (Optional[float], optional): Override for the initial
            posterior log-std.  Defaults to ``None`` (use checkpoint value).
            This value is ignored when ``state_init`` is passed to the
            underlying VI function (i.e. always during resume), but is kept
            for API symmetry.
        loss_fn (Optional[LossFn], optional): Optional custom reconstruction
            loss.  Defaults to ``None`` (sum-of-squares).
        noise_key (Optional[jnp.ndarray], optional): Optional key for
            stochastic pipelines.  Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Override for the
            transform tree.  Defaults to ``None`` (use checkpoint value).
        optimizer (Optional[optax.GradientTransformation], optional): Custom
            Optax optimizer.  Defaults to ``None`` (Adam).
        seed (Optional[int], optional): Override for the random seed.
            Defaults to ``None`` (use checkpoint value).  This value is
            ignored when ``state_init`` is passed to the underlying VI
            function (i.e. always during resume), but is kept for API
            symmetry.

    Raises:
        ValueError: If the checkpoint kind is not ``'variational'`` or the
            checkpoint is missing valid result/state fields.

    Returns:
        tuple[VariationalResult, VariationalState]: Resumed result and
            updated internal state.
    """
    if checkpoint.get("kind") != "variational":
        raise ValueError("checkpoint kind must be 'variational'")

    result = checkpoint.get("result")
    state = checkpoint.get("state")
    if not isinstance(result, VariationalResult) or not isinstance(
        state, VariationalState
    ):
        raise ValueError("variational checkpoint is missing valid result/state")

    stored_config: dict[str, Any] = checkpoint.get("config") or {}
    resolved_learning_rate: float = (
        learning_rate if learning_rate is not None else stored_config.get("learning_rate", 5e-3)
    )
    resolved_tol: float = (
        tol if tol is not None else stored_config.get("tol", 1e-6)
    )
    resolved_num_samples: int = (
        num_samples if num_samples is not None else stored_config.get("num_samples", 4)
    )
    resolved_beta_kl: float = (
        beta_kl if beta_kl is not None else stored_config.get("beta_kl", 1e-3)
    )
    resolved_init_log_std: float = (
        init_log_std if init_log_std is not None else stored_config.get("init_log_std", -2.0)
    )
    resolved_seed: int = (
        seed if seed is not None else stored_config.get("seed", 0)
    )
    resolved_transforms: Optional[TransformTree] = (
        transforms if transforms is not None else stored_config.get("transforms")
    )

    resumed = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=result.posterior_mean_constrained_params,
        static_data=static_data,
        target=target,
        learning_rate=resolved_learning_rate,
        max_steps=max_steps,
        tol=resolved_tol,
        num_samples=resolved_num_samples,
        beta_kl=resolved_beta_kl,
        init_log_std=resolved_init_log_std,
        loss_fn=loss_fn,
        noise_key=noise_key,
        transforms=resolved_transforms,
        optimizer=optimizer,
        seed=resolved_seed,
        state_init=state,
        return_state=True,
    )
    return resumed
