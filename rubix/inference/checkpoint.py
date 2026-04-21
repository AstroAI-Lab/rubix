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
    learning_rate: float = 1e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> tuple[OptimizationResult, OptimizationState]:
    """Resume gradient optimization from an optimization checkpoint."""
    if checkpoint.get("kind") != "optimization":
        raise ValueError("checkpoint kind must be 'optimization'")

    result = checkpoint.get("result")
    state = checkpoint.get("state")
    if not isinstance(result, OptimizationResult) or not isinstance(
        state, OptimizationState
    ):
        raise ValueError("optimization checkpoint is missing valid result/state")

    resumed = optimize_params(
        pipeline=pipeline,
        params_init=result.params,
        static_data=static_data,
        target=target,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tol=tol,
        loss_fn=loss_fn,
        noise_key=noise_key,
        transforms=transforms,
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
    learning_rate: float = 5e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    num_samples: int = 4,
    beta_kl: float = 1e-3,
    init_log_std: float = -2.0,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    seed: int = 0,
) -> tuple[VariationalResult, VariationalState]:
    """Resume variational inference from a variational checkpoint."""
    if checkpoint.get("kind") != "variational":
        raise ValueError("checkpoint kind must be 'variational'")

    result = checkpoint.get("result")
    state = checkpoint.get("state")
    if not isinstance(result, VariationalResult) or not isinstance(
        state, VariationalState
    ):
        raise ValueError("variational checkpoint is missing valid result/state")

    resumed = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=result.posterior_mean_constrained_params,
        static_data=static_data,
        target=target,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tol=tol,
        num_samples=num_samples,
        beta_kl=beta_kl,
        init_log_std=init_log_std,
        loss_fn=loss_fn,
        noise_key=noise_key,
        transforms=transforms,
        optimizer=optimizer,
        seed=seed,
        state_init=state,
        return_state=True,
    )
    return resumed
