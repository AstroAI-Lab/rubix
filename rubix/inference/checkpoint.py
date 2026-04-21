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
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a checkpoint payload for optimization resume."""
    return {
        "kind": "optimization",
        "result": result,
        "state": state,
        "metadata": dict(metadata or {}),
    }


def make_variational_checkpoint(
    result: VariationalResult,
    state: VariationalState,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a checkpoint payload for variational resume."""
    return {
        "kind": "variational",
        "result": result,
        "state": state,
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
