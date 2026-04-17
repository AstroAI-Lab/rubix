from dataclasses import dataclass
from typing import Mapping, Optional

import jax
import jax.numpy as jnp
import optax
from beartype.typing import Any, Callable

from .api import LossFn, loss
from .parameterization import TransformTree, apply_transforms

ParamsTree = Mapping[str, Mapping[str, Any]]


@dataclass
class OptimizationResult:
    """Container for optimization outputs."""

    params: dict[str, dict[str, Any]]
    best_params: dict[str, dict[str, Any]]
    loss_history: list[float]
    grad_norm_history: list[float]
    best_loss: float
    steps_run: int
    converged: bool


def _tree_to_dict(tree: ParamsTree) -> dict[str, dict[str, Any]]:
    """Return a mutable dictionary copy from a nested parameter tree."""
    return {component: dict(fields) for component, fields in tree.items()}


def optimize_params(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: Any,
    target: jnp.ndarray,
    learning_rate: float = 1e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> OptimizationResult:
    """Run gradient-based parameter optimization with Optax.

    Args:
        pipeline (Any): Pipeline-like object consumed by :func:`rubix.inference.loss`.
        params_init (ParamsTree): Initial parameters in constrained space.
        static_data (Any): Baseline RubixData passed to the forward model.
        target (jnp.ndarray): Target datacube or statistic.
        learning_rate (float, optional): Step size for default Adam optimizer.
            Defaults to 1e-3.
        max_steps (int, optional): Maximum optimization steps. Defaults to 500.
        tol (float, optional): Convergence threshold on global update norm.
            Defaults to 1e-6.
        loss_fn (Optional[LossFn], optional): Optional custom loss function.
            Defaults to ``None`` (sum-of-squares).
        noise_key (Optional[jnp.ndarray], optional): Optional key for stochastic
            pipelines. Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional transform tree
            that maps unconstrained parameters to constrained parameters.
            Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Custom
            Optax optimizer. Defaults to ``None`` (Adam with ``learning_rate``).

    Returns:
        OptimizationResult: Final/best params and optimization traces.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    if transforms is None:
        trainable_params = _tree_to_dict(params_init)
    else:
        trainable_params = apply_transforms(
            params=params_init,
            transforms=transforms,
            direction="inverse",
        )

    opt_state = optimizer.init(trainable_params)
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    best_loss = jnp.inf
    best_params = trainable_params
    converged = False
    steps_run = 0

    def train_loss(train_params):
        if transforms is None:
            constrained_params = train_params
        else:
            constrained_params = apply_transforms(
                params=train_params,
                transforms=transforms,
                direction="forward",
            )
        return loss(
            pipeline=pipeline,
            params=constrained_params,
            static_data=static_data,
            target=target,
            loss_fn=loss_fn,
            noise_key=noise_key,
        )

    for step in range(max_steps):
        value, grads = jax.value_and_grad(train_loss)(trainable_params)
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)

        grad_norm = optax.global_norm(grads)
        update_norm = optax.global_norm(updates)

        loss_value = float(value)
        loss_history.append(loss_value)
        grad_norm_history.append(float(grad_norm))

        if value < best_loss:
            best_loss = value
            best_params = trainable_params

        steps_run = step + 1
        if float(update_norm) < tol:
            converged = True
            break

    if transforms is None:
        final_params = trainable_params
        final_best_params = best_params
    else:
        final_params = apply_transforms(
            params=trainable_params,
            transforms=transforms,
            direction="forward",
        )
        final_best_params = apply_transforms(
            params=best_params,
            transforms=transforms,
            direction="forward",
        )

    return OptimizationResult(
        params=_tree_to_dict(final_params),
        best_params=_tree_to_dict(final_best_params),
        loss_history=loss_history,
        grad_norm_history=grad_norm_history,
        best_loss=float(best_loss),
        steps_run=steps_run,
        converged=converged,
    )
