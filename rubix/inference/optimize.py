from dataclasses import dataclass
from typing import Mapping, Optional

import jax
import jax.numpy as jnp
import optax
from beartype.typing import Any

from rubix.core.data import RubixData

from .api import LossFn, loss
from .objectives import build_ifu_cube_loss
from .parameterization import TransformTree, apply_transforms

ParamsTree = Mapping[str, Mapping[str, Any]]


@dataclass
class OptimizationResult:
    """Container for optimization outputs.

    Attributes:
        params (dict[str, dict[str, Any]]): Final parameters (constrained
            space) after ``steps_run`` steps.
        best_params (dict[str, dict[str, Any]]): Parameters (constrained space)
            that achieved ``best_loss``.
        loss_history (list[float]): Loss values recorded at the *pre-update* parameters each
            step, for the first ``steps_run`` steps.  ``loss_history[-1]``
            therefore corresponds to the step *before* the final parameter
            update, not to ``params``.  Use ``final_loss`` for the loss at
            ``params``.
        grad_norm_history (list[float]): Global gradient-norm for each active step.
        best_loss (float): Minimum loss seen across all active steps; equals
            ``loss(pipeline, best_params, ...)``.
        final_loss (float): Loss evaluated at the returned ``params`` (post-update).
        steps_run (int): Number of active (non-frozen) optimization steps taken.
        converged (bool): ``True`` if update-norm fell below ``tol`` before
            ``max_steps``.
    """

    params: dict[str, dict[str, Any]]
    best_params: dict[str, dict[str, Any]]
    loss_history: list[float]
    grad_norm_history: list[float]
    best_loss: float
    final_loss: float
    steps_run: int
    converged: bool


@dataclass
class OptimizationState:
    """Serializable optimization state for checkpointing/resume."""

    trainable_params: dict[str, dict[str, Any]]
    opt_state: Any
    best_trainable_params: dict[str, dict[str, Any]]
    best_loss: float


def _tree_to_dict(tree: ParamsTree) -> dict[str, dict[str, Any]]:
    """Return a mutable dictionary copy from a nested parameter tree."""
    return {component: dict(fields) for component, fields in tree.items()}


def optimize_params(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    learning_rate: float = 1e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    state_init: Optional[OptimizationState] = None,
    return_state: bool = False,
) -> Any:
    """Run gradient-based parameter optimization with Optax.

    Args:
        pipeline (Any): Pipeline-like object consumed by :func:`rubix.inference.loss`.
        params_init (ParamsTree): Initial parameters in constrained space.
        static_data (RubixData): Baseline RubixData passed to the forward model.
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
        state_init (Optional[OptimizationState], optional): Optional internal
            state for exact resume. Defaults to ``None``.
        return_state (bool, optional): If ``True``, also return updated
            :class:`OptimizationState`. Defaults to ``False``.

    Returns:
        OptimizationResult: Final/best params, optimization traces, and the
            loss evaluated at the returned ``params`` (``final_loss``).
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    if state_init is None:
        if transforms is None:
            trainable_params = _tree_to_dict(params_init)
        else:
            trainable_params = apply_transforms(
                params=params_init,
                transforms=transforms,
                direction="inverse",
            )
        opt_state = optimizer.init(trainable_params)
        best_trainable_params = trainable_params
        best_loss_init = jnp.array(jnp.inf)
    else:
        trainable_params = state_init.trainable_params
        opt_state = state_init.opt_state
        best_trainable_params = state_init.best_trainable_params
        best_loss_init = jnp.asarray(state_init.best_loss)

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

    def scan_step(carry, _):
        params, opt_state, best_params, best_loss, done = carry

        def active_step(active_carry):
            params, opt_state, best_params, best_loss, _ = active_carry
            value, grads = jax.value_and_grad(train_loss)(params)
            param_updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, param_updates)

            grad_norm = optax.global_norm(grads)
            update_norm = optax.global_norm(param_updates)

            # Track best params on-device without Python-side branching.
            # `value` is evaluated at the pre-update `params`, so keep
            # `best_params` aligned with that same parameter state.
            is_better = value < best_loss
            new_best_loss = jnp.where(is_better, value, best_loss)
            new_best_params = jax.tree_util.tree_map(
                lambda current, old: jnp.where(is_better, current, old),
                params,
                best_params,
            )
            new_done = update_norm < tol
            new_carry = (
                new_params,
                new_opt_state,
                new_best_params,
                new_best_loss,
                new_done,
            )
            return new_carry, (value, grad_norm, update_norm)

        def finished_step(finished_carry):
            params, opt_state, best_params, best_loss, done = finished_carry
            zero = jnp.array(0.0)
            return (
                params,
                opt_state,
                best_params,
                best_loss,
                done,
            ), (best_loss, zero, zero)

        return jax.lax.cond(done, finished_step, active_step, carry)

    init_carry = (
        trainable_params,
        opt_state,
        best_trainable_params,
        best_loss_init,
        jnp.array(False),
    )
    (
        final_params,
        final_opt_state,
        final_best_params,
        best_loss_val,
        converged_flag,
    ), (
        loss_arr,
        grad_norm_arr,
        update_norm_arr,
    ) = jax.lax.scan(
        scan_step, init_carry, None, length=max_steps
    )

    # Detect convergence post-hoc; no device->host sync until here.
    # Because the scan freezes once `update_norm < tol`, the first True marks
    # the actual stopping point and later iterations cannot invalidate it.
    converged_mask = update_norm_arr < tol
    converged = bool(converged_flag)
    steps_run = (int(jnp.argmax(converged_mask)) + 1) if converged else max_steps

    # Materialize history once at the end
    loss_history: list[float] = loss_arr[:steps_run].tolist()
    grad_norm_history: list[float] = grad_norm_arr[:steps_run].tolist()

    # Compute loss at the returned final params so callers have a value
    # consistent with `result.params` (`loss_history` is pre-update).
    final_loss_val = float(train_loss(final_params))

    if transforms is None:
        result_params = _tree_to_dict(final_params)
        result_best_params = _tree_to_dict(final_best_params)
    else:
        result_params = _tree_to_dict(
            apply_transforms(
                params=final_params,
                transforms=transforms,
                direction="forward",
            )
        )
        result_best_params = _tree_to_dict(
            apply_transforms(
                params=final_best_params,
                transforms=transforms,
                direction="forward",
            )
        )

    result = OptimizationResult(
        params=result_params,
        best_params=result_best_params,
        loss_history=loss_history,
        grad_norm_history=grad_norm_history,
        best_loss=float(best_loss_val),
        final_loss=final_loss_val,
        steps_run=steps_run,
        converged=converged,
    )

    state = OptimizationState(
        trainable_params=_tree_to_dict(final_params),
        opt_state=final_opt_state,
        best_trainable_params=_tree_to_dict(final_best_params),
        best_loss=float(best_loss_val),
    )

    if return_state:
        return result, state
    return result


def optimize_ifu_cube(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    normalize_loss: bool = True,
    learning_rate: float = 1e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    state_init: Optional[OptimizationState] = None,
    return_state: bool = False,
) -> Any:
    """Optimize parameters against a full IFU cube with optional weighting.

    Args:
        pipeline (Any): Pipeline-like object consumed by :func:`optimize_params`.
        params_init (ParamsTree): Initial parameters in constrained space.
        static_data (RubixData): Baseline RubixData passed to the forward model.
        target (jnp.ndarray): Target IFU datacube.
        mask (Optional[jnp.ndarray], optional): Optional binary voxel mask with
            same shape as ``target``. Defaults to ``None``.
        weights (Optional[jnp.ndarray], optional): Optional per-voxel weights
            with same shape as ``target``. Defaults to ``None``.
        normalize_loss (bool, optional): Whether to normalize weighted squared
            residuals by effective weight sum. Defaults to ``True``.
        learning_rate (float, optional): Step size for default Adam optimizer.
            Defaults to 1e-3.
        max_steps (int, optional): Maximum optimization steps. Defaults to 500.
        tol (float, optional): Convergence threshold on global update norm.
            Defaults to 1e-6.
        noise_key (Optional[jnp.ndarray], optional): Optional key for stochastic
            pipelines. Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional transform tree
            for constrained optimization. Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Custom
            Optax optimizer. Defaults to ``None`` (Adam with ``learning_rate``).
        state_init (Optional[OptimizationState], optional): Optional internal
            state for exact resume. Defaults to ``None``.
        return_state (bool, optional): If ``True``, also return updated
            :class:`OptimizationState`. Defaults to ``False``.

    Raises:
        ValueError: If ``target`` is not 3D, or if ``weights``/``mask`` have
            invalid values.

    Returns:
        OptimizationResult: Standard optimization traces and best/final params.
    """
    if target.ndim != 3:
        raise ValueError("target must be a 3D IFU datacube")

    if weights is not None:
        if not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError("weights must be finite")
        if not bool(jnp.all(weights >= 0)):
            raise ValueError("weights must be non-negative")

    if mask is not None:
        if not bool(jnp.all(mask >= 0)):
            raise ValueError("mask must be non-negative")

    cube_loss_fn = build_ifu_cube_loss(
        mask=mask,
        weights=weights,
        normalize=normalize_loss,
    )

    return optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tol=tol,
        loss_fn=cube_loss_fn,
        noise_key=noise_key,
        transforms=transforms,
        optimizer=optimizer,
        state_init=state_init,
        return_state=return_state,
    )
