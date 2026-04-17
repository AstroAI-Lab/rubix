from dataclasses import fields, replace
from typing import Callable, Mapping, Optional

import jax
import jax.numpy as jnp
from beartype.typing import Any

from rubix.core.data import RubixData

ParamsTree = Mapping[str, Mapping[str, Any]]
LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _replace_component_fields(
    component: Any,
    updates: Mapping[str, Any],
    component_name: str,
) -> Any:
    """Return a dataclass copy with selected fields replaced."""
    if component is None:
        raise ValueError(
            f"Cannot update '{component_name}' because it is not present in RubixData."
        )

    valid_fields = {field.name for field in fields(component)}
    unknown_fields = sorted(set(updates) - valid_fields)
    if unknown_fields:
        raise ValueError(
            f"Unknown fields for '{component_name}': {unknown_fields}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    return replace(component, **dict(updates))


def apply_params(static_data: RubixData, params: ParamsTree) -> RubixData:
    """Return a copy of ``static_data`` updated with optimization parameters.

    Args:
        static_data (RubixData): Baseline data that should remain unchanged.
        params (ParamsTree): Nested updates keyed by component name
            (``"galaxy"``, ``"stars"``, ``"gas"``) and then field names.

    Returns:
        RubixData: A new RubixData object with requested fields replaced.

    Raises:
        ValueError: If ``params`` references unknown components or fields, or
            if a requested component is not present in ``static_data``.
    """
    updated = replace(static_data)

    allowed_components = {"galaxy", "stars", "gas"}
    unknown_components = sorted(set(params) - allowed_components)
    if unknown_components:
        raise ValueError(
            f"Unknown RubixData components: {unknown_components}. "
            f"Valid components: {sorted(allowed_components)}"
        )

    if "galaxy" in params:
        updated.galaxy = _replace_component_fields(
            updated.galaxy, params["galaxy"], "galaxy"
        )
    if "stars" in params:
        updated.stars = _replace_component_fields(
            updated.stars, params["stars"], "stars"
        )
    if "gas" in params:
        updated.gas = _replace_component_fields(updated.gas, params["gas"], "gas")

    return updated


def forward(
    pipeline: Any,
    params: ParamsTree,
    static_data: RubixData,
) -> jnp.ndarray:
    """Run the Rubix forward model with updated parameters.

    Args:
        pipeline (Any): Object exposing ``run_sharded(RubixData)``.
        params (ParamsTree): Parameter updates applied to ``static_data``.
        static_data (RubixData): Baseline particle data.

    Returns:
        jnp.ndarray: Predicted IFU datacube.
    """
    model_input = apply_params(static_data, params)
    return pipeline.run_sharded(model_input)


def loss(
    pipeline: Any,
    params: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    loss_fn: Optional[LossFn] = None,
) -> jnp.ndarray:
    """Compute a scalar loss between predicted and target datacubes."""
    prediction = forward(pipeline=pipeline, params=params, static_data=static_data)
    if loss_fn is None:
        return jnp.sum((prediction - target) ** 2)
    return loss_fn(prediction, target)


def value_and_grad(
    pipeline: Any,
    params: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    loss_fn: Optional[LossFn] = None,
):
    """Return loss value and gradient with respect to ``params``."""

    def _loss_fn(current_params):
        return loss(
            pipeline=pipeline,
            params=current_params,
            static_data=static_data,
            target=target,
            loss_fn=loss_fn,
        )

    return jax.value_and_grad(_loss_fn)(params)
