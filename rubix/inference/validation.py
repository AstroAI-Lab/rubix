from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from beartype.typing import Any, Mapping
from jax.flatten_util import ravel_pytree

ParamsTree = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class GradientComparison:
    """Summary metrics comparing autodiff and finite-difference gradients."""

    max_abs_error: float
    l2_error: float
    l2_reference: float
    relative_l2_error: float


def finite_difference_grad(
    loss_fn: Callable[[ParamsTree], jnp.ndarray],
    params: ParamsTree,
    eps: float = 1e-5,
) -> Any:
    """Compute central finite-difference gradients on an arbitrary pytree.

    Args:
        loss_fn (Callable[[ParamsTree], jnp.ndarray]): Scalar loss function.
        params (ParamsTree): Parameter pytree at which to evaluate gradient.
        eps (float, optional): Central-difference step size. Defaults to 1e-5.

    Raises:
        ValueError: If ``eps`` is not strictly positive.

    Returns:
        Any: Pytree gradient matching the structure of ``params``.
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive")

    flat, unravel = ravel_pytree(params)

    def f_flat(flat_params):
        return loss_fn(unravel(flat_params))

    def fd_at_index(i):
        basis = jnp.zeros_like(flat).at[i].set(1.0)
        return (f_flat(flat + eps * basis) - f_flat(flat - eps * basis)) / (2.0 * eps)

    def body_fun(i, grad_accum):
        return grad_accum.at[i].set(fd_at_index(i))

    grad_flat = jax.lax.fori_loop(
        0,
        flat.size,
        body_fun,
        jnp.zeros_like(flat),
    )
    return unravel(grad_flat)


def compare_gradients(autodiff_grad: Any, fd_grad: Any) -> GradientComparison:
    """Compare autodiff and finite-difference gradients.

    Args:
        autodiff_grad (Any): Gradient pytree from autodiff.
        fd_grad (Any): Gradient pytree from finite differences.

    Raises:
        ValueError: If the flattened gradient vectors do not have the same shape.

    Returns:
        GradientComparison: Error metrics for the flattened gradient vectors.
    """
    auto_flat, _ = ravel_pytree(autodiff_grad)
    fd_flat, _ = ravel_pytree(fd_grad)

    if auto_flat.shape != fd_flat.shape:
        raise ValueError(
            "autodiff_grad and fd_grad must flatten to the same shape; "
            f"got {auto_flat.shape} and {fd_flat.shape}"
        )
    diff = auto_flat - fd_flat
    max_abs_error = jnp.max(jnp.abs(diff))
    l2_error = jnp.linalg.norm(diff)
    l2_reference = jnp.linalg.norm(fd_flat)
    relative_l2_error = l2_error / jnp.maximum(l2_reference, 1e-12)

    return GradientComparison(
        max_abs_error=float(max_abs_error),
        l2_error=float(l2_error),
        l2_reference=float(l2_reference),
        relative_l2_error=float(relative_l2_error),
    )
