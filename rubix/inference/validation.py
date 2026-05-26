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
    batch_size: int = 32,
    jit_compile: bool = True,
) -> Any:
    """Compute central finite-difference gradients on an arbitrary pytree.

    Args:
        loss_fn (Callable[[ParamsTree], jnp.ndarray]): Scalar loss function.
        params (ParamsTree): Parameter pytree at which to evaluate gradient.
        eps (float, optional): Central-difference step size. Defaults to 1e-5.
        batch_size (int, optional): Number of finite-difference directions to
            evaluate together. Defaults to 32.
        jit_compile (bool, optional): If ``True``, JIT-compile the flattened
            loss and batch-evaluation kernels. Defaults to ``True``.

    Raises:
        ValueError: If ``eps`` is not strictly positive or if ``loss_fn`` does
            not return a scalar.

    Returns:
        Any: Pytree gradient matching the structure of ``params``.
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive")

    sample_value = loss_fn(params)
    if jnp.ndim(sample_value) != 0:
        raise ValueError(
            "loss_fn must return a scalar. "
            f"Got shape {jnp.shape(sample_value)} instead."
        )

    flat, unravel = ravel_pytree(params)

    def f_flat(flat_params):
        return loss_fn(unravel(flat_params))

    if jit_compile:
        f_flat = jax.jit(f_flat)

    def fd_from_basis(basis):
        return (f_flat(flat + eps * basis) - f_flat(flat - eps * basis)) / (2.0 * eps)

    def unit_basis(i):
        return jnp.zeros_like(flat).at[i].set(1.0)

    def chunk_fd(start_idx):
        idx = start_idx + jnp.arange(batch_size)
        valid = idx < flat.size
        idx_safe = jnp.minimum(idx, flat.size - 1)
        basis = jax.vmap(unit_basis)(idx_safe) * valid[:, None]
        chunk_grad = jax.vmap(fd_from_basis)(basis)
        return chunk_grad * valid

    if jit_compile:
        chunk_fd = jax.jit(chunk_fd)

    starts = jnp.arange(0, flat.size, batch_size)
    grad_chunks = jax.vmap(chunk_fd)(starts)
    grad_flat = grad_chunks.reshape(-1)[: flat.size]
    return unravel(grad_flat)


def compare_gradients(autodiff_grad: Any, fd_grad: Any) -> GradientComparison:
    """Compare autodiff and finite-difference gradients.

    Args:
        autodiff_grad (Any): Gradient pytree from autodiff.
        fd_grad (Any): Gradient pytree from finite differences.

    Raises:
        ValueError: If flattened gradient vectors have different shapes.

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
