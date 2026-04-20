from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from beartype.typing import Callable

from rubix.core.data import RubixData

from .api import LossFn
from .optimize import OptimizationResult, optimize_params
from .parameterization import TransformTree

ParamsTree = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class IFUCubeBenchmarkResult:
    """Runtime and memory diagnostics for IFU-cube optimization benchmarks.

    Attributes:
        repeats (int): Number of recorded optimization runs.
        warmup (bool): Whether an unrecorded warmup run was executed.
        runtimes_s (list[float]): Per-run wall times in seconds.
        mean_runtime_s (float): Mean recorded runtime.
        median_runtime_s (float): Median recorded runtime.
        min_runtime_s (float): Minimum recorded runtime.
        max_runtime_s (float): Maximum recorded runtime.
        steps_run (int): Number of optimization steps in the last run.
        final_loss (float): Final loss at returned params in the last run.
        best_loss (float): Best loss encountered in the last run.
        target_nbytes (int): Size in bytes of target cube array.
        mask_nbytes (int): Size in bytes of mask array (0 when absent).
        weights_nbytes (int): Size in bytes of weights array (0 when absent).
        estimated_objective_working_set_nbytes (int): Approximate memory used
            by objective-side arrays (prediction + residual + target + optional
            mask/weights).
    """

    repeats: int
    warmup: bool
    runtimes_s: list[float]
    mean_runtime_s: float
    median_runtime_s: float
    min_runtime_s: float
    max_runtime_s: float
    steps_run: int
    final_loss: float
    best_loss: float
    target_nbytes: int
    mask_nbytes: int
    weights_nbytes: int
    estimated_objective_working_set_nbytes: int


def estimate_array_nbytes(shape: tuple[int, ...], dtype: Any) -> int:
    """Estimate memory footprint of a dense array in bytes.

    Args:
        shape (tuple[int, ...]): Array shape.
        dtype (Any): Array dtype or dtype-like object.

    Returns:
        int: Number of bytes for a dense array with given shape/dtype.
    """
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def _block_tree(tree: Any) -> None:
    """Block on all JAX leaves in a pytree to ensure accurate timing."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def benchmark_callable(
    run_once: Callable[[], Any],
    repeats: int,
    warmup: bool = True,
) -> list[float]:
    """Benchmark a callable over multiple repeats.

    Args:
        run_once (Callable[[], Any]): Callable to execute.
        repeats (int): Number of timed repeats.
        warmup (bool, optional): Whether to execute one untimed warmup run.
            Defaults to ``True``.

    Raises:
        ValueError: If ``repeats`` is smaller than one.

    Returns:
        list[float]: Per-repeat runtime in seconds.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    if warmup:
        _block_tree(run_once())

    runtimes_s: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        _block_tree(run_once())
        runtimes_s.append(perf_counter() - start)

    return runtimes_s


def benchmark_ifu_cube_optimization(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    normalize_loss: bool = True,
    loss_fn: Optional[LossFn] = None,
    learning_rate: float = 1e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    repeats: int = 3,
    warmup: bool = True,
) -> IFUCubeBenchmarkResult:
    """Benchmark full-IFU optimization runtime and memory diagnostics.

    Args:
        pipeline (Any): Pipeline-like object consumed by ``optimize_params``.
        params_init (ParamsTree): Initial constrained parameters.
        static_data (RubixData): Static baseline data.
        target (jnp.ndarray): Target IFU datacube.
        mask (Optional[jnp.ndarray], optional): Optional voxel mask.
            Defaults to ``None``.
        weights (Optional[jnp.ndarray], optional): Optional voxel weights.
            Defaults to ``None``.
        normalize_loss (bool, optional): Whether to normalize weighted loss.
            Defaults to ``True``.
        loss_fn (Optional[LossFn], optional): Optional custom loss. When
            provided, ``mask``/``weights``/``normalize_loss`` are ignored.
            Defaults to ``None``.
        learning_rate (float, optional): Default Adam learning rate.
            Defaults to 1e-3.
        max_steps (int, optional): Maximum optimization steps. Defaults to 500.
        tol (float, optional): Convergence threshold on update norm.
            Defaults to 1e-6.
        noise_key (Optional[jnp.ndarray], optional): Optional stochastic key.
            Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional transform tree.
            Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Custom
            optimizer override. Defaults to ``None``.
        repeats (int, optional): Number of timed optimization runs.
            Defaults to 3.
        warmup (bool, optional): Whether to run one untimed warmup.
            Defaults to ``True``.

    Raises:
        ValueError: If ``repeats`` is smaller than one, or if ``loss_fn`` is
            ``None`` and prediction/target/mask/weights shapes are
            inconsistent during evaluation.
        RuntimeError: If no optimization result is produced.

    Returns:
        IFUCubeBenchmarkResult: Runtime summary, memory diagnostics, and final
            optimization quality metrics from the last run.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    last_result: Optional[OptimizationResult] = None

    def _masked_weighted_loss(
        prediction: jnp.ndarray, truth: jnp.ndarray
    ) -> jnp.ndarray:
        if prediction.shape != truth.shape:
            raise ValueError("prediction and target must have the same shape")
        if mask is not None and mask.shape != truth.shape:
            raise ValueError("mask must have the same shape as target")
        if weights is not None and weights.shape != truth.shape:
            raise ValueError("weights must have the same shape as target")

        residual_sq = (prediction - truth) ** 2

        if mask is None:
            if weights is None:
                numerator = jnp.sum(residual_sq)
                if not normalize_loss:
                    return numerator
                denom = jnp.asarray(residual_sq.size, dtype=residual_sq.dtype)
                denom = jnp.maximum(
                    denom, jnp.asarray(1e-12, dtype=residual_sq.dtype)
                )
                return numerator / denom

            weights_f = weights.astype(residual_sq.dtype)
            residual_sq = residual_sq * weights_f
            numerator = jnp.sum(residual_sq)
            if not normalize_loss:
                return numerator
            denom = jnp.sum(weights_f)
            denom = jnp.maximum(denom, jnp.asarray(1e-12, dtype=residual_sq.dtype))
            return numerator / denom

        mask_f = mask.astype(residual_sq.dtype)
        residual_sq = residual_sq * mask_f

        if weights is not None:
            weights_f = weights.astype(residual_sq.dtype)
            residual_sq = residual_sq * weights_f
            denom = jnp.sum(weights_f * mask_f)
        else:
            denom = jnp.sum(mask_f)

        numerator = jnp.sum(residual_sq)
        if not normalize_loss:
            return numerator
        denom = jnp.maximum(denom, jnp.asarray(1e-12, dtype=residual_sq.dtype))
        return numerator / denom

    if loss_fn is None:
        effective_loss_fn: Optional[LossFn] = _masked_weighted_loss
    else:
        effective_loss_fn = loss_fn

    def _run_once() -> OptimizationResult:
        nonlocal last_result
        last_result = optimize_params(
            pipeline=pipeline,
            params_init=params_init,
            static_data=static_data,
            target=target,
            learning_rate=learning_rate,
            max_steps=max_steps,
            tol=tol,
            loss_fn=effective_loss_fn,
            noise_key=noise_key,
            transforms=transforms,
            optimizer=optimizer,
        )
        _block_tree(last_result.params)
        _block_tree(last_result.best_params)
        return last_result

    runtimes_s = benchmark_callable(_run_once, repeats=repeats, warmup=warmup)

    if last_result is None:  # pragma: no cover
        raise RuntimeError("benchmark did not produce an optimization result")

    target_nbytes = estimate_array_nbytes(target.shape, target.dtype)
    mask_nbytes = (
        estimate_array_nbytes(mask.shape, mask.dtype) if mask is not None else 0
    )
    weights_nbytes = (
        estimate_array_nbytes(weights.shape, weights.dtype)
        if weights is not None
        else 0
    )

    # Approximate objective-side working set:
    # prediction + residual + target + optional mask/weights.
    estimated_objective_working_set_nbytes = (
        3 * target_nbytes + mask_nbytes + weights_nbytes
    )

    return IFUCubeBenchmarkResult(
        repeats=repeats,
        warmup=warmup,
        runtimes_s=runtimes_s,
        mean_runtime_s=float(np.mean(runtimes_s)),
        median_runtime_s=float(np.median(runtimes_s)),
        min_runtime_s=float(np.min(runtimes_s)),
        max_runtime_s=float(np.max(runtimes_s)),
        steps_run=last_result.steps_run,
        final_loss=last_result.final_loss,
        best_loss=last_result.best_loss,
        target_nbytes=target_nbytes,
        mask_nbytes=mask_nbytes,
        weights_nbytes=weights_nbytes,
        estimated_objective_working_set_nbytes=estimated_objective_working_set_nbytes,
    )


def benchmark_result_to_dict(result: IFUCubeBenchmarkResult) -> dict[str, Any]:
    """Convert benchmark result dataclass to a JSON-serializable dictionary."""
    return asdict(result)
