from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rubix.core.data import RubixData

from .parameterization import TransformTree
from .variational import VariationalResult, optimize_variational_ifu_cube

ParamsTree = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class VIBenchmarkResult:
    """Runtime and quality diagnostics for VI IFU-cube benchmarks."""

    repeats: int
    warmup: bool
    runtimes_s: list[float]
    mean_runtime_s: float
    median_runtime_s: float
    min_runtime_s: float
    max_runtime_s: float
    steps_run: int
    final_objective: float
    best_objective: float
    final_reconstruction: float
    final_kl: float
    target_nbytes: int


def estimate_array_nbytes(shape: tuple[int, ...], dtype: Any) -> int:
    """Estimate memory footprint of a dense array in bytes."""
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def _block_tree(tree: Any) -> None:
    """Block on all JAX leaves in a pytree to ensure accurate timing."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def benchmark_variational_inference(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    sigma: Optional[jnp.ndarray] = None,
    inv_variance: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    normalize_loss: bool = True,
    huber_delta: Optional[float] = None,
    huber_weight: float = 0.0,
    learning_rate: float = 5e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    num_samples: int = 4,
    beta_kl: float = 1e-3,
    init_log_std: float = -2.0,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    seed: int = 0,
    repeats: int = 3,
    warmup: bool = True,
) -> VIBenchmarkResult:
    """Benchmark full-IFU variational inference runtime and objective quality.

    Args:
        pipeline (Any): Pipeline-like object consumed by VI optimization.
        params_init (ParamsTree): Initial constrained parameters.
        static_data (RubixData): Static baseline data.
        target (jnp.ndarray): Target IFU datacube.
        sigma (Optional[jnp.ndarray], optional): Per-voxel uncertainty cube.
            Defaults to ``None``.
        inv_variance (Optional[jnp.ndarray], optional): Per-voxel inverse
            variance cube. Defaults to ``None``.
        mask (Optional[jnp.ndarray], optional): Optional binary voxel mask.
            Defaults to ``None``.
        normalize_loss (bool, optional): Whether to normalize data term.
            Defaults to ``True``.
        huber_delta (Optional[float], optional): Optional Huber transition.
            Defaults to ``None``.
        huber_weight (float, optional): Weight of robust Huber term.
            Defaults to 0.0.
        learning_rate (float, optional): Adam learning rate. Defaults to 5e-3.
        max_steps (int, optional): Maximum VI steps. Defaults to 500.
        tol (float, optional): Convergence threshold. Defaults to 1e-6.
        num_samples (int, optional): Monte Carlo samples per VI step.
            Defaults to 4.
        beta_kl (float, optional): KL regularization weight. Defaults to 1e-3.
        init_log_std (float, optional): Initial posterior log-std.
            Defaults to -2.0.
        noise_key (Optional[jnp.ndarray], optional): Optional stochastic key.
            Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional parameter
            transform tree. Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Optional
            optimizer override. Defaults to ``None``.
        seed (int, optional): Base random seed. Defaults to 0.
        repeats (int, optional): Number of timed benchmark runs.
            Defaults to 3.
        warmup (bool, optional): Whether to run one untimed warmup.
            Defaults to ``True``.

    Raises:
        ValueError: If ``repeats`` is smaller than one.
        RuntimeError: If no VI result is produced.

    Returns:
        VIBenchmarkResult: Benchmark runtime summary and final VI diagnostics.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    last_result: Optional[VariationalResult] = None

    def _run_once() -> VariationalResult:
        nonlocal last_result
        last_result = optimize_variational_ifu_cube(
            pipeline=pipeline,
            params_init=params_init,
            static_data=static_data,
            target=target,
            sigma=sigma,
            inv_variance=inv_variance,
            mask=mask,
            normalize_loss=normalize_loss,
            huber_delta=huber_delta,
            huber_weight=huber_weight,
            learning_rate=learning_rate,
            max_steps=max_steps,
            tol=tol,
            num_samples=num_samples,
            beta_kl=beta_kl,
            init_log_std=init_log_std,
            noise_key=noise_key,
            transforms=transforms,
            optimizer=optimizer,
            seed=seed,
        )
        _block_tree(last_result)
        return last_result

    if warmup:
        _ = _run_once()

    runtimes_s: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        _ = _run_once()
        runtimes_s.append(perf_counter() - start)

    if last_result is None:  # pragma: no cover
        raise RuntimeError("benchmark did not produce a variational result")

    return VIBenchmarkResult(
        repeats=repeats,
        warmup=warmup,
        runtimes_s=runtimes_s,
        mean_runtime_s=float(np.mean(runtimes_s)),
        median_runtime_s=float(np.median(runtimes_s)),
        min_runtime_s=float(np.min(runtimes_s)),
        max_runtime_s=float(np.max(runtimes_s)),
        steps_run=last_result.steps_run,
        final_objective=last_result.final_objective,
        best_objective=last_result.best_objective,
        final_reconstruction=last_result.final_reconstruction,
        final_kl=last_result.final_kl,
        target_nbytes=estimate_array_nbytes(target.shape, target.dtype),
    )


def vi_benchmark_result_to_dict(result: VIBenchmarkResult) -> dict[str, Any]:
    """Convert VI benchmark dataclass to a JSON-serializable dictionary."""
    return asdict(result)
