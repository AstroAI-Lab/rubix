import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference.benchmark import (
    benchmark_callable,
    benchmark_ifu_cube_optimization,
    estimate_array_nbytes,
)


class CubeScalePipeline:
    """Pipeline that scales a fixed cube template by stars.age[0]."""

    def __init__(self, template: jnp.ndarray):
        self.template = template

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        scale = rubixdata.stars.age[0]
        return scale * self.template


def _make_rubix_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
            age=jnp.array([0.0]),
            metallicity=jnp.array([0.01]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def test_estimate_array_nbytes_matches_expected():
    expected = 2 * 3 * 4  # float32 itemsize=4
    assert estimate_array_nbytes((2, 3), jnp.float32) == expected


def test_benchmark_callable_rejects_invalid_repeats():
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        _ = benchmark_callable(lambda: None, repeats=0)


def test_benchmark_callable_runs_expected_number_of_times():
    calls = {"count": 0}

    def _run_once():
        calls["count"] += 1

    _ = benchmark_callable(_run_once, repeats=3, warmup=True)
    assert calls["count"] == 4


def test_benchmark_ifu_cube_optimization_returns_summary():
    cube_shape = (2, 2, 4)
    template = jnp.ones(cube_shape, dtype=jnp.float32)
    pipeline = CubeScalePipeline(template)

    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.1])}}
    target = 1.5 * template

    mask = jnp.ones(cube_shape, dtype=jnp.float32)
    weights = jnp.linspace(1.0, 2.0, cube_shape[-1], dtype=jnp.float32)
    weights = jnp.broadcast_to(weights, cube_shape)

    result = benchmark_ifu_cube_optimization(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        mask=mask,
        weights=weights,
        learning_rate=0.2,
        max_steps=80,
        tol=1e-8,
        repeats=2,
        warmup=False,
    )

    assert result.repeats == 2
    assert len(result.runtimes_s) == 2
    assert result.mean_runtime_s > 0.0
    assert result.target_nbytes == estimate_array_nbytes(cube_shape, target.dtype)
    assert result.mask_nbytes == estimate_array_nbytes(cube_shape, mask.dtype)
    assert result.weights_nbytes == estimate_array_nbytes(cube_shape, weights.dtype)
    assert result.best_loss <= result.final_loss
