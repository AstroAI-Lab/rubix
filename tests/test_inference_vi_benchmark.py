import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference.vi_benchmark import (
    benchmark_variational_inference,
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
    assert estimate_array_nbytes((2, 3), jnp.float32) == 24


def test_benchmark_variational_inference_rejects_invalid_repeats():
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        benchmark_variational_inference(
            pipeline=CubeScalePipeline(jnp.ones((1, 1, 1))),
            params_init={"stars": {"age": jnp.array([0.2])}},
            static_data=_make_rubix_data(),
            target=jnp.ones((1, 1, 1)),
            repeats=0,
        )


def test_benchmark_variational_inference_returns_summary():
    cube_shape = (2, 2, 4)
    template = jnp.ones(cube_shape, dtype=jnp.float32)
    pipeline = CubeScalePipeline(template)
    target = 1.5 * template
    sigma = jnp.ones_like(target)

    result = benchmark_variational_inference(
        pipeline=pipeline,
        params_init={"stars": {"age": jnp.array([0.2])}},
        static_data=_make_rubix_data(),
        target=target,
        sigma=sigma,
        learning_rate=5e-2,
        max_steps=60,
        tol=1e-8,
        num_samples=3,
        beta_kl=1e-4,
        repeats=2,
        warmup=False,
        seed=5,
    )

    assert result.repeats == 2
    assert len(result.runtimes_s) == 2
    assert result.mean_runtime_s > 0.0
    # best_objective is the EMA-smoothed selection value and need not be below a
    # single raw final evaluation; it must at least be finite.
    assert jnp.isfinite(result.best_objective)
    assert result.target_nbytes == estimate_array_nbytes(cube_shape, target.dtype)
