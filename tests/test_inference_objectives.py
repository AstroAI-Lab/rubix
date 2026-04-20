import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import build_ifu_cube_loss, masked_weighted_mse, optimize_ifu_cube


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


def test_masked_weighted_mse_matches_expected_value():
    prediction = jnp.array([[[1.0, 3.0], [2.0, 2.0]]])
    target = jnp.array([[[2.0, 1.0], [2.0, 2.0]]])
    mask = jnp.array([[[1.0, 0.0], [1.0, 1.0]]])
    weights = jnp.array([[[2.0, 5.0], [0.5, 1.0]]])

    value = masked_weighted_mse(
        prediction=prediction,
        target=target,
        mask=mask,
        weights=weights,
        normalize=True,
    )

    # Active weighted residual sum: 2*(1^2) + 0.5*(0^2) + 1*(0^2) = 2
    # Effective weight sum over active voxels: 2 + 0.5 + 1 = 3.5
    assert jnp.isclose(value, 2.0 / 3.5)


def test_ifu_cube_loss_rejects_mask_shape_mismatch():
    loss_fn = build_ifu_cube_loss(mask=jnp.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="mask must have the same shape as target"):
        _ = loss_fn(jnp.ones((1, 2, 2)), jnp.ones((1, 2, 2)))


def test_optimize_ifu_cube_uses_mask_to_ignore_unselected_voxels():
    template = jnp.array(
        [
            [[2.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    pipeline = CubeScalePipeline(template)
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}

    # Only one voxel should define the optimum scale: 2*scale ~= 4 -> scale ~= 2.
    target = jnp.array(
        [
            [[4.0, 999.0], [999.0, 999.0]],
            [[999.0, 999.0], [999.0, 999.0]],
        ]
    )
    mask = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )

    result = optimize_ifu_cube(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        mask=mask,
        learning_rate=0.2,
        max_steps=200,
        tol=1e-8,
    )

    assert result.loss_history[0] > result.loss_history[-1]
    assert abs(float(result.params["stars"]["age"][0]) - 2.0) < 5e-2


def test_optimize_ifu_cube_requires_3d_target():
    pipeline = CubeScalePipeline(jnp.ones((2, 2, 2)))
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}

    with pytest.raises(ValueError, match="target must be a 3D IFU datacube"):
        _ = optimize_ifu_cube(
            pipeline=pipeline,
            params_init=params_init,
            static_data=static_data,
            target=jnp.ones((2, 2)),
        )


def test_optimize_ifu_cube_rejects_non_finite_weights():
    pipeline = CubeScalePipeline(jnp.ones((2, 2, 2)))
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}
    bad_weights = jnp.array([[[1.0, float("inf")], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])

    with pytest.raises(ValueError, match="weights must be finite"):
        _ = optimize_ifu_cube(
            pipeline=pipeline,
            params_init=params_init,
            static_data=static_data,
            target=jnp.ones((2, 2, 2)),
            weights=bad_weights,
        )


def test_optimize_ifu_cube_rejects_negative_weights():
    pipeline = CubeScalePipeline(jnp.ones((2, 2, 2)))
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}
    bad_weights = jnp.array([[[-1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])

    with pytest.raises(ValueError, match="weights must be non-negative"):
        _ = optimize_ifu_cube(
            pipeline=pipeline,
            params_init=params_init,
            static_data=static_data,
            target=jnp.ones((2, 2, 2)),
            weights=bad_weights,
        )


def test_optimize_ifu_cube_rejects_negative_mask():
    pipeline = CubeScalePipeline(jnp.ones((2, 2, 2)))
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}
    bad_mask = jnp.array([[[-1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])

    with pytest.raises(ValueError, match="mask must be non-negative"):
        _ = optimize_ifu_cube(
            pipeline=pipeline,
            params_init=params_init,
            static_data=static_data,
            target=jnp.ones((2, 2, 2)),
            mask=bad_mask,
        )


def test_optimize_ifu_cube_with_weights_converges():
    template = jnp.array(
        [
            [[2.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    pipeline = CubeScalePipeline(template)
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.5])}}

    # Target: scale=3 -> template*3; only first voxel gets high weight
    target = jnp.full((2, 2, 2), 3.0) * template
    weights = jnp.zeros((2, 2, 2))
    weights = weights.at[0, 0, 0].set(10.0)
    weights = weights.at[0, 0, 1].set(1.0)

    result = optimize_ifu_cube(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        weights=weights,
        normalize_loss=True,
        learning_rate=0.1,
        max_steps=300,
        tol=1e-8,
    )

    assert result.loss_history[0] > result.loss_history[-1]
    assert abs(float(result.params["stars"]["age"][0]) - 3.0) < 0.1


def test_optimize_ifu_cube_normalize_loss_false_converges():
    template = jnp.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    pipeline = CubeScalePipeline(template)
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.5])}}
    target = 2.5 * template

    result = optimize_ifu_cube(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        normalize_loss=False,
        learning_rate=0.01,
        max_steps=600,
        tol=1e-8,
    )

    assert result.loss_history[0] > result.loss_history[-1]
    assert abs(float(result.params["stars"]["age"][0]) - 2.5) < 0.1
