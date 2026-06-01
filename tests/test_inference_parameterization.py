import jax.numpy as jnp
import pytest

from rubix.inference import (
    IdentityTransform,
    SigmoidBounds,
    SoftplusLowerBound,
    VelocityZBoundsTransform,
    apply_transforms,
    build_age_metallicity_transforms,
    build_age_metallicity_velocity_transforms,
    inverse_transforms,
)


def test_identity_transform_roundtrip():
    transform = IdentityTransform()
    values = jnp.array([-2.0, 0.0, 3.0])

    constrained = transform.forward(values)
    recovered = transform.inverse(constrained)

    assert jnp.allclose(constrained, values)
    assert jnp.allclose(recovered, values)


def test_sigmoid_bounds_forward_and_inverse():
    transform = SigmoidBounds(lower=0.0, upper=20.0)
    unconstrained = jnp.array([-3.0, 0.0, 3.0])

    constrained = transform.forward(unconstrained)
    recovered = transform.inverse(constrained)

    assert jnp.all(constrained > 0.0)
    assert jnp.all(constrained < 20.0)
    assert jnp.allclose(recovered, unconstrained, atol=1e-5, rtol=1e-5)


def test_softplus_lower_bound_forward_and_inverse():
    transform = SoftplusLowerBound(lower=0.0)
    unconstrained = jnp.array([-6.0, 0.0, 4.0])

    constrained = transform.forward(unconstrained)
    recovered = transform.inverse(constrained)

    assert jnp.all(constrained > 0.0)
    assert jnp.allclose(recovered, unconstrained, atol=1e-5, rtol=1e-5)


def test_softplus_lower_bound_inverse_is_stable_for_large_values():
    transform = SoftplusLowerBound(lower=0.0)
    constrained = jnp.array([1e2, 1e4], dtype=jnp.float32)
    recovered = transform.inverse(constrained)

    assert not jnp.isnan(recovered).any()
    assert not jnp.isinf(recovered).any()


def test_apply_transforms_tree_roundtrip():
    params = {
        "stars": {
            "age": jnp.array([-1.0, 0.0, 1.0]),
            "metallicity": jnp.array([-2.0, 0.5, 2.0]),
            "mass": jnp.array([1.0, 2.0, 3.0]),
        }
    }
    transforms = build_age_metallicity_transforms(
        age_lower=0.0,
        age_upper=20.0,
        metallicity_lower=0.0,
        metallicity_upper=0.05,
    )

    constrained = apply_transforms(params, transforms, direction="forward")
    recovered = inverse_transforms(constrained, transforms)

    assert jnp.all(constrained["stars"]["age"] > 0.0)
    assert jnp.all(constrained["stars"]["age"] < 20.0)
    assert jnp.all(constrained["stars"]["metallicity"] > 0.0)
    assert jnp.all(constrained["stars"]["metallicity"] < 0.05)
    assert jnp.allclose(recovered["stars"]["age"], params["stars"]["age"], atol=1e-5)
    assert jnp.allclose(
        recovered["stars"]["metallicity"],
        params["stars"]["metallicity"],
        atol=1e-5,
    )
    assert jnp.allclose(recovered["stars"]["mass"], params["stars"]["mass"])


def test_apply_transforms_raises_on_bad_direction():
    params = {"stars": {"age": jnp.array([0.0])}}
    transforms = build_age_metallicity_transforms()

    with pytest.raises(ValueError, match="direction must be one of"):
        apply_transforms(params, transforms, direction="bad")


def test_sigmoid_bounds_raises_for_invalid_bounds():
    with pytest.raises(ValueError, match="upper must be strictly larger than lower"):
        SigmoidBounds(lower=1.0, upper=1.0)


def test_velocity_z_bounds_transform_keeps_xy_fixed_and_roundtrips_z():
    fixed_xy = jnp.array([[10.0, -5.0], [7.0, 3.0]])
    transform = VelocityZBoundsTransform(
        lower_z=-300.0, upper_z=300.0, fixed_xy=fixed_xy
    )
    unconstrained = jnp.array([[4.0, -2.0, -1.0], [9.0, 9.0, 2.0]])

    constrained = transform.forward(unconstrained)
    recovered = transform.inverse(constrained)

    assert jnp.allclose(constrained[:, :2], fixed_xy)
    assert jnp.all(constrained[:, 2] > -300.0)
    assert jnp.all(constrained[:, 2] < 300.0)
    assert jnp.allclose(recovered[:, :2], jnp.zeros_like(recovered[:, :2]))
    assert jnp.allclose(recovered[:, 2], unconstrained[:, 2], atol=1e-5, rtol=1e-5)


def test_build_age_metallicity_velocity_transforms_adds_velocity_transform():
    fixed_xy = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    transforms = build_age_metallicity_velocity_transforms(
        fixed_velocity_xy=fixed_xy,
        age_lower=0.5,
        age_upper=12.0,
        metallicity_lower=5e-4,
        metallicity_upper=0.01,
        vz_lower=-200.0,
        vz_upper=200.0,
    )

    params = {
        "stars": {
            "age": jnp.array([0.0, 1.0]),
            "metallicity": jnp.array([-1.0, 2.0]),
            "velocity": jnp.array([[5.0, 6.0, -2.0], [7.0, 8.0, 1.5]]),
        }
    }

    constrained = apply_transforms(params, transforms, direction="forward")
    assert jnp.allclose(constrained["stars"]["velocity"][:, :2], fixed_xy)
    assert jnp.all(constrained["stars"]["velocity"][:, 2] > -200.0)
    assert jnp.all(constrained["stars"]["velocity"][:, 2] < 200.0)
