import jax.numpy as jnp
import pytest

from rubix.inference import (
    IdentityTransform,
    SigmoidBounds,
    SoftplusLowerBound,
    apply_transforms,
    build_age_metallicity_transforms,
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
