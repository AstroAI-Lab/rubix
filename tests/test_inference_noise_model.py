import jax.numpy as jnp
import pytest

from rubix.inference import flux_scaled_sigma


def test_constant_floor_when_no_flux_terms():
    flux = jnp.array([[[0.0, 1.0, 100.0]]])
    sigma = flux_scaled_sigma(flux, relative_noise=0.0, floor=0.02, poisson_scale=0.0)
    assert jnp.allclose(sigma, 0.02)
    assert sigma.shape == flux.shape


def test_relative_noise_scales_with_flux():
    flux = jnp.array([1.0, 10.0, 100.0])
    sigma = flux_scaled_sigma(flux, relative_noise=0.1, floor=0.0)
    # sigma = 0.1 * |flux| -> S/N = 10 everywhere.
    assert jnp.allclose(sigma, jnp.array([0.1, 1.0, 10.0]))
    assert jnp.allclose(flux / sigma, 10.0)


def test_terms_add_in_quadrature():
    flux = jnp.array([4.0])
    sigma = flux_scaled_sigma(flux, relative_noise=0.5, floor=3.0, poisson_scale=2.0)
    # (0.5*4)^2 + 2*4 + 3^2 = 4 + 8 + 9 = 21
    assert float(sigma[0]) == pytest.approx(jnp.sqrt(21.0), rel=1e-6)


def test_floor_keeps_sigma_positive_in_empty_voxels():
    flux = jnp.zeros((2, 2))
    sigma = flux_scaled_sigma(flux, relative_noise=0.1, floor=1e-3)
    assert bool(jnp.all(sigma > 0.0))
    assert jnp.allclose(sigma, 1e-3)


def test_poisson_term_uses_nonnegative_flux():
    # Negative flux must not create a negative variance under the sqrt.
    flux = jnp.array([-5.0])
    sigma = flux_scaled_sigma(flux, floor=1.0, poisson_scale=10.0)
    assert jnp.isfinite(sigma[0])
    assert float(sigma[0]) == pytest.approx(1.0)  # poisson term clipped to 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relative_noise": -0.1},
        {"floor": -1.0},
        {"poisson_scale": -2.0},
    ],
)
def test_rejects_negative_coefficients(kwargs):
    with pytest.raises(ValueError, match="must be non-negative"):
        flux_scaled_sigma(jnp.ones((2,)), **kwargs)
