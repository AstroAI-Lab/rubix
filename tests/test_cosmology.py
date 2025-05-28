import pytest
from astropy.cosmology import Planck15 as astropy_cosmo
from jax import numpy as jnp

from rubix.cosmology import PLANCK15 as rubix_cosmo

# Define the cosmological parameters similar to the ones used in the BaseCosmology class

TOLERANCE = 1e-1


# Test cases for each of the methods in the BaseCosmology class that compare with Astropy calculations
@pytest.mark.parametrize("z", [0.1, 0.2, 0.5, 1.0, 2.0])
def test_angular_diameter_distance(z):
    rubix_distance = rubix_cosmo.angular_diameter_distance_to_z(z)
    astropy_distance = astropy_cosmo.angular_diameter_distance(z).value
    assert jnp.isclose(rubix_distance, astropy_distance, rtol=TOLERANCE)


@pytest.mark.parametrize("z", [0.1, 0.2, 0.5, 1.0, 2.0])
def test_comoving_distance(z):
    rubix_distance = rubix_cosmo.comoving_distance_to_z(z)
    astropy_distance = astropy_cosmo.comoving_distance(z).value
    assert jnp.isclose(rubix_distance, astropy_distance, rtol=TOLERANCE)


@pytest.mark.parametrize("z", [0.1, 0.2, 0.5, 1.0, 2.0])
def test_lookback_time(z):
    rubix_time = rubix_cosmo.lookback_to_z(z)
    astropy_time = astropy_cosmo.lookback_time(z).value
    assert jnp.isclose(rubix_time, astropy_time, rtol=TOLERANCE)


def test_scale_factor_to_redshift():
    a = 0.5
    z = rubix_cosmo.scale_factor_to_redshift(a)
    assert jnp.isclose(z, 1.0 / a - 1.0, rtol=TOLERANCE)


def test_luminosity_distance():
    z = 0.5
    rubix_distance = rubix_cosmo.luminosity_distance_to_z(z)
    astropy_distance = astropy_cosmo.luminosity_distance(z).value
    assert jnp.isclose(rubix_distance, astropy_distance, rtol=TOLERANCE)


def test_distance_modulus_to_z():
    z = 0.5
    rubix_distance = rubix_cosmo.distance_modulus_to_z(z)
    astropy_distance = astropy_cosmo.distmod(z).value
    assert jnp.isclose(rubix_distance, astropy_distance, rtol=TOLERANCE)


@pytest.mark.parametrize("z", [0.1, [0.1, 0.2, 0.5]])
def test_age_at_z(z):
    if isinstance(z, float):
        rubix_age = rubix_cosmo.age_at_z(z)
        astropy_age = astropy_cosmo.age(z).value
        assert jnp.isclose(rubix_age, astropy_age, rtol=TOLERANCE)
    else:
        z = jnp.array(z)
        rubix_age = rubix_cosmo.age_at_z(z)
        astropy_age = astropy_cosmo.age(z).value
        for r, a in zip(rubix_age, astropy_age):
            assert jnp.isclose(r, a, rtol=TOLERANCE)


@pytest.mark.parametrize("z", [0.1, 0.2, 0.5, 1.0, 2.0])
def test_angular_scale(z):
    rubix_scale = rubix_cosmo.angular_scale(z)
    # Compute the scale using Astropy's angular diameter distance in Mpc and converting to kpc/arcsec
    astropy_scale = (
        astropy_cosmo.angular_diameter_distance(z).value
        * (jnp.pi / (180 * 60 * 60))
        * 1e3
    )
    assert jnp.isclose(rubix_scale, astropy_scale, rtol=TOLERANCE)
