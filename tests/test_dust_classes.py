import jax.numpy as jnp
import pytest

from rubix.spectra.dust.extinction_models import Cardelli89, Gordon23
from rubix.spectra.dust.generic_models import (
    FM90,
    Drude1d,
    Polynomial1d,
    PowerLaw1d,
    _modified_drude,
)


def test_PowerLaw1d():
    x = jnp.array([1.0, 2.0, 3.0])
    amplitude = 2.0
    x_0 = 1.0
    alpha = 1.0
    expected = jnp.array([2.0, 1.0, 0.66666667])
    result = PowerLaw1d(x, amplitude, x_0, alpha)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_Polynomial1d():
    x = jnp.array([1.0, 2.0, 3.0])
    coeffs = jnp.array([1.0, 2.0, 3.0])
    expected = jnp.array([6.0, 17.0, 34.0])
    result = Polynomial1d(x, coeffs)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_Polynomial1d_single_coefficient():
    x = jnp.array([1.0, 2.0, 3.0])
    coeffs = jnp.array([1.0])
    expected = jnp.array([1.0, 1.0, 1.0])
    result = Polynomial1d(x, coeffs)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_Drude1d():
    x = jnp.array([1.0, 2.0, 3.0])
    amplitude = 1.0
    x_0 = 1.0
    fwhm = 1.0
    expected = jnp.array([1.0, 0.30769232, 0.12328766])
    result = Drude1d(x, amplitude, x_0, fwhm)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_Drude1d_value_error():
    x = jnp.array([1.0, 2.0, 3.0])
    amplitude = 1.0
    x_0 = 0.0
    fwhm = 1.0
    expected = jnp.array([1.0, 0.30769232, 0.12328766])
    with pytest.raises(ValueError, match="0 is not an allowed value for x_0"):
        result = Drude1d(x, amplitude, x_0, fwhm)


def test_modified_drude():
    x = jnp.array([1.0, 2.0, 3.0])
    scale = 1.0
    x_o = 1.0
    gamma_o = 1.0
    asym = 0.0
    expected = jnp.array([1.0, 0.30769232, 0.12328766])
    result = _modified_drude(x, scale, x_o, gamma_o, asym)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_FM90():
    x = jnp.array([4.0, 5.0, 6.0])
    C1 = 0.10
    C2 = 0.70
    C3 = 3.23
    C4 = 0.41
    xo = 4.59
    gamma = 0.95
    expected = jnp.array([4.1879544, 5.723751, 4.7574277])
    result = FM90(x, C1, C2, C3, C4, xo, gamma)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_FM90_value_errors():
    x = jnp.array([4.0, 5.0, 6.0])

    # Test C1 out of bounds
    with pytest.raises(ValueError, match="C1 is out of bounds: 6.0"):
        FM90(x, C1=6.0, C2=0.5, C3=3.0, C4=0.5, xo=4.5, gamma=0.5)

    # Test C2 out of bounds
    with pytest.raises(ValueError, match="C2 is out of bounds: -1.5"):
        FM90(x, C1=0.0, C2=-1.5, C3=3.0, C4=0.5, xo=4.5, gamma=0.5)

    # Test C3 out of bounds
    with pytest.raises(ValueError, match="C3 is out of bounds: 7.0"):
        FM90(x, C1=0.0, C2=0.5, C3=7.0, C4=0.5, xo=4.5, gamma=0.5)

    # Test C4 out of bounds
    with pytest.raises(ValueError, match="C4 is out of bounds: -1.5"):
        FM90(x, C1=0.0, C2=0.5, C3=3.0, C4=-1.5, xo=4.5, gamma=0.5)

    # Test xo out of bounds
    with pytest.raises(ValueError, match="xo is out of bounds: 5.5"):
        FM90(x, C1=0.0, C2=0.5, C3=3.0, C4=0.5, xo=5.5, gamma=0.5)

    # Test gamma out of bounds
    with pytest.raises(ValueError, match="gamma is out of bounds: 0.1"):
        FM90(x, C1=0.0, C2=0.5, C3=3.0, C4=0.5, xo=4.5, gamma=0.1)


def test_cardelli89_evaluate():
    # Test with a sample wavelength array
    wave = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    model = Cardelli89(Rv=3.1)
    result = model.evaluate(wave)

    # Check the shape of the result
    assert result.shape == wave.shape

    # Check the values are within expected range
    assert jnp.all(result >= 0)
    assert jnp.all(result <= 10)


def test_cardelli89_no_AV_noEbv():
    # Test with a sample wavelength array
    wave = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    model = Cardelli89(Rv=3.1)
    with pytest.raises(
        ValueError, match="neither Av or Ebv passed, one of them is required!"
    ):
        result = model.extinguish(wave)


def test_cardelli89_no_AV():
    # Test with a sample wavelength array
    wave = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    model = Cardelli89(Rv=3.1)
    result = model.extinguish(wave, Ebv=1.0)

    # Calculate expected extinction values
    Av = 3.1 * 1.0  # Since Ebv=1.0, Av = Rv * Ebv = 3.1 * 1.0 = 3.1
    expected = (
        model.evaluate(wave) * Av
    )  # Since Ebv=1.0, Av = Rv * Ebv = 3.1 * 1.0 = 3.1
    expected_extinction = jnp.power(10.0, -0.4 * expected)

    assert jnp.allclose(
        result, expected_extinction
    )  # , f"Expected {expected_extinction}, but got {result}"


def test_gordon23_evaluate():
    # Test with a sample wavelength array
    wave = jnp.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    model = Gordon23(Rv=3.1)
    result = model.evaluate(wave)

    # Check the shape of the result
    assert result.shape == wave.shape

    # Check the values are within expected range
    assert jnp.all(result >= 0)
    assert jnp.all(result <= 10)


"""
def test_cardelli89_invalid_rv():
    # Test with an invalid Rv value
    with pytest.raises(ValueError):
        model = Cardelli89(Rv=7.0)  # Rv out of range

def test_gordon23_invalid_rv():
    # Test with an invalid Rv value
    with pytest.raises(ValueError):
        model = Gordon23(Rv=6.0)  # Rv out of range

def test_cardelli89_wave_out_of_range():
    # Test with a wavelength out of range
    model = Cardelli89(Rv=3.1)
    wave = jnp.array([0.1, 15.0])  # Out of range wavelengths
    with pytest.raises(ValueError):
        model.evaluate(wave)

def test_gordon23_wave_out_of_range():
    # Test with a wavelength out of range
    model = Gordon23(Rv=3.1)
    wave = jnp.array([0.05, 40.0])  # Out of range wavelengths
    with pytest.raises(ValueError):
        model.evaluate(wave)
"""
