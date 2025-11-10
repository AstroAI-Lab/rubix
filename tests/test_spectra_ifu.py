import jax.numpy as jnp
import numpy as np
import pytest

from rubix.spectra.ifu import (
    _get_velocity_component_multiple,
    _get_velocity_component_single,
    calculate_cube,
    calculate_diff,
    convert_luminoisty_to_flux,
    cosmological_doppler_shift,
    get_velocity_component,
    resample_spectrum,
    velocity_doppler_shift,
)

# Assuming the functions are imported from the module

# Mock config dictionary for testing
mock_config = {
    "constants": {
        "LSOL_TO_ERG": 3.828e33,
        "MPC_TO_CM": 3.086e24,
        "SPEED_OF_LIGHT": 299792.458,  # km/s
    }
}


def test_cosmological_doppler_shift():
    wavelength = jnp.array([5000.0, 6000.0, 7000.0])
    z = 0.1
    shifted_wavelength = cosmological_doppler_shift(z, wavelength)
    expected_shifted_wavelength = wavelength * (1 + z)
    assert jnp.allclose(shifted_wavelength, expected_shifted_wavelength)


def test_calculate_diff():
    vec = jnp.array([1.0, 2.0, 4.0, 7.0])
    differences = calculate_diff(vec)
    expected_differences = jnp.array([0, 1, 2, 3])
    assert jnp.array_equal(differences, expected_differences)


def test_calculate_diff_without_padding():
    vec = jnp.array([1.0, 2.0, 4.0, 7.0])
    differences = calculate_diff(vec, pad_with_zero=False)
    expected_differences = jnp.array([1, 2, 3])
    assert jnp.array_equal(
        differences, expected_differences
    ), f"Expected {expected_differences}, got {differences}"


def test_get_velocity_component_single():
    vec = jnp.array([10.0, 20.0, 30.0])
    assert jnp.isclose(_get_velocity_component_single(vec, "x"), 10.0)
    assert jnp.isclose(_get_velocity_component_single(vec, "y"), 20.0)
    assert jnp.isclose(_get_velocity_component_single(vec, "z"), 30.0)


def test_get_velocity_component_single_value_error():
    # Test for vector of incorrect size
    vec_invalid_size = jnp.array([1.0, 2.0])
    with pytest.raises(ValueError):
        _get_velocity_component_single(vec_invalid_size, "x")

    # Test for invalid direction
    vec_valid = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _get_velocity_component_single(vec_valid, "invalid_direction")


def test_get_velocity_component_multiple():
    vecs = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    assert jnp.array_equal(
        _get_velocity_component_multiple(vecs, "x"), jnp.array([10.0, 40.0])
    )
    assert jnp.array_equal(
        _get_velocity_component_multiple(vecs, "y"), jnp.array([20.0, 50.0])
    )
    assert jnp.array_equal(
        _get_velocity_component_multiple(vecs, "z"), jnp.array([30.0, 60.0])
    )


def test_get_velocity_component_multiple_value_error():
    # Test for vectors of incorrect shape
    vecs_invalid_shape = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Shape (2, 2)
    with pytest.raises(
        ValueError,
    ):
        _get_velocity_component_multiple(vecs_invalid_shape, "x")

    # Test for invalid direction
    vecs_valid = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)
    with pytest.raises(ValueError):
        _get_velocity_component_multiple(vecs_valid, "invalid_direction")


def test_function_get_velocity_component_multiple():
    vecs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = get_velocity_component(vecs, "y")
    expected_result = jnp.array([2.0, 5.0])
    assert jnp.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"


def test_get_velocity_component_value_error():
    # Test for invalid shape
    vec_invalid_shape = jnp.array([1.0, 2.0])  # Shape (2,)
    with pytest.raises(ValueError):
        get_velocity_component(vec_invalid_shape, "x")

    # Test for unsupported shape
    vec_unsupported_shape = jnp.array([[[1.0, 2.0, 3.0]]])  # Shape (1, 1, 3)
    with pytest.raises(ValueError):
        get_velocity_component(vec_unsupported_shape, "x")


def test_convert_luminoisty_to_flux():
    luminosity = jnp.array([1e33, 2e33, 3e33])
    observation_lum_dist = 1e2  # Mpc
    observation_z = 0.5
    pixel_size = 1.0

    flux = convert_luminoisty_to_flux(
        luminosity,
        observation_lum_dist,
        observation_z,
        pixel_size,
        CONSTANTS=mock_config["constants"],
    )
    # Recalculate expected flux values based on the correct formula
    CONST = mock_config["constants"]["LSOL_TO_ERG"] / (
        mock_config["constants"]["MPC_TO_CM"] ** 2
    )
    FACTOR = (
        CONST
        / (4 * jnp.pi * observation_lum_dist**2)
        / (1 + observation_z)
        / pixel_size
    )
    expected_flux = luminosity * FACTOR

    print("Computed Flux:", flux)
    print("Expected Flux:", expected_flux)
    assert jnp.allclose(flux, expected_flux, rtol=1e-5)


def test_velocity_doppler_shift():
    wavelength = jnp.array([5000.0, 6000.0, 7000.0])
    velocity = jnp.array([[300.0, 400.0, 500.0], [600.0, 700.0, 800.0]])
    doppler_shifted_wavelength = velocity_doppler_shift(
        wavelength,
        velocity,
        direction="y",
        SPEED_OF_LIGHT=mock_config["constants"]["SPEED_OF_LIGHT"],
    )
    # Calculate expected shifted wavelengths correctly
    expected_shifted_wavelength = jnp.array(
        [
            wavelength
            * jnp.exp(velocity[0, 1] / mock_config["constants"]["SPEED_OF_LIGHT"]),
            wavelength
            * jnp.exp(velocity[1, 1] / mock_config["constants"]["SPEED_OF_LIGHT"]),
        ]
    )
    print("Computed Doppler Shifted Wavelength:", doppler_shifted_wavelength)
    print("Expected Doppler Shifted Wavelength:", expected_shifted_wavelength)
    assert jnp.allclose(
        doppler_shifted_wavelength, expected_shifted_wavelength, rtol=1e-5
    )


def test_resample_spectrum():
    initial_spectrum = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    initial_wavelength = jnp.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0])
    target_wavelength = jnp.array([4500.0, 5500.0, 6500.0, 7500.0])

    resampled_spectrum = resample_spectrum(
        initial_spectrum, initial_wavelength, target_wavelength
    )

    # Calculate expected resampled spectrum using correct interpolation
    in_range_mask = (initial_wavelength >= jnp.min(target_wavelength)) & (
        initial_wavelength <= jnp.max(target_wavelength)
    )
    intrinsic_wave_diff = calculate_diff(initial_wavelength) * in_range_mask
    total_lum = jnp.sum(initial_spectrum * intrinsic_wave_diff)
    particle_lum = jnp.interp(target_wavelength, initial_wavelength, initial_spectrum)
    new_total_lum = jnp.sum(particle_lum * calculate_diff(target_wavelength))
    scale_factor = total_lum / new_total_lum
    expected_resampled_spectrum = particle_lum * scale_factor

    print("Computed Resampled Spectrum:", resampled_spectrum)
    print("Expected Resampled Spectrum:", expected_resampled_spectrum)
    assert jnp.allclose(resampled_spectrum, expected_resampled_spectrum, rtol=1e-5)
    # assert that it does not contain nan values
    assert not jnp.any(jnp.isnan(resampled_spectrum))


def test_resample_spectrum_if_spec_is_zero():
    initial_spectrum = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 0
    initial_wavelength = jnp.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0])
    target_wavelength = jnp.array([4500.0, 5500.0, 6500.0, 7500.0])

    resampled_spectrum = resample_spectrum(
        initial_spectrum, initial_wavelength, target_wavelength
    )

    # Calculate expected resampled spectrum using correct interpolation
    in_range_mask = (initial_wavelength >= jnp.min(target_wavelength)) & (
        initial_wavelength <= jnp.max(target_wavelength)
    )
    intrinsic_wave_diff = calculate_diff(initial_wavelength) * in_range_mask
    total_lum = jnp.sum(initial_spectrum * intrinsic_wave_diff)
    particle_lum = jnp.interp(target_wavelength, initial_wavelength, initial_spectrum)
    new_total_lum = jnp.sum(particle_lum * calculate_diff(target_wavelength))
    scale_factor = total_lum / new_total_lum
    scale_factor = jnp.nan_to_num(scale_factor, nan=0.0)
    expected_resampled_spectrum = particle_lum * scale_factor

    print("Computed Resampled Spectrum:", resampled_spectrum)
    print("Expected Resampled Spectrum:", expected_resampled_spectrum)
    assert jnp.allclose(resampled_spectrum, expected_resampled_spectrum, rtol=1e-5)
    # assert that it does not contain nan values
    assert not jnp.any(jnp.isnan(resampled_spectrum))
    assert (resampled_spectrum == 0).all()


def test_calculate_cube():
    # Test data
    spectra = jnp.array(
        [[100, 200, 300], [400, 500, 600], [700, 800, 900], [1, 2, 3]],
        dtype=jnp.float32,
    )
    spaxel_index = jnp.array([0, 1, 1, 3], dtype=jnp.int32)
    num_spaxels = 2

    # Expected result
    expected_cube = jnp.array(
        [[[100, 200, 300], [1100, 1300, 1500]], [[0, 0, 0], [1, 2, 3]]]
    )

    # Call the function
    result_cube = calculate_cube(spectra, spaxel_index, num_spaxels)

    print("Expected Cube:", expected_cube)
    print("Computed Cube:", result_cube)

    print("Expected cube: bin 0:", expected_cube[0, 0])
    print("Computed cube: bin 0:", result_cube[0, 0])

    print("Expected cube: bin 1:", expected_cube[0, 1])
    print("Computed cube: bin 1:", result_cube[0, 1])
    # Assertion
    np.testing.assert_array_equal(result_cube, expected_cube)
