import jax.numpy as jnp

from rubix.telescope.lsf.lsf import apply_lsf


def test_apply_lsf_with_delta_function():
    # Parameters for the test
    n_stars = 1
    wave_bins = 100
    delta_position = 50  # Index of the delta function
    lsf_sigma = 2.0
    wave_resolution = 1.0

    # Create a spectrum with a single peak at index 50
    spectra = jnp.zeros((n_stars, wave_bins))
    spectra = spectra.at[0, delta_position].set(1)

    # Apply the LSF
    result = apply_lsf(spectra, lsf_sigma, wave_resolution)

    # Generate expected Gaussian distribution manually for comparison
    x = jnp.arange(wave_bins)
    expected_gaussian = jnp.exp(-0.5 * ((x - delta_position) ** 2) / lsf_sigma**2)
    expected_gaussian /= jnp.sum(
        expected_gaussian
    )  # Normalize to match the convolution result

    # Check if the result matches the expected Gaussian
    assert jnp.allclose(
        result[0], expected_gaussian, atol=1e-5
    ), "The convolved spectrum does not match the expected Gaussian distribution"


def test_apply_lsf_multiple_stars():
    # Parameters for the test
    n_stars = 3
    wave_bins = 100
    delta_positions = [20, 50, 75]  # Indices of the delta functions for each star
    lsf_sigma = 2.0
    wave_resolution = 1.0

    # Create spectra for multiple stars, each with a peak at different positions
    spectra = jnp.zeros((n_stars, wave_bins))
    for i, pos in enumerate(delta_positions):
        spectra = spectra.at[i, pos].set(1)

    # Apply the LSF
    results = apply_lsf(spectra, lsf_sigma, wave_resolution)

    # Check results for each star
    for i, pos in enumerate(delta_positions):
        # Generate expected Gaussian distribution for the current star
        x = jnp.arange(wave_bins)
        expected_gaussian = jnp.exp(-0.5 * ((x - pos) ** 2) / lsf_sigma**2)
        expected_gaussian /= jnp.sum(expected_gaussian)  # Normalize

        # Use np.allclose from jax.numpy to compare arrays
        assert jnp.allclose(
            results[i], expected_gaussian, atol=1e-5
        ), f"The convolved spectrum for star {i} does not match the expected Gaussian distribution"
