import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.signal import convolve2d

from rubix.telescope.psf.psf import apply_psf, get_psf_kernel


def test_get_psf_kernel_gaussian():
    m, n = 3, 3
    sigma = 2.0
    kernel = get_psf_kernel("gaussian", m, n, sigma=sigma)
    assert kernel.shape == (m, n)
    assert np.sum(kernel) == pytest.approx(1)


def test_get_psf_kernel_unknown():
    with pytest.raises(ValueError):
        get_psf_kernel("unknown", 3, 3)


def test_apply_psf_single_plane():
    # Create a single plane datacube and a simple PSF kernel
    datacube = jnp.zeros((10, 10, 1))
    datacube = datacube.at[5, 5, 0].set(1)  # A single bright pixel
    psf_kernel = jnp.ones((3, 3))  # Simple 3x3 kernel

    # Apply PSF
    result = apply_psf(datacube, psf_kernel)
    expected = convolve2d(datacube[:, :, 0], psf_kernel, mode="same")

    # Check if the output matches expected convolved data
    assert jnp.allclose(
        result[:, :, 0], expected
    ), "PSF application does not match expected convolution output"


def test_apply_psf_multiple_planes():
    # Create a multi-plane datacube and a simple PSF kernel
    datacube = jnp.zeros((10, 10, 3))
    datacube = datacube.at[5, 5, :].set(1)  # Bright pixel across all spectral planes
    psf_kernel = jnp.ones((3, 3))  # Simple 3x3 kernel

    # Apply PSF
    result = apply_psf(datacube, psf_kernel)
    expected_planes = [
        convolve2d(datacube[:, :, i], psf_kernel, mode="same") for i in range(3)
    ]

    # Check each spectral plane
    for i in range(3):
        assert jnp.allclose(
            result[:, :, i], expected_planes[i]
        ), f"Plane {i} PSF application does not match expected output"
