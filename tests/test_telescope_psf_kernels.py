import jax.numpy as jnp

from rubix.telescope.psf.kernels import gaussian_kernel_2d


def test_gaussian_kernel_properties():
    # Define the size, sigma, and expected properties
    m, n, sigma = 5, 5, 1.0
    kernel = gaussian_kernel_2d(m, n, sigma)

    # Test if the output kernel has the correct shape
    assert kernel.shape == (m, n), f"Kernel shape should be ({m}, {n})"

    # Test if all elements are non-negative (property of Gaussian function)
    assert jnp.all(
        kernel >= 0
    ), "All elements in the Gaussian kernel should be non-negative"

    # Test if the sum of all elements is approximately equal to 1 (normalized)
    assert jnp.isclose(
        jnp.sum(kernel), 1
    ), "The sum of all elements in the Gaussian kernel should be close to 1"

    # Test if the center value is the maximum value (property of Gaussian distribution)
    center_value = kernel[m // 2, n // 2]
    assert center_value == jnp.max(
        kernel
    ), "The center of the Gaussian kernel should have the highest value"
