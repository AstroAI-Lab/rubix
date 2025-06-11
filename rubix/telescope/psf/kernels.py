import jax.numpy as jnp
from jaxtyping import Array, Float


def gaussian_kernel_2d(m: int, n: int, sigma: float) -> Float[Array, "m n"]:
    """Create a 2D Gaussian kernel of size m x n with standard deviation sigma.

    The kernel is normalized so that the sum of all elements is 1.

    Parameters
    ----------
    m : int
        The number of rows in the kernel.

    n : int
        The number of columns in the kernel.

    sigma : float
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    Float[Array, "m n"]
        The 2D Gaussian kernel of size m x n with standard deviation sigma.
    """
    x = jnp.arange(-((m - 1) / 2), ((m - 1) / 2) + 1)
    y = jnp.arange(-((n - 1) / 2), ((n - 1) / 2) + 1)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    values = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    normalized = values / jnp.sum(values)
    return normalized
