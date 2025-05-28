"""
Mainly reimplmented from SimSpin:
https://github.com/kateharborne/SimSpin/blob/4e8f0af0ebc0e43cc31729978deb3a554e039f6b/R/utilities.R#L570
"""

import jax.numpy as jnp
from jax import vmap
from jax.scipy.signal import convolve
from jaxtyping import Array, Float


def gaussian1d(x: Float[Array, " n_x"], sigma: float) -> Float[Array, " n_x"]:
    res = jnp.exp(-0.5 * (x**2) / sigma**2)
    # return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
    return res / jnp.sum(res)


def _convolve_kernel(spec, kernel, mode="full"):
    return convolve(spec, kernel, mode=mode)


def _get_kernel(sigma: float, wave_res: float, factor: int = 12):
    x = jnp.arange(-factor * wave_res, factor * wave_res + wave_res, step=wave_res)

    kernel = gaussian1d(x, sigma)
    return kernel


def apply_lsf_spectra(
    spectra: Float[Array, "n_spectra wave_bins"],
    lsf_sigma: float,
    wave_resolution: float,
    extend_factor: int = 12,
) -> Float[Array, "n_spectra wave_bins"]:
    """Apply the Line Spread Function (LSF) to multiple spectra.

    This function applies the LSF to multiple spectra in parallel using JAX's vmap.
    Currently only supports a Gaussian kernel and fixed wave resolution across all spectra and wavelenghts.

    Parameters
    ----------
    spectra : ndarray
        The input spectra to apply the LSF to.
    lsf_sigma : float
        The sigma of the LSF. Currently a Gaussian kernel.

    wave_resolution : float
        The wave resolution of the spectra.

    extend_factor : int
        The factor to extend the kernel by.

    Returns
    -------
    convolved : ndarray
        The convolved spectra.

    """
    kernel = _get_kernel(lsf_sigma, wave_resolution, factor=extend_factor)

    # Vmap the convolution across all stars
    convolved = vmap(_convolve_kernel, in_axes=(0, None))(spectra, kernel)

    end = spectra.shape[1] + kernel.shape[0] - 1 - extend_factor
    return convolved[:, extend_factor:end]


def apply_lsf(
    datacube: Float[Array, "n1 n2 wave_bins"],
    lsf_sigma: float,
    wave_resolution: float,
    extend_factor: int = 12,
) -> Float[Array, "n1 n2 wave_bins"]:
    """Apply the Line Spread Function (LSF) to a datacube.

    This function first flattens the datacube, applies the LSF to the spectra, and then reshapes the datacube back to the original shape.

    Parameters
    ----------
    datacube : ndarray
        The input datacube to apply the LSF to.
    lsf_sigma : float
        The sigma of the LSF. Currently a Gaussian kernel.

    wave_resolution : float
        The wave resolution of the spectra inside the datacube.

    extend_factor : int
        The factor to extend the kernel by.

    Returns
    -------
    convolved : ndarray
        The convolved datacube.
    """
    dimensions = datacube.shape

    # flatten the datacube
    datacube = datacube.reshape(-1, dimensions[-1])

    # Apply LSF to the spectra
    convolved = apply_lsf_spectra(datacube, lsf_sigma, wave_resolution, extend_factor)
    # Reshape back to the original shape
    # This assumes that input and output shape after convolution are the same
    return convolved.reshape(dimensions)
