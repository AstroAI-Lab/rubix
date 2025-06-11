import jax.numpy as jnp
from jax import vmap
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Float

from .kernels import gaussian_kernel_2d


def _convolve_plane(plane, kernel):
    """Convolve a single plane of a datacube with a kernel."""
    return convolve2d(plane, kernel, mode="same")


def get_psf_kernel(name: str, m: int, n: int, **kwargs) -> Float[Array, "m n"]:
    """Get a point spread function (PSF) kernel.
    Parameters
    ----------
    name : str
        The name of the PSF kernel to get.
    **kwargs
        Additional keyword arguments to pass to the PSF kernel function.
    Returns
    -------
    Float[Array, "m n"]
        The PSF kernel.
    """
    if name == "gaussian":
        return gaussian_kernel_2d(m=m, n=n, **kwargs)
    else:
        raise ValueError(f"Unknown PSF kernel name: {name}")


def apply_psf(
    datacube: Float[Array, "n_pixel n_pixel wave_bins"], psf_kernel: Float[Array, "m n"]
) -> Float[Array, "n_pixel n_pixel wave_bins"]:
    """Apply a point spread function (PSF) to the spectral datacube.

    The PSF kernel is convolved with each spectral plane of the datacube to simulate the
    blurring effect of the telescope.


    Parameters
    ----------
    datacube : Float[Array, "n_pixel n_pixel wave_bins"]
        The spectral datacube to convolve with the PSF kernel.
    psf_kernel : Float[Array, "m n"]
        The 2D PSF kernel to apply to the datacube.

    Returns
    -------
    Float[Array, "n_pixel n_pixel wave_bins"]
        The datacube convolved with the PSF kernel.
    """
    # Convolve each plane of the datacube with the PSF kernel
    # Vmap the convolution operation over the spectral dimension
    convolved = vmap(_convolve_plane, in_axes=(2, None))(datacube, psf_kernel)
    transposed = jnp.transpose(convolved, (1, 2, 0))  # Reorder to original shape
    return transposed
