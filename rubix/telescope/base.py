from typing import List, Optional
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker
import numpy as np

import equinox as eqx


@jaxtyped(typechecker=typechecker)
class BaseTelescope(eqx.Module):
    """
    Base class for the telescope module.
    This class contains the base parameters for the telescope module.

    Args:
        fov (float): The field of view of the telescope.
        spatial_res (float): The spatial resolution of the telescope.
        wave_range (list): The wavelength range of the telescope.
        wave_res (float): The wavelength resolution of the telescope.
        lsf_fwhm (float): The full width at half maximum of the line spread function.
        signal_to_noise (float): The signal to noise ratio of the telescope.
        sbin (int): The size of the spatial bin in each direction for the aperture mask.
        aperture_region (jnp.ndarray): The aperture region of the telescope.
        pixel_type (str): The type of pixel used in the telescope.
        wave_seq (jnp.ndarray): The wavelength sequence of the telescope.
        wave_edges (jnp.ndarray): The wavelength edges of the telescope.
    """

    fov: float
    spatial_res: float
    wave_range: List[float]  # upper and lower limits
    wave_res: float
    lsf_fwhm: float
    signal_to_noise: float
    sbin: int
    aperture_region: Float[Array, " sbin*sbin"]
    pixel_type: str
    wave_seq: Float[Array, "..."]
    wave_edges: Float[Array, "..."]
