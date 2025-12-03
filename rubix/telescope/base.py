import equinox as eqx
import numpy as np
from beartype import beartype as typechecker
from beartype.typing import List, Optional, Union
from jaxtyping import Array, Float, Int, jaxtyped


@jaxtyped(typechecker=typechecker)
class BaseTelescope(eqx.Module):
    """
    Base class for the telescope module.

    Attributes:
        fov (Union[float, int]): Field of view of the telescope.
        spatial_res (Union[float, int]): Spatial resolution of the telescope.
        wave_range (List[float]): Wavelength range (lower, upper).
        wave_res (Union[float, int]): Wavelength resolution.
        lsf_fwhm (Union[float, int]): Line-spread-function FWHM.
        signal_to_noise (Optional[float]): Target signal-to-noise ratio.
        sbin (np.int64): Spatial binning size (integer).
        aperture_region (Union[Float[Array, '...'], Int[Array, '...']]): Aperture array.
        pixel_type (str): Pixel geometry/type string.
        wave_seq (Float[Array, '...']): Wavelength sequence array.
        wave_edges (Float[Array, '...']): Wavelength edges array.
    """

    fov: Union[float, int]
    spatial_res: Union[float, int]
    wave_range: List[float]  # upper and lower limits
    wave_res: Union[float, int]
    lsf_fwhm: Union[float, int]
    signal_to_noise: Optional[float]
    sbin: np.int64
    aperture_region: Union[Float[Array, "..."], Int[Array, "..."]]
    pixel_type: str
    wave_seq: Float[Array, "..."]
    wave_edges: Float[Array, "..."]
