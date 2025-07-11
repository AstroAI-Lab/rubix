from typing import Callable

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.telescope.lsf.lsf import apply_lsf

from .data import RubixData
from .telescope import get_telescope


@jaxtyped(typechecker=typechecker)
def get_convolve_lsf(config: dict) -> Callable:
    """
    Get the line spread function (LSF) kernel based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        The function to convolve the datacube with the LSF kernel.

    Example
    -------
    >>> config = {
    ...     ...
    ...     "telescope": {
    ...         "name": "MUSE",
    ...         "psf": {"name": "gaussian", "size": 5, "sigma": 0.6},
    ...         "lsf": {"sigma": 0.5},
    ...         "noise": {"signal_to_noise": 1,"noise_distribution": "normal"},
    ...    },
    ...     ...
    ... }

    >>> from rubix.core.lsf import get_convolve_lsf
    >>> convolve_lsf = get_convolve_lsf(config)
    >>> rubixdata = convolve_lsf(rubixdata)
    """

    logger = get_logger(config.get("logger", None))
    # Check if key exists in config file
    if "lsf" not in config["telescope"]:
        raise ValueError("LSF configuration not found in telescope configuration")

    if "sigma" not in config["telescope"]["lsf"]:
        raise ValueError("LSF sigma size not found in telescope configuration")

    sigma = config["telescope"]["lsf"]["sigma"]

    telescope = get_telescope(config)

    wave_resolution = jnp.array(telescope.wave_res)  # Wave Resolution of the telescope

    # Define the function to convolve the datacube with the PSF kernel
    def convolve_lsf(rubixdata: RubixData) -> RubixData:
        """Convolve the input datacube with the LSF kernel using immutable operations."""
        logger.info("Convolving with LSF...")

        # Check if datacube exists
        if rubixdata.stars.datacube is None:
            logger.warning("No datacube found, skipping LSF convolution")
            return rubixdata

        # Apply LSF convolution
        convolved_datacube = apply_lsf(
            rubixdata.stars.datacube, lsf_sigma=sigma, wave_resolution=wave_resolution
        )

        # Use immutable update with .replace()
        updated_stars = rubixdata.stars.replace(datacube=convolved_datacube)
        updated_rubixdata = rubixdata.replace(stars=updated_stars)

        logger.debug(
            f"LSF convolution applied to datacube shape: {convolved_datacube.shape}"
        )
        return updated_rubixdata

    return convolve_lsf
