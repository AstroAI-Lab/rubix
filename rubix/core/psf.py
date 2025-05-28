from typing import Callable, Dict

import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.telescope.psf.psf import apply_psf, get_psf_kernel

from .data import RubixData


# TODO: add option to disable PSF convolution
@jaxtyped(typechecker=typechecker)
def get_convolve_psf(config: dict) -> Callable:
    """
    Get the point spread function (PSF) kernel based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        The function to convolve the datacube with the PSF kernel.

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

    >>> from rubix.core.psf import get_convolve_psf
    >>> convolve_psf = get_convolve_psf(config)
    >>> rubixdata = convolve_psf(rubixdata)
    """

    logger = get_logger(config.get("logger", None))

    # Check if key exists in config file
    if "psf" not in config["telescope"]:
        raise ValueError("PSF configuration not found in telescope configuration")
    if "name" not in config["telescope"]["psf"]:
        raise ValueError("PSF name not found in telescope configuration")

    # Get the PSF kernel based on the configuration
    if config["telescope"]["psf"]["name"] == "gaussian":
        # Check if the PSF size and sigma are defined
        if "size" not in config["telescope"]["psf"]:
            raise ValueError("PSF size not found in telescope configuration")
        if "sigma" not in config["telescope"]["psf"]:
            raise ValueError("PSF sigma not found in telescope configuration")

        m, n = config["telescope"]["psf"]["size"], config["telescope"]["psf"]["size"]
        sigma = config["telescope"]["psf"]["sigma"]
        psf_kernel = get_psf_kernel("gaussian", m, n, sigma=sigma)

    else:
        raise ValueError(
            f"Unknown PSF kernel name: {config['telescope']['psf']['name']}"
        )

    # Define the function to convolve the datacube with the PSF kernel
    def convolve_psf(rubixdata: RubixData) -> RubixData:
        """Convolve the input datacube with the PSF kernel."""
        logger.info("Convolving with PSF...")
        rubixdata.stars.datacube = apply_psf(rubixdata.stars.datacube, psf_kernel)
        return rubixdata

    return convolve_psf
