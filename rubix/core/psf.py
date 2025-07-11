from typing import Callable, Dict

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.telescope.psf.psf import apply_psf, get_psf_kernel

from .data import RubixData


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

    elif config["telescope"]["psf"]["name"] == "moffat":
        # Add support for Moffat PSF
        if "size" not in config["telescope"]["psf"]:
            raise ValueError("PSF size not found in telescope configuration")
        if "fwhm" not in config["telescope"]["psf"]:
            raise ValueError("PSF FWHM not found in telescope configuration")
        if "beta" not in config["telescope"]["psf"]:
            raise ValueError("PSF beta not found in telescope configuration")

        m, n = config["telescope"]["psf"]["size"], config["telescope"]["psf"]["size"]
        fwhm = config["telescope"]["psf"]["fwhm"]
        beta = config["telescope"]["psf"]["beta"]
        psf_kernel = get_psf_kernel("moffat", m, n, fwhm=fwhm, beta=beta)

    elif config["telescope"]["psf"]["name"] == "airy":
        # Add support for Airy PSF
        if "size" not in config["telescope"]["psf"]:
            raise ValueError("PSF size not found in telescope configuration")
        if "radius" not in config["telescope"]["psf"]:
            raise ValueError("PSF radius not found in telescope configuration")

        m, n = config["telescope"]["psf"]["size"], config["telescope"]["psf"]["size"]
        radius = config["telescope"]["psf"]["radius"]
        psf_kernel = get_psf_kernel("airy", m, n, radius=radius)

    else:
        raise ValueError(
            f"Unknown PSF kernel name: {config['telescope']['psf']['name']}. "
            f"Supported: gaussian, moffat, airy"
        )

    # Convert PSF kernel to JAX array for efficiency
    psf_kernel = jnp.array(psf_kernel)

    # Define the function to convolve the datacube with the PSF kernel
    @jaxtyped(typechecker=typechecker)
    def convolve_psf(rubixdata: RubixData) -> RubixData:
        """Convolve the input datacube with the PSF kernel using immutable operations."""
        logger.info("Convolving with PSF...")

        # Check if datacube exists
        if rubixdata.stars.datacube is None:
            logger.warning("No datacube found, skipping PSF convolution")
            return rubixdata

        # Apply PSF convolution
        convolved_datacube = apply_psf(rubixdata.stars.datacube, psf_kernel)

        # Use immutable update with .replace()
        updated_stars = rubixdata.stars.replace(datacube=convolved_datacube)
        updated_rubixdata = rubixdata.replace(stars=updated_stars)

        logger.debug(
            f"PSF convolution applied to datacube shape: {convolved_datacube.shape}"
        )
        return updated_rubixdata

    return convolve_psf
