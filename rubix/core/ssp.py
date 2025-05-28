from typing import Callable

import jax
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.spectra.ssp.factory import get_ssp_template


@jaxtyped(typechecker=typechecker)
def get_ssp(config: dict) -> object:
    """
    This function loads the simple stellar population (SSP) template defined in the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        SSP template
    """
    # Check if field exists
    if "ssp" not in config:
        raise ValueError("Configuration does not contain 'ssp' field")

    # Check if template exists
    if "template" not in config["ssp"]:
        raise ValueError("Configuration does not contain 'template' field")
    # Check if name exists
    if "name" not in config["ssp"]["template"]:
        raise ValueError("Configuration does not contain 'name' field")
    ssp = get_ssp_template(config["ssp"]["template"]["name"])

    return ssp


@jaxtyped(typechecker=typechecker)
def get_lookup_interpolation(config: dict) -> Callable:
    """
    Loads the SSP template defined in the configuration and returns the lookup function for the template.

    The lookup function is a function that takes in the metallicity and age of a star and returns the spectrum of the star.
    This is later used to vmap over the stars metallicities and ages, and pmap over multiple GPUs.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        Lookup function for the SSP template.
    """
    logger_config = config.get("logger", None)
    logger = get_logger(logger_config)

    ssp = get_ssp(config)

    # Check if method is defined
    if "method" not in config["ssp"]:
        logger.debug("Method not defined, using default method: cubic")
        method = "cubic"
    else:
        logger.debug(f"Using method defined in config: {config['ssp']['method']}")
        method = config["ssp"]["method"]

    lookup = ssp.get_lookup_interpolation(method=method)
    return lookup


@jaxtyped(typechecker=typechecker)
def get_lookup_interpolation_vmap(config: dict) -> Callable:
    """
    This function loads the SSP template defined in the configuration and returns the lookup function for the template,
    vmapped over the stars metallicities and ages.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        vmapped lookup function for the SSP template.
    """
    lookup = get_lookup_interpolation(config)
    lookup_vmap = jax.vmap(lookup, in_axes=(0, 0))
    return lookup_vmap


@jaxtyped(typechecker=typechecker)
def get_lookup_interpolation_pmap(config: dict) -> Callable:
    """
    Get the pmap version of the lookup function for the SSP template defined in the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        pmapped lookup function for the SSP template.
    """
    lookup_vmap = get_lookup_interpolation_vmap(config)
    lookup_pmap = jax.pmap(lookup_vmap, in_axes=(0, 0))  # type: ignore
    return lookup_pmap
