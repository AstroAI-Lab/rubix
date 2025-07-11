from typing import Callable

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.telescope.noise.noise import (
    SUPPORTED_NOISE_DISTRIBUTIONS,
    calculate_noise_cube,
)

from .data import RubixData


@jaxtyped(typechecker=typechecker)
def get_apply_noise(config: dict) -> Callable:
    """
    Get the function to apply noise to the datacube based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        The function to apply noise to the datacube.

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

    >>> from rubix.core.noise import get_apply_noise
    >>> apply_noise = get_apply_noise(config)
    >>> rubixdata = apply_noise(rubixdata)
    """
    if "noise" not in config["telescope"]:
        raise ValueError("Noise information not provided in telescope config")

    if "signal_to_noise" not in config["telescope"]["noise"]:
        raise ValueError("Signal to noise information not provided in noise config")

    if "noise_distribution" not in config["telescope"]["noise"]:
        raise ValueError(
            f"Noise distribution not provided in noise config. Currently supported distributions are: {SUPPORTED_NOISE_DISTRIBUTIONS}"
        )

    # Get the signal to noise ratio
    signal_to_noise = config["telescope"]["noise"]["signal_to_noise"]

    # Get the noise distribution
    noise_distribution = config["telescope"]["noise"]["noise_distribution"]
    seed = config["telescope"]["noise"].get("seed", 42)  # For reproducible results

    logger = get_logger()

    def apply_noise(rubixdata: RubixData) -> RubixData:
        """Apply noise to the input datacube using immutable operations."""
        logger.info(
            f"Applying noise to datacube with signal to noise ratio: {signal_to_noise} and noise distribution: {noise_distribution}"
        )

        # Check if datacube exists
        if rubixdata.stars.datacube is None:
            logger.warning("No datacube found, skipping noise application")
            return rubixdata

        # Generate random key for noise
        key = jax.random.PRNGKey(seed)

        datacube = rubixdata.stars.datacube
        # Define S2n for each spaxel
        S2N = jnp.ones(datacube.shape[:2]) * signal_to_noise

        # Calculate the noise cube
        noise_cube = calculate_noise_cube(
            datacube, S2N, noise_distribution=noise_distribution
        )

        # Add noise to the datacube
        # Use immutable update with .replace()
        updated_stars = rubixdata.stars.replace(datacube=datacube + noise_cube)
        updated_rubixdata = rubixdata.replace(stars=updated_stars)
        logger.debug(f"Noise applied to datacube shape: {updated_stars.datacube.shape}")

        return updated_rubixdata

    return apply_noise
