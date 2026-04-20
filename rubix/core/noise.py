import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Callable, Optional
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.telescope.noise.noise import (
    SUPPORTED_NOISE_DISTRIBUTIONS,
    calculate_noise_cube,
)

from .data import RubixData


def build_post_aggregation_noise_fn(
    config: dict,
) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
    """Build a post-aggregation noise function from telescope settings.

    This helper applies noise to an already aggregated datacube, which avoids
    applying noise independently on each shard before cross-device reduction.

    Args:
        config (dict): Configuration dict that includes ``telescope.noise``.

    Returns:
        Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
            Callable that maps ``(cube, key) -> noisy_cube``.

    Raises:
        ValueError: When required noise configuration keys are missing.
    """
    if "noise" not in config["telescope"]:
        raise ValueError("Noise information not provided in telescope config")

    if "signal_to_noise" not in config["telescope"]["noise"]:
        raise ValueError("Signal to noise information not provided in noise config")

    if "noise_distribution" not in config["telescope"]["noise"]:
        raise ValueError(
            "Noise distribution missing. Supported ones: "
            f"{SUPPORTED_NOISE_DISTRIBUTIONS}"
        )

    signal_to_noise = config["telescope"]["noise"]["signal_to_noise"]
    noise_distribution = config["telescope"]["noise"]["noise_distribution"]

    def apply_noise_to_cube(
        cube: jnp.ndarray, key: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        noise_cube = calculate_noise_cube(
            cube,
            signal_to_noise,
            noise_distribution=noise_distribution,
            key=key,
        )
        return cube + noise_cube

    return apply_noise_to_cube


@jaxtyped(typechecker=typechecker)
def get_apply_noise(config: dict) -> Callable[[RubixData], RubixData]:
    """Build the noise application function described by ``config``.

    Args:
        config (dict): Configuration dict that includes ``telescope.noise``.

    Returns:
        Callable[[RubixData], RubixData]: Function that adds noise to data.

    Raises:
        ValueError: When required noise configuration keys are missing.

    Example:

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
            "Noise distribution missing. Supported ones: "
            f"{SUPPORTED_NOISE_DISTRIBUTIONS}"
        )

    signal_to_noise = config["telescope"]["noise"]["signal_to_noise"]
    noise_distribution = config["telescope"]["noise"]["noise_distribution"]

    logger = get_logger()

    def apply_noise(rubixdata: RubixData) -> RubixData:
        logger.info(
            "Applying noise to datacube with signal to noise ratio: "
            f"{signal_to_noise} and noise distribution: {noise_distribution}"
        )
        datacube = rubixdata.stars.datacube

        noise_key = rubixdata.noise_key
        noise_cube = calculate_noise_cube(
            datacube,
            signal_to_noise,
            noise_distribution=noise_distribution,
            key=noise_key,
        )

        rubixdata.stars.datacube += noise_cube
        return rubixdata

    return apply_noise
