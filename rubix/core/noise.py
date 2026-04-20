from beartype import beartype as typechecker
from beartype.typing import Any, Callable, Optional
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.telescope.noise.noise import (
    SUPPORTED_NOISE_DISTRIBUTIONS,
    calculate_noise_cube,
)

from .data import RubixData


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

    # Get the signal to noise ratio
    signal_to_noise = config["telescope"]["noise"]["signal_to_noise"]

    # Get the noise distribution
    noise_distribution = config["telescope"]["noise"]["noise_distribution"]

    logger = get_logger()

    def apply_noise(rubixdata: RubixData) -> RubixData:
        logger.info(
            "Applying noise to datacube with signal to noise ratio: "
            f"{signal_to_noise} and noise distribution: {noise_distribution}"
        )
        datacube = rubixdata.stars.datacube

        # Calculate the noise cube
        noise_key = rubixdata.noise_key
        noise_cube = calculate_noise_cube(
            datacube,
            signal_to_noise,
            noise_distribution=noise_distribution,
            key=noise_key,
        )

        # Add noise to the datacube
        rubixdata.stars.datacube += noise_cube
        return rubixdata

    return apply_noise


def build_post_aggregation_noise_fn(
    config: dict,
) -> Optional[Callable[[Any, Any], Any]]:
    """Return a callable that adds noise to a raw datacube, or ``None``.

    This is used by :py:class:`~rubix.core.pipeline.RubixPipeline` in
    stochastic mode to apply noise *once* to the fully aggregated cube after
    the cross-device reduction, avoiding the incorrect noise statistics that
    arise when noise is applied independently on each shard before the psum.

    Args:
        config (dict): Standard Rubix configuration dictionary.  Noise is
            configured under ``config["telescope"]["noise"]``.

    Returns:
        Callable[[cube, key], cube] if noise is configured, else ``None``.
    """
    noise_cfg = config.get("telescope", {}).get("noise", {})
    signal_to_noise = noise_cfg.get("signal_to_noise")

    if signal_to_noise is None:
        return None

    if "noise_distribution" not in noise_cfg:
        raise ValueError(
            "Missing required noise configuration key 'noise_distribution' "
            "under config['telescope']['noise'] when 'signal_to_noise' is set."
        )

    noise_distribution = noise_cfg["noise_distribution"]
    if noise_distribution not in SUPPORTED_NOISE_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported noise distribution '{noise_distribution}'. "
            f"Supported noise distributions are: "
            f"{sorted(SUPPORTED_NOISE_DISTRIBUTIONS)}."
        )
    logger = get_logger()

    def _apply(cube, key):
        logger.info(
            "Applying post-aggregation noise (S/N=%s, dist=%s).",
            signal_to_noise,
            noise_distribution,
        )
        noise_cube = calculate_noise_cube(
            cube,
            signal_to_noise,
            noise_distribution=noise_distribution,
            key=key,
        )
        return cube + noise_cube

    return _apply
