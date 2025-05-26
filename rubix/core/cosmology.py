from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.cosmology import RubixCosmology
from rubix.logger import get_logger


@jaxtyped(typechecker=typechecker)
def get_cosmology(config: dict):
    """
    Get the cosmology from the configuration

    Args:
        config : Configuration dictionary

    Returns:
        RubixCosmology

    Example
    --------
    >>> config = {
    ...     ...
    ...     "cosmology":
    ...         {"name": "PLANCK15"},
    ...     ...
    ... }
    """
    logger = get_logger(config.get("logger", None))

    logger.info("Getting cosmology...")

    if config["cosmology"]["name"].upper() == "PLANCK15":
        from rubix.cosmology import PLANCK15

        return PLANCK15

    elif config["cosmology"]["name"].upper() == "CUSTOM":
        return RubixCosmology(**config["cosmology"]["args"])

    else:
        raise ValueError(
            f"Cosmology {config['cosmology']['name']} not supported. Try PLANCK15 or CUSTOM."
        )
