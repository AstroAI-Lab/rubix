import logging
from typing import Optional, Union
from unittest.mock import MagicMock

from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from .base import BaseHandler
from .illustris import IllustrisHandler
from .pynbody import PynbodyHandler

__all__ = ["IllustrisHandler", "BaseHandler"]


@jaxtyped(typechecker=typechecker)
def get_input_handler(
    config: dict, logger: Optional[logging.Logger] = None
) -> Union[BaseHandler, MagicMock]:
    """
    Create a handler based on the config.

    Args:
        config (dict): Configuration for the handler.
        logger (Optional[logging.Logger]): Optional logger object.

    Returns:
        BaseHandler: Handler based on the config, or a MagicMock for tests.

    Raises:
        ValueError: If the simulation type specified in the config is unsupported.
    """
    if config["simulation"]["name"] == "IllustrisTNG":
        return IllustrisHandler(**config["simulation"]["args"], logger=logger)
    elif config["simulation"]["name"] == "NIHAO":
        logger.info("Using PynbodyHandler to load a NIHAO galaxy")
        simulation_args = config["simulation"]["args"]
        if "galaxy" in config and "dist_z" in config["galaxy"]:
            simulation_args["dist_z"] = config["galaxy"]["dist_z"]
        return PynbodyHandler(**simulation_args, logger=logger)
    else:
        raise ValueError(f"Simulation {config['simulation']} is not supported")
