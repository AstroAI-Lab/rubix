from typing import Union
from unittest.mock import MagicMock

from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from .base import BaseHandler
from .illustris import IllustrisHandler
from .pynbody import PynbodyHandler

__all__ = ["IllustrisHandler", "BaseHandler"]


@jaxtyped(typechecker=typechecker)
def get_input_handler(config: dict, logger=None) -> Union[BaseHandler, MagicMock]:
    """
    Creates a handler based on the config

    Args:
        config (dict): Configuration for the handler
        logger (Logger): Logger object

    Returns:
        BaseHandler based on the config
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
