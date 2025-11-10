import os
import warnings
from typing import Optional, Union

import numpy as np
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.telescope.apertures import (
    CIRCULAR_APERTURE,
    HEXAGONAL_APERTURE,
    SQUARE_APERTURE,
)
from rubix.telescope.base import BaseTelescope
from rubix.telescope.utils import calculate_wave_edges, calculate_wave_seq
from rubix.utils import read_yaml

PATH = os.path.dirname(os.path.abspath(__file__))
TELESCOPE_CONFIG_PATH = os.path.join(PATH, "telescopes.yaml")


class TelescopeFactory:
    @jaxtyped(typechecker=typechecker)
    def __init__(self, telescopes_config: Optional[Union[dict, str]] = None) -> None:
        if telescopes_config is None:
            warnings.warn(
                "No telescope config provided, using default stored in {}".format(
                    TELESCOPE_CONFIG_PATH
                )
            )
            self.telescopes_config = read_yaml(TELESCOPE_CONFIG_PATH)
        elif isinstance(telescopes_config, str):
            self.telescopes_config = read_yaml(telescopes_config)
        else:
            self.telescopes_config = telescopes_config

    @jaxtyped(typechecker=typechecker)
    def create_telescope(self, name: str) -> BaseTelescope:
        """
        Function to create a telescope object from the given configuration.

        Args:
            name (str): The name of the telescope to create.

        Returns:
            The telescope object as BaseTelescope.

        Example 1 (Uses the defined telescope configuration)
        -----------------------------------------------------
        >>> from rubix.telescope import TelescopeFactory
        >>> telescope_config = {
        ...     "MUSE": {
        ...         "fov": 5,
        ...         "spatial_res": 0.2,
        ...         "wave_range": [4700.15, 9351.4],
        ...         "wave_res": 1.25,
        ...         "lsf_fwhm": 2.51,
        ...         "signal_to_noise": None,
        ...         "wave_centre": 6975.775,
        ...         "aperture_type": "square",
        ...         "pixel_type": "square"
        ...     }
        ... }
        >>> factory = TelescopeFactory(telescope_config)
        >>> telescope = factory.create_telescope("MUSE")
        >>> print(telescope)

        Example 2 (Uses the default telescope configuration)
        -----------------------------------------------------
        >>> from rubix.telescope import TelescopeFactory
        >>> factory = TelescopeFactory()
        >>> telescope = factory.create_telescope("MUSE")
        >>> print(telescope)
        """
        if name not in self.telescopes_config:
            raise ValueError(f"Telescope {name} not found in config")
        config = self.telescopes_config[name]

        # Get some parameters from the config
        sbin = np.floor(config["fov"] / config["spatial_res"]).astype(int)
        aperture_region = self._get_aperture(config["aperture_type"], sbin)
        wave_seq = calculate_wave_seq(config["wave_range"], config["wave_res"])
        wave_edges = calculate_wave_edges(wave_seq, config["wave_res"])

        telescope = BaseTelescope(
            fov=config["fov"],
            spatial_res=config["spatial_res"],
            wave_range=config["wave_range"],
            wave_res=config["wave_res"],
            lsf_fwhm=config["lsf_fwhm"],
            signal_to_noise=config["signal_to_noise"],
            sbin=sbin,
            aperture_region=aperture_region,
            pixel_type=config["pixel_type"],
            wave_seq=wave_seq,
            wave_edges=wave_edges,
        )
        telescope.__class__.__name__ = name

        return telescope

    def _get_aperture(self, type, size):
        if type == "square":
            return SQUARE_APERTURE(size)
        elif type == "circular":
            return CIRCULAR_APERTURE(size)
        elif type == "hexagonal":
            return HEXAGONAL_APERTURE(size)
        else:
            raise ValueError(f"Unknown aperture type: {type}")
