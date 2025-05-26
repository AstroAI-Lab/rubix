import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import astropy.units as u
import h5py
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from rubix import config
from rubix.logger import get_logger


@jaxtyped(typechecker=typechecker)
def create_rubix_galaxy(
    file_path: str,
    particle_data: dict,
    galaxy_data: dict,
    simulation_metadata: dict,
    units: dict,
    config: dict,
    logger: logging.Logger,
):
    """
    Create a Rubix file with the given data.

    Args:
        file_path (str): Path to save the Rubix file.
        particle_data (dict): Dictionary containing the particle data.
        galaxy_data (dict): Dictionary containing the galaxy data.
        simulation_metadata (dict): Dictionary containing the simulation metadata.
        units (dict): Dictionary containing the units.
        config (dict): Dictionary containing the configuration.
        logger (logging.Logger): Logger object to log messages.

    Returns:
        None
    """
    logger.debug("Creating Rubix file at path: %s", file_path)

    with h5py.File(file_path, "w") as f:
        # Create groups
        meta_group = f.create_group("meta")
        galaxy_group = f.create_group("galaxy")
        particle_group = f.create_group("particles")

        # Save the simulation metadata
        for key, value in simulation_metadata.items():
            meta_group.create_dataset(key, data=value)

        # Save the galaxy data: Create a dataset for each field and add the units as attributes
        for key, value in galaxy_data.items():
            logger.debug(
                f"Converting {key} for galaxy data into {config['galaxy'][key]}"
            )
            value = u.Quantity(value, units["galaxy"][key]).to(config["galaxy"][key])
            galaxy_group.create_dataset(key, data=value)
            galaxy_group[key].attrs["unit"] = config["galaxy"][key]

        # Save the particle data: Create a dataset for each field and add the units as attributes
        for key in particle_data:
            particle_group.create_group(key)
            for field, value in particle_data[key].items():
                logger.debug(
                    f"Converting {field} for particle type {key} into {config['particles'][key][field]}"
                )
                value = u.Quantity(value, units[key][field]).to(
                    config["particles"][key][field]
                )
                particle_group[key].create_dataset(field, data=value)  # type: ignore
                particle_group[key][field].attrs["unit"] = config["particles"][key][field]  # type: ignore

    logger.info(f"Rubix file saved at {file_path}")


@jaxtyped(typechecker=typechecker)
class BaseHandler(ABC):
    """
    Base class for handling input data and converting it to Rubix format.

    Args:
        config (dict): Configuration for the BaseHandler.
        _logger (logging.Logger): Logger object to log messages.
    """

    def __init__(self, logger_config=None):
        """Initializes the BaseHandler class"""
        self.config = config["BaseHandler"]
        self._logger = get_logger(logger_config)

    @abstractmethod
    def get_particle_data(self) -> dict:
        """Returns the particle data in the required format"""

    @abstractmethod
    def get_galaxy_data(self) -> dict:
        """Returns the galaxy data in the required format"""

    @abstractmethod
    def get_simulation_metadata(self) -> dict:
        """Returns the simulation meta data in the required format"""

    @abstractmethod
    def get_units(self) -> dict:
        """Returns the units in the required format"""

    def to_rubix(self, output_path: str):
        """
        Converts the input data to Rubix format and saves it to the output path.

        Args:
            output_path (str): Path to save the Rubix file.
        """
        self._logger.debug("Converting to Rubix format..")

        os.makedirs(output_path, exist_ok=True)

        # Get the data
        particle_data = self.get_particle_data()
        galaxy_data = self.get_galaxy_data()
        simulation_metadata = self.get_simulation_metadata()

        # Get the units
        units = self.get_units()

        # Check if the input data is valid and in the correct format
        self._check_data(particle_data, galaxy_data, simulation_metadata, units)

        # Create the Rubix h5 file
        file_path = os.path.join(output_path, "rubix_galaxy.h5")
        self._logger.info(f"Rubix file saved at {file_path}")

        # Create the Rubix file
        create_rubix_galaxy(
            file_path,
            particle_data,
            galaxy_data,
            simulation_metadata,
            units,
            self.config,
            self._logger,
        )

    def _check_data(self, particle_data, galaxy_data, simulation_metadata, units):
        # Check if all required fields are present
        self._check_galaxy_data(galaxy_data, units)
        self._check_particle_data(particle_data, units)
        self._check_simulation_metadata(simulation_metadata)

    def _check_simulation_metadata(self, simulation_metadata):
        """Check if all required fields are present in the simulation metadata

        Currently we do not have any required fields to check here.
        """

    def _check_galaxy_data(self, galaxy_data, units):
        # Check if all required fields are present
        for field in self.config["galaxy"]:
            if field not in galaxy_data:
                raise ValueError(f"Missing field {field} in galaxy data")
        # Check if the units are correct
        for field in galaxy_data:
            if field not in units["galaxy"]:
                raise ValueError(f"Units for {field} not found in units")

    """
    def _check_particle_data(self, particle_data, units):
        # Check if all required fields are present
        for key in self.config["particles"]:
            if key not in particle_data:
                raise ValueError(f"Missing particle type {key} in particle data")
            for field in self.config["particles"][key]:
                if field not in particle_data[key]:
                    raise ValueError(
                        f"Missing field {field} in particle data for particle type {key}"
                    )

        # Check if the units are correct
        for key in particle_data:
            for field in particle_data[key]:
                if field not in units[key]:
                    raise ValueError(f"Units for {field} not found in units")
    """

    def _check_particle_data(self, particle_data, units):
        # Get the list of expected particle types from the configuration
        expected_particle_types = list(self.config["particles"].keys())

        # Find the particle types that are actually present in the particle data
        present_particle_types = [
            key for key in expected_particle_types if key in particle_data
        ]

        # If none of the expected particle types are present, raise a ValueError
        if not present_particle_types:
            raise ValueError(
                f"None of the expected particle types {expected_particle_types} are present in particle data"
            )

        # For each particle type that is present, check that all required fields are present
        for particle_type in present_particle_types:
            required_fields = self.config["particles"][particle_type]
            for field in required_fields:
                if field not in particle_data[particle_type]:
                    raise ValueError(
                        f"Missing field {field} in particle data for particle type {particle_type}"
                    )

            # Now check if units are specified for all fields in particle_data[particle_type]
            particle_units = units.get(particle_type, {})
            for field in particle_data[particle_type]:
                if field not in particle_units:
                    raise ValueError(
                        f"Units for {field} not found in units for particle type {particle_type}"
                    )
