import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.galaxy import IllustrisAPI, get_input_handler
from rubix.galaxy.alignment import center_particles
from rubix.logger import get_logger
from rubix.utils import load_galaxy_data, read_yaml

# class Particles:
#    def __init__(self, particle_data: object):
#        self.particle_data = particle_data
#        self.attributes = self._filter_attributes()
#
#    def _filter_attributes(self) -> list:
#        """
#        Filters the attributes of the particle_data object based on the specified criteria.
#        """
#        return [
#            attr
#            for attr in dir(self.particle_data)
#            if not attr.startswith("__")
#            and not callable(getattr(self.particle_data, attr))
#        ]
#
#    def get_attributes(self) -> list:
#        """
#        Returns the filtered attributes.
#        """
#        return self.attributes


# class Particles:
#    def __init__(self, particle_data: object):
#        self.particle_data = particle_data
#        self.attributes = self._filter_attributes()
#
#    def _filter_attributes(self) -> list:
#        """
#        Filters the attributes of the particle_data object based on the specified criteria.
#        """
#        return [
#            attr
#            for attr in dir(self.particle_data)
#            if not attr.startswith("__")
#            and not callable(getattr(self.particle_data, attr))
#        ]
#
#    def get_attributes(self) -> list:
#        """
#        Returns the filtered attributes.
#        """
#        return self.attributes


# Registering the dataclass with JAX for automatic tree traversal
@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class Galaxy:
    """
    Dataclass for storing the galaxy data

    Args:
        redshift: Redshift of the galaxy
        center: Center coordinates of the galaxy
        halfmassrad_stars: Half mass radius of the stars in the galaxy
    """

    redshift: Optional[jnp.ndarray] = None
    center: Optional[jnp.ndarray] = None
    halfmassrad_stars: Optional[jnp.ndarray] = None

    def __repr__(self):
        representationString = ["Galaxy:"]
        for k, v in self.__dict__.items():
            if not k.endswith("_unit"):
                if v is not None:
                    attrString = f"{k}: shape = {v.shape}, dtype = {v.dtype}"
                    if hasattr(self, k + "_unit") and getattr(self, k + "_unit") != "":
                        attrString += f", unit = {getattr(self, k + '_unit')}"
                    representationString.append(attrString)
                else:
                    representationString.append(f"{k}: None")
        return "\n\t".join(representationString)

    def tree_flatten(self):
        """
        Flattens the Galaxy object into a tuple of children and auxiliary data

        Returns:
            children (tuple) - A tuple containing the redshift, center, and halfmassrad_stars

            aux_data (dict) - An empty dictionary (no auxiliary data)
        """
        children = (self.redshift, self.center, self.halfmassrad_stars)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the Galaxy object from children and auxiliary data

        Args:
            aux_data (dict): An empty dictionary (no auxiliary data)
            children (tuple): A tuple containing the redshift, center, and halfmassrad_stars

        Returns:
            The reconstructed Galaxy object.
        """
        return cls(*children)


@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class StarsData:
    """
    Dataclass for storing the stars data

    Args:
        coords: Coordinates of the stars
        velocity: Velocities of the stars
        mass: Mass of the stars
        metallicity: Metallicity of the stars
        age: Age of the stars
        pixel_assignment: Pixel assignment of the stars in the IFU grid
        spatial_bin_edges: Spatial bin edges of the IFU grid
        mask: Mask for the stars
        spectra: Spectra for each stellar particle
        datacube: IFU datacube for the stellar component

    """

    coords: Optional[jnp.ndarray] = None
    velocity: Optional[jnp.ndarray] = None
    mass: Optional[jnp.ndarray] = None
    metallicity: Optional[jnp.ndarray] = None
    age: Optional[jnp.ndarray] = None
    pixel_assignment: Optional[jnp.ndarray] = None
    spatial_bin_edges: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    spectra: Optional[jnp.ndarray] = None
    datacube: Optional[jnp.ndarray] = None

    def __repr__(self):
        representationString = ["StarsData:"]
        for k, v in self.__dict__.items():
            if not k.endswith("_unit"):
                if v is not None:
                    attrString = f"{k}: shape = {v.shape}, dtype = {v.dtype}"
                    if hasattr(self, k + "_unit") and getattr(self, k + "_unit") != "":
                        attrString += f", unit = {getattr(self, k + '_unit')}"
                    representationString.append(attrString)
                else:
                    representationString.append(f"{k}: None")
        return "\n\t".join(representationString)

    def tree_flatten(self):
        """
        Flattens the Stars object into a tuple of children and auxiliary data

        Returns:
            children (tuple) - A tuple containing the coordinates, velocity, mass, metallicity, age, pixel_assignment, spatial_bin_edges, mask, spectra, and datacube

            aux_data (dict) - An empty dictionary (no auxiliary data)
        """
        children = (
            self.coords,
            self.velocity,
            self.mass,
            self.metallicity,
            self.age,
            self.pixel_assignment,
            self.spatial_bin_edges,
            self.mask,
            self.spectra,
            self.datacube,
        )
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the Stars object from children and auxiliary data

        Args:
            aux_data (dict): An empty dictionary (no auxiliary data)
            children (tuple): A tuple containing the coordinates, velocity, mass, metallicity, age, pixel_assignment, spatial_bin_edges, mask, spectra, and datacube

        Returns:
            The reconstructed Stars object.
        """
        return cls(*children)


@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class GasData:
    """
    Dataclass for storing Gas data

    Args:
        coords: Coordinates of the gas particles
        velocity: Velocities of the gas particles
        mass: Mass of the gas particles
        density: Density of the gas particles
        internal_energy: Internal energy of the gas particles
        metallicity: Metallicity of the gas particles
        sfr: Star formation rate of the gas particles
        electron_abundance: Electron abundance of the gas particles
        pixel_assignment: Pixel assignment of the gas particles in the IFU grid
        spatial_bin_edges: Spatial bin edges of the IFU grid
        mask: Mask for the gas particles
        spectra: Spectra for each gas particle
        datacube: IFU datacube for the gas component
    """

    coords: Optional[jnp.ndarray] = None
    velocity: Optional[jnp.ndarray] = None
    mass: Optional[jnp.ndarray] = None
    density: Optional[jnp.ndarray] = None
    internal_energy: Optional[jnp.ndarray] = None
    metallicity: Optional[jnp.ndarray] = None
    metals: Optional[jnp.ndarray] = None
    sfr: Optional[jnp.ndarray] = None
    electron_abundance: Optional[jnp.ndarray] = None
    pixel_assignment: Optional[jnp.ndarray] = None
    spatial_bin_edges: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    spectra: Optional[jnp.ndarray] = None
    datacube: Optional[jnp.ndarray] = None

    def __repr__(self):
        representationString = ["GasData:"]
        for k, v in self.__dict__.items():
            if not k.endswith("_unit"):
                if v is not None:
                    attrString = f"{k}: shape = {v.shape}, dtype = {v.dtype}"
                    if hasattr(self, k + "_unit") and getattr(self, k + "_unit") != "":
                        attrString += f", unit = {getattr(self, k + '_unit')}"
                    representationString.append(attrString)
                else:
                    representationString.append(f"{k}: None")
        return "\n\t".join(representationString)

    def tree_flatten(self):
        """
        Flattens the Gas object into a tuple of children and auxiliary data

        Returns:
            children (tuple) - A tuple containing the coordinates, velocity, mass, density, internal_energy, metallicity, sfr, electron_abundance, pixel_assignment, spatial_bin_edges, mask, spectra, and datacube

            aux_data (dict) - An empty dictionary (no auxiliary data)
        """
        children = (
            self.coords,
            self.velocity,
            self.mass,
            self.density,
            self.internal_energy,
            self.metallicity,
            self.metals,
            self.sfr,
            self.electron_abundance,
            self.pixel_assignment,
            self.spatial_bin_edges,
            self.mask,
            self.spectra,
            self.datacube,
        )
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the Gas object from children and auxiliary data

        Args:
            aux_data (dict): An empty dictionary (no auxiliary data)
            children (tuple): A tuple containing the coordinates, velocity, mass, density, internal_energy, metallicity, sfr, electron_abundance, pixel_assignment, spatial_bin_edges, mask, spectra, and datacube

        Returns:
            The reconstructed Gas object.
        """
        return cls(*children)


@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class RubixData:
    """
    Dataclass for storing Rubix data. The RubixData object contains the galaxy, stars, and gas data.

    Args:
        galaxy: Galaxy object containing the galaxy data
        stars: StarsData object containing the stars data
        gas: GasData object containing the gas data
    """

    galaxy: Optional[Galaxy] = None
    stars: Optional[StarsData] = None
    gas: Optional[GasData] = None

    def __repr__(self):
        representationString = ["RubixData:"]
        for k, v in self.__dict__.items():
            representationString.append("\n\t".join(f"{k}: {v}".split("\n")))
        return "\n\t".join(representationString)

    # def __post_init__(self):
    #    if self.stars is not None:
    #        self.stars = Particles(self.stars)
    #    if self.gas is not None:
    #        self.gas = Particles(self.gas)

    def tree_flatten(self):
        """
        Flattens the RubixData object into a tuple of children and auxiliary data

        Returns:
            children (tuple) - A tuple containing the galaxy, stars, and gas objects

            aux_data (dict) - An empty dictionary (no auxiliary data)
        """
        children = (self.galaxy, self.stars, self.gas)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the RubixData object from children and auxiliary data

        Args:
            aux_data (dict): An empty dictionary (no auxiliary data)
            children (tuple): A tuple containing the galaxy, stars, and gas objects

        Returns:
            The reconstructed RubixData object.
        """
        return cls(*children)


@jaxtyped(typechecker=typechecker)
def convert_to_rubix(config: Union[dict, str]):
    """
    This function converts the data to Rubix format. The data can be loaded from an API or from a file, is then
    converted to Rubix format and saved to a file (hdf5 format). This ensures that the Rubix pipeline depends
    not on the simulation data format and basically can hndle any data.
    If the file already exists, the conversion is skipped.

    Args:
        config (dict or str): The configuration for the conversion. This can be a dictionary or a path to a YAML file containing the configuration.

    Returns:
        The configuration used for the conversion. This can be used to pass the output path to the next step in the pipeline.

    Example
    -------

    >>> import os
    >>> from rubix.core.data import convert_to_rubix

    >>> # Define the configuration (example configuration)
    >>> config = {
    ...    "logger": {
    ...        "log_level": "DEBUG",
    ...        "log_file_path": None,
    ...        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ...    },
    ...    "data": {
    ...        "name": "IllustrisAPI",
    ...        "args": {
    ...            "api_key": os.environ.get("ILLUSTRIS_API_KEY"),
    ...            "particle_type": ["stars","gas"],
    ...            "simulation": "TNG50-1",
    ...            "snapshot": 99,
    ...            "save_data_path": "data",
    ...        },
    ...        "load_galaxy_args": {
    ...            "id": 12,
    ...            "reuse": True,
    ...        },
    ...        "subset": {
    ...            "use_subset": True,
    ...            "subset_size": 1000,
    ...        },
    ...    },
    ...    "simulation": {
    ...        "name": "IllustrisTNG",
    ...        "args": {
    ...            "path": "data/galaxy-id-12.hdf5",
    ...        },
    ...    },
    ...    "output_path": "output",
    ... }

    >>> # Convert the data to Rubix format
    >>> convert_to_rubix(config)

    """
    # Check if the file already exists
    # Create the input handler based on the config and create rubix galaxy data
    if isinstance(config, str):
        config = read_yaml(config)

    logger = get_logger(config.get("logger", None))

    if os.path.exists(os.path.join(config["output_path"], "rubix_galaxy.h5")):
        logger.info("Rubix galaxy file already exists, skipping conversion")
        return config["output_path"]

    # If the simulationtype is IllustrisAPI, get data from IllustrisAPI

    # TODO: we can do this more elgantly
    if "data" in config:
        if config["data"]["name"] == "IllustrisAPI":
            logger.info("Loading data from IllustrisAPI")
            api = IllustrisAPI(**config["data"]["args"], logger=logger)
            api.load_galaxy(**config["data"]["load_galaxy_args"])

            # Load the saved data into the input handler
    logger.info("Loading data into input handler")
    input_handler = get_input_handler(config, logger=logger)
    input_handler.to_rubix(output_path=config["output_path"])

    print("Converted to Rubix format!")

    return config["output_path"]


@jaxtyped(typechecker=typechecker)
def reshape_array(arr: jax.Array) -> jax.Array:
    """Reshapes an array to be compatible with JAX parallelization

    The function reshapes an array of shape (n_particles, n_features) to an array of shape (n_gpus, particles_per_gpu, n_features).

    Padding with zero is added if necessary to ensure that the number of particles per GPU is the same for all GPUs.

    Args:
        arr (jnp.ndarray): The array to reshape

    Returns:
        The reshaped array as jnp.ndarray
    """

    n_gpus = jax.device_count()
    n_particles = arr.shape[0]

    # Check if arr is 1D or 2D
    is_1d = arr.ndim == 1

    if is_1d:
        # Convert 1D array to 2D by adding a second dimension
        arr = arr[:, None]
    # Calculate the number of particles per GPU
    particles_per_gpu = (n_particles + n_gpus - 1) // n_gpus

    # Calculate the total number of particles after padding
    total_particles = particles_per_gpu * n_gpus

    # Pad the array with zeros if necessary
    if total_particles > n_particles:
        padding = total_particles - n_particles
        arr = jnp.pad(arr, ((0, padding), (0, 0)), "constant")

    # Reshape the array to (n_gpus, particles_per_gpu, arr.shape[1])
    reshaped_arr = arr.reshape(n_gpus, particles_per_gpu, *arr.shape[1:])

    if is_1d:
        # Remove the second dimension added for 1D case
        reshaped_arr = reshaped_arr.squeeze(-1)
    return reshaped_arr


@jaxtyped(typechecker=typechecker)
def prepare_input(config: Union[dict, str]) -> RubixData:
    """
    This function prepares the input data for the pipeline. It loads the data from the file and converts it to Rubix format.

    Args:
        config (dict or str): The configuration for the conversion. This can be a dictionary or a path to a YAML file containing the configuration.

    Returns:
        The RubixData object containing the galaxy, stars, and gas data.

    Example
    -------
    >>> import os
    >>> from rubix.core.data import convert_to_rubix, prepare_input

    >>> # Define the configuration (example configuration)
    >>> config = {
    >>>            ...
    >>>           }

    >>> # Convert the data to Rubix format
    >>> convert_to_rubix(config)

    >>> # Prepare the input data
    >>> rubixdata = prepare_input(config)
    >>> # Access the galaxy data, e.g. the stellar coordintates
    >>> rubixdata.stars.coords
    """

    logger_config = config["logger"] if "logger" in config else None  # type:ignore
    logger = get_logger(logger_config)
    file_path = config["output_path"]
    file_path = os.path.join(file_path, "rubix_galaxy.h5")

    # Load the data from the file
    # TODO: maybe also pass the units here, currently this is not used
    data, units = load_galaxy_data(file_path)

    # Create the RubixData object
    rubixdata = RubixData(Galaxy(), StarsData(), GasData())

    # Set the galaxy attributes
    rubixdata.galaxy.redshift = data["redshift"]
    rubixdata.galaxy.redshift_unit = units["galaxy"]["redshift"]
    rubixdata.galaxy.center = data["subhalo_center"]
    rubixdata.galaxy.center_unit = units["galaxy"]["center"]
    rubixdata.galaxy.halfmassrad_stars = data["subhalo_halfmassrad_stars"]
    rubixdata.galaxy.halfmassrad_stars_unit = units["galaxy"]["halfmassrad_stars"]

    # Set the particle attributes
    for partType in config["data"]["args"]["particle_type"]:
        if partType in data["particle_data"]:
            # Convert attributes to JAX arrays and set them on rubixdata
            for attribute, value in data["particle_data"][partType].items():
                jax_value = jnp.array(value)
                setattr(getattr(rubixdata, partType), attribute, jax_value)
                setattr(
                    getattr(rubixdata, partType),
                    attribute + "_unit",
                    units[partType][attribute],
                )

            # Center the particles
            logger.info(f"Centering {partType} particles")
            rubixdata = center_particles(rubixdata, key=partType)

            if (
                "data" in config
                and "subset" in config["data"]
                and config["data"]["subset"]["use_subset"]
            ):
                size = config["data"]["subset"]["subset_size"]
                # Randomly sample indices
                # Set random seed for reproducibility
                np.random.seed(42)
                if rubixdata.stars.coords is not None:
                    indices = np.random.choice(
                        np.arange(len(rubixdata.stars.coords)),
                        size=size,  # type:ignore
                        replace=False,
                    )  # type:ignore
                elif rubixdata.gas.coords is not None:
                    indices = np.random.choice(
                        np.arange(len(rubixdata.gas.coords)),
                        size=size,  # type:ignore
                        replace=False,
                    )
                else:
                    raise ValueError("Neither stars nor gas coordinates are available.")

                # Subset the attributes
                jax_indices = jnp.array(indices)
                for attribute in data["particle_data"][partType].keys():
                    attr_value = getattr(getattr(rubixdata, partType), attribute)
                    if attr_value.ndim == 2:  # For attributes with shape (N, 3)
                        setattr(
                            getattr(rubixdata, partType),
                            attribute,
                            attr_value[jax_indices, :],
                        )
                    else:  # For attributes with shape (N,)
                        setattr(
                            getattr(rubixdata, partType),
                            attribute,
                            attr_value[jax_indices],
                        )

                # Log the subset warning
                logger.warning(
                    f"The Subset value is set in config. Using only subset of size {size} for {partType}"
                )

    return rubixdata


@jaxtyped(typechecker=typechecker)
def get_rubix_data(config: Union[dict, str]) -> RubixData:
    """
    Returns the Rubix data

    First the function converts the data to Rubix format (``convert_to_rubix(config)``) and then prepares the input data (``prepare_input(config)``).

    Args:
        config (dict or str): The configuration for the conversion. This can be a dictionary or a path to a YAML file containing the configuration.

    Returns:
        The RubixData object containing the galaxy, stars, and gas data.
    """
    convert_to_rubix(config)
    return prepare_input(config)


@jaxtyped(typechecker=typechecker)
def process_attributes(obj: Union[StarsData, GasData], logger: logging.Logger) -> None:
    """
    Process the attributes of the given object and reshape them if they are arrays.
    """
    attributes = [attr for attr in dir(obj) if not attr.startswith("__")]
    for key in attributes:
        attr_value = getattr(obj, key)
        if attr_value is None or not isinstance(attr_value, (jnp.ndarray, np.ndarray)):
            logger.warning(f"Attribute value of {key} is None or not an array")
            continue
        reshaped_value = reshape_array(attr_value)
        setattr(obj, key, reshaped_value)


@jaxtyped(typechecker=typechecker)
def get_reshape_data(config: Union[dict, str]) -> Callable:
    """
    Returns a function to reshape the data

    Maps the `reshape_array` function to the input data dictionary.

    Args:
        config (dict or str): The configuration for the conversion. This can be a dictionary or a path to a YAML file containing the configuration.

    Returns:
        A function to reshape the data.

    Example
    -------
    >>> from rubix.core.data import get_reshape_data
    >>> reshape_data = get_reshape_data(config)
    >>> rubixdata = reshape_data(rubixdata)
    """
    # Setup a logger based on the config
    logger_config = config["logger"] if "logger" in config else None
    logger = get_logger(logger_config)

    def reshape_data(rubixdata: RubixData) -> RubixData:
        # Check if input_data has 'stars' and 'gas' attributes and process them separately
        if rubixdata.stars.coords is not None:
            process_attributes(rubixdata.stars, logger)

        if rubixdata.gas.coords is not None:
            process_attributes(rubixdata.gas, logger)

        return rubixdata

    return reshape_data
