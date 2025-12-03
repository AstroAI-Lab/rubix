import logging
import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from beartype.typing import Any, Callable, Optional, Union
from jaxtyping import jaxtyped

from rubix.galaxy import IllustrisAPI, get_input_handler
from rubix.galaxy.alignment import center_particles
from rubix.logger import get_logger
from rubix.utils import load_galaxy_data, read_yaml


# Registering the dataclass with JAX for automatic tree traversal
@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class Galaxy:
    """Dataclass for storing galaxy metadata.

    Attributes:
        redshift (Optional[Any]): Redshift of the galaxy.
        center (Optional[Any]): Center coordinates of the galaxy.
        halfmassrad_stars (Optional[Any]): Half mass radius of the stars.
    """

    redshift: Optional[Any] = None
    center: Optional[Any] = None
    halfmassrad_stars: Optional[Any] = None

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
        """Flatten the Galaxy object into a tuple of children and auxiliary data for JAX traversal."""
        children = (self.redshift, self.center, self.halfmassrad_stars)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "Galaxy":
        """Reconstruct the Galaxy object from the flattened tuple of children and auxiliary data."""
        return cls(*children)


@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class StarsData:
    """Dataclass for storing the stellar component data.

    Attributes:
        coords (Optional[Any]): Coordinates of the stars.
        velocity (Optional[Any]): Velocities of the stars.
        mass (Optional[Any]): Mass of the stars.
        metallicity (Optional[Any]): Metallicity of the stars.
        age (Optional[Any]): Age of the stars.
        pixel_assignment (Optional[Any]): Pixel assignment in the IFU grid.
        spatial_bin_edges (Optional[Any]): Spatial bin edges.
        mask (Optional[Any]): Mask for the stars.
        extinction (Optional[Any]): Extinction per particle.
        spectra (Optional[Any]): Spectra for each stellar particle.
        datacube (Optional[Any]): IFU datacube of the stellar component.
    """

    coords: Optional[Any] = None
    velocity: Optional[Any] = None
    mass: Optional[Any] = None
    metallicity: Optional[Any] = None
    age: Optional[Any] = None
    pixel_assignment: Optional[Any] = None
    spatial_bin_edges: Optional[Any] = None
    mask: Optional[Any] = None
    extinction: Optional[Any] = None
    spectra: Optional[Any] = None
    datacube: Optional[Any] = None

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
        """Flatten the StarsData object for JAX tree traversal."""
        children = (
            self.coords,
            self.velocity,
            self.mass,
            self.metallicity,
            self.age,
            self.pixel_assignment,
            self.spatial_bin_edges,
            self.mask,
            self.extinction,
            self.spectra,
            self.datacube,
        )
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "StarsData":
        """Reconstruct the StarsData object from the flattened data."""
        return cls(*children)


@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class GasData:
    """Dataclass for storing the gas component data.

    Attributes:
        coords (Optional[Any]): Coordinates of the gas particles.
        velocity (Optional[Any]): Velocities of the gas particles.
        mass (Optional[Any]): Mass of the gas particles.
        density (Optional[Any]): Density of the gas particles.
        internal_energy (Optional[Any]): Internal energy values.
        metallicity (Optional[Any]): Metallicity of the gas particles.
        metals (Optional[Any]): Metal tracers attached to the particles.
        sfr (Optional[Any]): Star formation rate values.
        electron_abundance (Optional[Any]): Electron abundance values.
        pixel_assignment (Optional[Any]): Pixel assignment in the IFU grid.
        spatial_bin_edges (Optional[Any]): Spatial bin edges.
        mask (Optional[Any]): Mask data for the gas.
        spectra (Optional[Any]): Spectra for each gas particle.
        datacube (Optional[Any]): IFU datacube of the gas component.
    """

    coords: Optional[Any] = None
    velocity: Optional[Any] = None
    mass: Optional[Any] = None
    density: Optional[Any] = None
    internal_energy: Optional[Any] = None
    metallicity: Optional[Any] = None
    metals: Optional[Any] = None
    sfr: Optional[Any] = None
    electron_abundance: Optional[Any] = None
    pixel_assignment: Optional[Any] = None
    spatial_bin_edges: Optional[Any] = None
    mask: Optional[Any] = None
    spectra: Optional[Any] = None
    datacube: Optional[Any] = None

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
            children (tuple): A tuple containing the coordinates, velocity, mass, density, internal_energy, metallicity, sfr, electron_abundance, pixel_assignment, spatial_bin_edges, mask, spectra, and datacube.

            aux_data (dict): An empty dictionary (no auxiliary data).
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
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "GasData":
        """Reconstruct the GasData object from the flattened data."""
        return cls(*children)


@jaxtyped(typechecker=typechecker)
@partial(jax.tree_util.register_pytree_node_class)
@dataclass
class RubixData:
    """
    Dataclass for storing Rubix data. The RubixData object contains the galaxy, stars, and gas data.

    Attributes:
        galaxy (Optional[Galaxy]): Galaxy metadata.
        stars (Optional[StarsData]): Stellar part data.
        gas (Optional[GasData]): Gas part data.
    """

    galaxy: Optional[Galaxy] = None
    stars: Optional[StarsData] = None
    gas: Optional[GasData] = None

    def __repr__(self):
        representationString = ["RubixData:"]
        for k, v in self.__dict__.items():
            representationString.append("\n\t".join(f"{k}: {v}".split("\n")))
        return "\n\t".join(representationString)

    def tree_flatten(self):
        """Flatten the RubixData object into children and aux data."""
        children = (self.galaxy, self.stars, self.gas)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "RubixData":
        """Reconstruct the RubixData object from the flattened data."""
        return cls(*children)


@jaxtyped(typechecker=typechecker)
def convert_to_rubix(config: Union[dict, str]):
    """
    This function converts the data to Rubix format. The data can be loaded from an API or from a file, is then
    converted to Rubix format and saved to a file (hdf5 format). This ensures that the Rubix pipeline depends
    not on the simulation data format and basically can hndle any data.
    If the file already exists, the conversion is skipped.

    Args:
        config (Union[dict, str]): Configuration dict or path to a YAML file describing the conversion.

    Returns:
        str: The output directory where `rubix_galaxy.h5` is written.

    Raises:
        ValueError: When ``config['data']['name']`` is unsupported.

    Example:

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
        elif config["data"]["name"] == "NihaoHandler":  # pragma no cover
            logger.info("Loading data from Nihao simulation")
        else:
            raise ValueError(f"Unknown data source: {config['data']['name']}.")

    # Load the saved data into the input handler
    logger.info("Loading data into input handler")
    input_handler = get_input_handler(config, logger=logger)
    input_handler.to_rubix(output_path=config["output_path"])

    print("Converted to Rubix format!")

    return config["output_path"]


@jaxtyped(typechecker=typechecker)
def reshape_array(arr: jax.Array) -> jax.Array:
    """Reshape an array so it can be sharded across devices.

        The function reshapes an array of shape (n_particles, n_features) to an array of shape (n_gpus, particles_per_gpu, n_features).
        Padding with zero is added if necessary to ensure that the number of particles per GPU is the same for all GPUs.

    Args:
        arr (jax.Array): Array of shape ``(n_particles, n_features)`` that should be spread over devices.

    Returns:
        jax.Array: Array shaped as ``(n_gpus, particles_per_gpu, ...)`` suitable for JAX parallelism.
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
    """Load the converted Rubix dataset into Python objects.

    Args:
        config (Union[dict, str]): Configuration dict or path describing the conversion.

    Returns:
        RubixData: The dataset containing galaxy, stars, and gas objects.

    Raises:
        ValueError: When subset mode is enabled but neither stars nor gas coordinates exist.

    Example:

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
    rubixdata.galaxy.redshift = jnp.float64(data["redshift"])
    rubixdata.galaxy.redshift_unit = units["galaxy"]["redshift"]
    rubixdata.galaxy.center = jnp.array(data["subhalo_center"], dtype=jnp.float32)
    rubixdata.galaxy.center_unit = units["galaxy"]["center"]
    rubixdata.galaxy.halfmassrad_stars = jnp.float64(data["subhalo_halfmassrad_stars"])
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
            rubixdata = center_particles(rubixdata, partType)

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
    Returns the Rubix data.
    First the function converts the data to Rubix format (``convert_to_rubix(config)``) and then prepares the input data (``prepare_input(config)``).

    Args:
        config (Union[dict, str]):
            Configuration dict or YAML file path for conversion.

    Returns:
        RubixData:
            RubixData object containing the galaxy, stars, and gas data.
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
        config (Union[dict, str]):
            Configuration dict or path to the YAML file describing the conversion.

    Returns:
        Callable[[RubixData], RubixData]:
            Function that reshapes a `RubixData` instance.

    Example:

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
