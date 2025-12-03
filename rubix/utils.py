# Description: Utility functions for Rubix
import os
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

import h5py
import jax.numpy as jnp
import yaml
from astropy.cosmology import Planck15 as cosmo

if TYPE_CHECKING:  # pragma: no cover
    from rubix.core.data import RubixData


def get_config(config: Union[str, Dict]) -> Dict:
    """
    Get the configuration from a file or a dictionary.

    Args:
        config (Union[str, Dict]):
            The configuration as a file path or a dictionary

    Returns:
        The configuration as a dictionary.
    """
    if isinstance(config, str):
        return read_yaml(config)
    else:
        return config


def get_pipeline_config(name: str) -> Dict[str, Any]:
    """
    Get the configuration of a pipeline by name.

    Args:
        name (str): The name of the pipeline to look up

    Raises:
        ValueError: If the requested pipeline is not defined

    Returns:
        Dict[str, Any]: The configuration dictionary for the pipeline
    """
    from rubix import config

    pipelines_config = config["pipelines"]

    # Get the pipeline configuration
    if name not in pipelines_config:
        raise ValueError(f"Pipeline {name} not found in the configuration")
    config = pipelines_config[name]
    return config


def read_yaml(path_to_file: str) -> Dict[str, Any]:
    """
    Read a YAML file into a dictionary.

    Args:
        path_to_file (str): Path to the YAML file to read

    Raises:
        RuntimeError: If the file cannot be read or parsed

    Returns:
        Dict[str, Any]: Contents of the YAML file
    """
    cfg = {}
    try:
        with open(path_to_file, "r") as cfgfile:
            cfg = yaml.safe_load(cfgfile)
    except Exception as e:
        raise RuntimeError(
            f"Something went wrong while reading yaml file {str(path_to_file)}"
        ) from e
    return cfg


def convert_values_to_physical(
    value: float,
    a: float,
    a_scale_exponent: float,
    hubble_param: float,
    hubble_scale_exponent: float,
    CGS_conversion_factor: float,
) -> float:
    """
    Convert values from cosmological simulations to physical units
    Source:
        https://kateharborne.github.io/SimSpin/examples/generating_hdf5.html#attributes

    Args:
        value (float): Value from Simulation Parameter to be converted
        a (float): Scale factor, given as 1/(1+z)
        a_scale_exponent (float): Exponent of the scale factor
        hubble_param (float): Hubble parameter
        hubble_scale_exponent (float): Exponent of the Hubble parameter
        CGS_conversion_factor (float): Conversion factor to CGS units

    Returns:
        float: Value in physical units
    """
    # check if CGS_conversion_factor is 0
    if CGS_conversion_factor == 0:
        # Sometimes IllustrisTNG returns 0 for the conversion factor,
        # in which case we assume it is already in CGS
        CGS_conversion_factor = 1.0
    # convert to physical units
    value = (
        value
        * a**a_scale_exponent
        * hubble_param**hubble_scale_exponent
        * CGS_conversion_factor
    )
    return value


def SFTtoAge(a: float) -> float:
    """
    Convert a scale factor to a stellar age in Gyr.

    The lookback time is calculated as the difference between the current age
    of the universe and the age at redshift $z = 1/a - 1$, giving the time
    since the formation of a star with scale factor $a$.

    Args:
        a (float): Scale factor

    Returns:
        float: Age in Gyr
    """
    # TODO maybe implement this in JAX?
    return cosmo.lookback_time((1 / a) - 1).value


def print_hdf5_file_structure(file_path: str) -> str:
    """
    Print the structure of an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file

    Returns:
        str: The structure of the HDF5 file as a string
    """
    return_string = f"File: {file_path}\n"
    with h5py.File(file_path, "r") as f:
        return_string += _print_hdf5_group_structure(f)
    return return_string


def _print_hdf5_group_structure(group, indent=0):
    return_string = ""
    for key in group.keys():
        sub_group = group[key]
        if isinstance(sub_group, h5py.Group):
            return_string += f"{' ' * indent}Group: {key}\n"
            return_string += _print_hdf5_group_structure(sub_group, indent + 4)
        else:
            return_string += (
                f"{' ' * indent}Dataset: {key} "
                f"({sub_group.dtype}) ({sub_group.shape})\n"
            )
    return return_string


def load_galaxy_data(
    path_to_file: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load galaxy data and unit metadata from an HDF5 file.

    Args:
        path_to_file (str): Path to the HDF5 file to load

    Raises:
        FileNotFoundError: If the file cannot be found

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Galaxy data and associated units

    Example:

        >>> from rubix.utils import load_galaxy_data
        >>> galaxy_data, units = load_galaxy_data("path/to/file.hdf5")
    """
    galaxy_data = {}
    units = {}
    # Check if the file exists

    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"File {str(path_to_file)} does not exist")

    with h5py.File(path_to_file, "r") as f:
        galaxy_data["subhalo_center"] = f["galaxy/center"][()]
        galaxy_data["subhalo_halfmassrad_stars"] = f["galaxy/halfmassrad_stars"][()]
        galaxy_data["redshift"] = f["galaxy/redshift"][()]

        units["galaxy"] = {}
        for key in f["galaxy"].keys():
            units["galaxy"][key] = f["galaxy"][key].attrs["unit"]
        # Load the particle data
        galaxy_data["particle_data"] = {}
        for key in f["particles"].keys():
            galaxy_data["particle_data"][key] = {}
            units[key] = {}
            for field in f["particles"][key].keys():
                galaxy_data["particle_data"][key][field] = f[
                    f"particles/{key}/{field}"
                ][()]
                units[key][field] = f[f"particles/{key}/{field}"].attrs["unit"]

    return galaxy_data, units


def _pad_particles(inputdata: "RubixData", pad: int) -> "RubixData":
    """
    Pad the particle arrays so their length is divisible by the device count.

    This is necessary for JAX sharding to succeed.

    Args:
        inputdata (RubixData): The input data containing particle arrays.
        pad (int): The number of particles to append along the first axis

    Returns:
        RubixData: The padded input data
    """

    # pad along the first axis
    inputdata.stars.coords = jnp.pad(inputdata.stars.coords, ((0, pad), (0, 0)))
    inputdata.stars.velocity = jnp.pad(inputdata.stars.velocity, ((0, pad), (0, 0)))
    inputdata.stars.mass = jnp.pad(inputdata.stars.mass, ((0, pad)))
    inputdata.stars.age = jnp.pad(inputdata.stars.age, ((0, pad)))
    inputdata.stars.metallicity = jnp.pad(inputdata.stars.metallicity, ((0, pad)))

    return inputdata
