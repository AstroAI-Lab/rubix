import logging
import os
from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped

from rubix.galaxy import IllustrisAPI, get_input_handler
from rubix.galaxy.alignment import center_particles
from rubix.logger import get_logger
from rubix.utils import load_galaxy_data, read_yaml


@jaxtyped(typechecker=typechecker)
class Galaxy(eqx.Module):
    """Minimal galaxy data structure containing only essential fields."""

    # Core galaxy properties (no unit fields - handle units in config/conversion layer)
    redshift: Optional[Float[Array, ""]] = None
    center: Optional[Float[Array, "3"]] = None
    halfmassrad_stars: Optional[Float[Array, ""]] = None


@jaxtyped(typechecker=typechecker)
class StarsData(eqx.Module):
    """Streamlined stellar particle data - only fields actually used in pipeline."""

    # Essential particle properties
    coords: Optional[Float[Array, "n_stars 3"]] = None
    velocity: Optional[Float[Array, "n_stars 3"]] = None
    mass: Optional[Float[Array, "n_stars"]] = None
    age: Optional[Float[Array, "n_stars"]] = None
    metallicity: Optional[Float[Array, "n_stars"]] = None

    # Pipeline-generated fields
    pixel_assignment: Optional[Int[Array, "n_stars"]] = None
    datacube: Optional[Float[Array, "n_x n_y n_wave"]] = None
    spectra: Optional[Float[Array, "n_stars n_wave"]] = None

    # Optional extinction information (only if dust is enabled)
    extinction_av: Optional[Float[Array, "n_stars"]] = None


@jaxtyped(typechecker=typechecker)
class GasData(eqx.Module):
    """Streamlined gas particle data - only if dust extinction is used."""

    # Essential gas properties for dust calculations
    coords: Optional[Float[Array, "n_gas 3"]] = None
    mass: Optional[Float[Array, "n_gas"]] = None
    density: Optional[Float[Array, "n_gas"]] = None
    metallicity: Optional[Float[Array, "n_gas"]] = None

    # Pipeline-generated fields (only if needed)
    pixel_assignment: Optional[Int[Array, "n_gas"]] = None


@jaxtyped(typechecker=typechecker)
class RubixData(eqx.Module):
    """Main data container - now an Equinox module with minimal fields."""

    galaxy: Galaxy
    stars: StarsData
    gas: Optional[GasData] = None  # Only include if dust extinction is enabled


# Helper functions for efficient Equinox updates using eqx.tree_at
@jaxtyped(typechecker=typechecker)
def update_stars(rubixdata: RubixData, **updates) -> RubixData:
    """Update stellar data fields using eqx.tree_at."""
    current_stars = rubixdata.stars

    # Apply updates one by one (more reliable than batch updates)
    for key, value in updates.items():
        if hasattr(current_stars, key):
            current_stars = eqx.tree_at(lambda x: getattr(x, key), current_stars, value)

    # Update the rubixdata with the new stars
    return eqx.tree_at(lambda x: x.stars, rubixdata, current_stars)


@jaxtyped(typechecker=typechecker)
def update_gas(rubixdata: RubixData, **updates) -> RubixData:
    """Update gas data fields using eqx.tree_at."""
    if rubixdata.gas is None:
        return rubixdata

    current_gas = rubixdata.gas

    # Apply updates one by one
    for key, value in updates.items():
        if hasattr(current_gas, key):
            current_gas = eqx.tree_at(lambda x: getattr(x, key), current_gas, value)

    # Update the rubixdata with the new gas
    return eqx.tree_at(lambda x: x.gas, rubixdata, current_gas)


@jaxtyped(typechecker=typechecker)
def update_galaxy(rubixdata: RubixData, **updates) -> RubixData:
    """Update galaxy data fields using eqx.tree_at."""
    current_galaxy = rubixdata.galaxy

    # Apply updates one by one
    for key, value in updates.items():
        if hasattr(current_galaxy, key):
            current_galaxy = eqx.tree_at(
                lambda x: getattr(x, key), current_galaxy, value
            )

    # Update the rubixdata with the new galaxy
    return eqx.tree_at(lambda x: x.galaxy, rubixdata, current_galaxy)


# Helper function to create lambda with proper closure
def _make_getter(field_name):
    """Create a getter function for a specific field name."""
    return lambda x: getattr(x, field_name)


# Alternative: Batch update function for better performance
@jaxtyped(typechecker=typechecker)
def update_stars_batch(rubixdata: RubixData, **updates) -> RubixData:
    """
    Update multiple star attributes at once using proper Equinox tree_at patterns.

    Args:
        rubixdata: The RubixData object to update
        **updates: Star attributes to update (coords, velocity, mass, age, etc.)

    Returns:
        Updated RubixData object
    """
    # Filter out None values and invalid attributes
    valid_updates = {}
    for key, value in updates.items():
        if value is not None and hasattr(rubixdata.stars, key):
            valid_updates[key] = value

    if not valid_updates:
        return rubixdata

    # Apply updates one by one with proper None handling
    current_stars = rubixdata.stars
    updated_stars = current_stars

    for key, value in valid_updates.items():
        # Add is_leaf parameter to handle None values correctly
        updated_stars = eqx.tree_at(
            lambda x, k=key: getattr(x, k),
            updated_stars,
            value,
            is_leaf=lambda x: x is None,  # This fixes the None handling issue
        )

    # Update the rubixdata with the new stars
    return eqx.tree_at(
        lambda x: x.stars,
        rubixdata,
        updated_stars,
        is_leaf=lambda x: x is None,  # Add this here too for consistency
    )


@jaxtyped(typechecker=typechecker)
def update_gas_batch(rubixdata: RubixData, **updates) -> RubixData:
    """
    Update multiple gas attributes at once using proper Equinox tree_at patterns.

    Args:
        rubixdata: The RubixData object to update
        **updates: Gas attributes to update (coords, mass, density, etc.)

    Returns:
        Updated RubixData object
    """
    if rubixdata.gas is None:
        return rubixdata

    # Filter out None values and invalid attributes
    valid_updates = {}
    for key, value in updates.items():
        if value is not None and hasattr(rubixdata.gas, key):
            valid_updates[key] = value

    if not valid_updates:
        return rubixdata

    # Apply updates one by one with proper None handling
    current_gas = rubixdata.gas
    updated_gas = current_gas

    for key, value in valid_updates.items():
        # Add is_leaf parameter to handle None values correctly
        updated_gas = eqx.tree_at(
            lambda x, k=key: getattr(x, k),
            updated_gas,
            value,
            is_leaf=lambda x: x is None,  # This fixes the None handling issue
        )

    # Update the rubixdata with the new gas
    return eqx.tree_at(
        lambda x: x.gas,
        rubixdata,
        updated_gas,
        is_leaf=lambda x: x is None,  # Add this here too for consistency
    )


# Data scaling and processing functions
@jaxtyped(typechecker=typechecker)
def scale_particle_data(rubixdata: RubixData, factor: int) -> RubixData:
    """
    Helper function to scale particle data for testing purposes.

    Args:
        rubixdata: Input RubixData
        factor: Scaling factor (how many times to replicate the data)

    Returns:
        Scaled RubixData with factor times more particles
    """
    if factor <= 1:
        return rubixdata

    # Scale stellar data
    if rubixdata.stars.coords is not None:
        rubixdata = update_stars_batch(
            rubixdata,
            coords=jnp.concatenate([rubixdata.stars.coords] * factor, axis=0),
            velocity=jnp.concatenate([rubixdata.stars.velocity] * factor, axis=0),
            mass=jnp.concatenate([rubixdata.stars.mass] * factor, axis=0),
            age=jnp.concatenate([rubixdata.stars.age] * factor, axis=0),
            metallicity=jnp.concatenate([rubixdata.stars.metallicity] * factor, axis=0),
        )

    # Scale gas data if present
    if rubixdata.gas is not None and rubixdata.gas.coords is not None:
        rubixdata = update_gas_batch(
            rubixdata,
            coords=jnp.concatenate([rubixdata.gas.coords] * factor, axis=0),
            mass=jnp.concatenate([rubixdata.gas.mass] * factor, axis=0),
            density=jnp.concatenate([rubixdata.gas.density] * factor, axis=0),
            metallicity=jnp.concatenate([rubixdata.gas.metallicity] * factor, axis=0),
        )

    return rubixdata


@jaxtyped(typechecker=typechecker)
def center_particles_equinox(rubixdata: RubixData, particle_type: str) -> RubixData:
    """
    Centers particles using Equinox tree operations.
    """
    if particle_type == "stars" and rubixdata.stars.coords is not None:
        # Handle potential shape issues with center
        center = rubixdata.galaxy.center
        if center.ndim == 0:
            center = jnp.array([center, center, center])
        elif center.shape == (1,):
            center = jnp.array([center[0], center[0], center[0]])

        centered_coords = rubixdata.stars.coords - center[None, :]
        return update_stars(rubixdata, coords=centered_coords)

    elif (
        particle_type == "gas"
        and rubixdata.gas is not None
        and rubixdata.gas.coords is not None
    ):
        # Handle potential shape issues with center
        center = rubixdata.galaxy.center
        if center.ndim == 0:
            center = jnp.array([center, center, center])
        elif center.shape == (1,):
            center = jnp.array([center[0], center[0], center[0]])

        centered_coords = rubixdata.gas.coords - center[None, :]
        return update_gas(rubixdata, coords=centered_coords)

    return rubixdata


@jaxtyped(typechecker=typechecker)
def apply_subset(rubixdata: RubixData, config: dict, logger) -> RubixData:
    """
    Applies subsetting to the data using the new update functions.
    """
    subset_size = config["data"]["subset"]["subset_size"]

    if rubixdata.stars.coords is not None:
        n_stars = len(rubixdata.stars.coords)
        if n_stars > subset_size:
            # Create reproducible random indices
            key = jax.random.PRNGKey(42)
            indices = jax.random.choice(
                key, n_stars, shape=(subset_size,), replace=False
            )

            # Subset all star attributes using batch update
            rubixdata = update_stars_batch(
                rubixdata,
                coords=rubixdata.stars.coords[indices],
                velocity=rubixdata.stars.velocity[indices],
                mass=rubixdata.stars.mass[indices],
                age=rubixdata.stars.age[indices],
                metallicity=rubixdata.stars.metallicity[indices],
            )
            logger.warning(f"Using subset of {subset_size} stellar particles")

    # Similar for gas if present
    if rubixdata.gas is not None and rubixdata.gas.coords is not None:
        n_gas = len(rubixdata.gas.coords)
        if n_gas > subset_size:
            key = jax.random.PRNGKey(43)
            indices = jax.random.choice(key, n_gas, shape=(subset_size,), replace=False)

            rubixdata = update_gas_batch(
                rubixdata,
                coords=rubixdata.gas.coords[indices],
                mass=rubixdata.gas.mass[indices],
                density=rubixdata.gas.density[indices],
                metallicity=rubixdata.gas.metallicity[indices],
            )
            logger.warning(f"Using subset of {subset_size} gas particles")

    return rubixdata


@jaxtyped(typechecker=typechecker)
def convert_to_rubix(config: Union[dict, str]):
    """
    Converts data to Rubix format with minimal overhead.
    """
    if isinstance(config, str):
        config = read_yaml(config)

    logger = get_logger(config.get("logger", None))
    output_file = os.path.join(config["output_path"], "rubix_galaxy.h5")

    if os.path.exists(output_file):
        logger.info("Rubix galaxy file already exists, skipping conversion")
        return config["output_path"]

    # Load data based on configuration
    if "data" in config:
        if config["data"]["name"] == "IllustrisAPI":
            logger.info("Loading data from IllustrisAPI")
            api = IllustrisAPI(**config["data"]["args"], logger=logger)
            api.load_galaxy(**config["data"]["load_galaxy_args"])

    # Convert to Rubix format
    logger.info("Converting to Rubix format")
    input_handler = get_input_handler(config, logger=logger)
    input_handler.to_rubix(output_path=config["output_path"])

    logger.info("Conversion to Rubix format completed")
    return config["output_path"]


@jaxtyped(typechecker=typechecker)
def prepare_input(config: Union[dict, str]) -> RubixData:
    """
    Prepares input data using the new minimal structure.
    """
    if isinstance(config, str):
        config = read_yaml(config)

    logger = get_logger(config.get("logger", None))

    # Create minimal data structure
    rubixdata = create_minimal_rubix_data(config)

    # Apply centering if needed
    if rubixdata.stars.coords is not None:
        logger.info("Centering stellar particles")
        rubixdata = center_particles_equinox(rubixdata, "stars")

    if rubixdata.gas is not None and rubixdata.gas.coords is not None:
        logger.info("Centering gas particles")
        rubixdata = center_particles_equinox(rubixdata, "gas")

    # Apply subsetting if configured
    if config.get("data", {}).get("subset", {}).get("use_subset", False):
        rubixdata = apply_subset(rubixdata, config, logger)

    return rubixdata


@jaxtyped(typechecker=typechecker)
def get_rubix_data(config: Union[dict, str]) -> RubixData:
    """
    Returns the Rubix data using the new minimal structure.
    """
    convert_to_rubix(config)
    return prepare_input(config)


@jaxtyped(typechecker=typechecker)
def create_minimal_rubix_data(config: dict) -> RubixData:
    """
    Creates a minimal RubixData structure with only required fields.
    Automatically converts units and optimizes memory layout.
    """
    logger = get_logger(config.get("logger", None))

    # Load raw data
    file_path = os.path.join(config["output_path"], "rubix_galaxy.h5")
    raw_data, units = load_galaxy_data(file_path)

    # Create galaxy data with proper array handling
    def safe_array_conversion(data, key):
        """Safely convert data to JAX array, handling scalars and arrays."""
        if data.get(key) is not None:
            value = data[key]
            # Handle scalar values
            if jnp.isscalar(value) or (hasattr(value, "shape") and value.shape == ()):
                return jnp.array(value)
            # Handle arrays
            elif hasattr(value, "__len__"):
                return jnp.array(value)
            else:
                return jnp.array(value)
        return None

    galaxy = Galaxy(
        redshift=safe_array_conversion(raw_data, "redshift"),
        center=safe_array_conversion(raw_data, "subhalo_center"),
        halfmassrad_stars=safe_array_conversion(raw_data, "subhalo_halfmassrad_stars"),
    )

    # Create stars data (always required)
    stars_raw = raw_data["particle_data"]["stars"]

    # Ensure all stellar data is properly converted to JAX arrays
    stars = StarsData(
        coords=jnp.asarray(stars_raw["coords"]),
        velocity=jnp.asarray(stars_raw["velocity"]),
        mass=jnp.asarray(stars_raw["mass"]),
        age=jnp.asarray(stars_raw["age"]),
        metallicity=jnp.asarray(stars_raw["metallicity"]),
    )

    # Create gas data only if dust extinction is enabled
    gas = None
    if (
        config.get("ssp", {}).get("dust", {}).get("enabled", False)
        and "gas" in raw_data["particle_data"]
    ):
        gas_raw = raw_data["particle_data"]["gas"]
        gas = GasData(
            coords=jnp.asarray(gas_raw["coords"]),
            mass=jnp.asarray(gas_raw["mass"]),
            density=jnp.asarray(gas_raw["density"]),
            metallicity=jnp.asarray(gas_raw["metallicity"]),
        )
        logger.info(
            f"Loaded {len(gas_raw['coords'])} gas particles for dust extinction"
        )

    logger.info(
        f"Created minimal RubixData with {len(stars_raw['coords'])} stellar particles"
    )

    return RubixData(galaxy=galaxy, stars=stars, gas=gas)


def _pad_particles_equinox(rubixdata: RubixData, pad_size: int) -> RubixData:
    """
    Pads particle arrays to make them divisible by the number of devices.
    Works with Equinox modules.
    """

    def pad_array(arr):
        if arr is None:
            return None
        if arr.ndim == 1:
            # 1D array - pad with zeros
            return jnp.pad(arr, (0, pad_size), mode="constant", constant_values=0)
        elif arr.ndim == 2:
            # 2D array - pad first dimension
            return jnp.pad(
                arr, ((0, pad_size), (0, 0)), mode="constant", constant_values=0
            )
        else:
            return arr

    # Apply padding to all arrays in the structure using JAX tree_map
    padded_data = jtu.tree_map(pad_array, rubixdata, is_leaf=lambda x: x is None)
    return padded_data
