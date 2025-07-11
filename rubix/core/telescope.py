from typing import Callable, Union

import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from rubix.logger import get_logger
from rubix.telescope.base import BaseTelescope
from rubix.telescope.factory import TelescopeFactory
from rubix.telescope.utils import (
    calculate_spatial_bin_edges,
    mask_particles_outside_aperture,
    square_spaxel_assignment,
)

from .cosmology import get_cosmology
from .data import RubixData, update_gas_batch, update_stars_batch


@jaxtyped(typechecker=typechecker)
def get_telescope(config: Union[str, dict]) -> BaseTelescope:
    """
    Get the telescope object based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        The telescope object.

    Example
    -------
    >>> from rubix.core.telescope import get_telescope
    >>> config = {
    ...     "telescope":
    ...         {"name": "MUSE"},
    ...     }
    >>> telescope = get_telescope(config)
    >>> print(telescope)
    """
    # TODO: this currently only loads telescope that are supported.
    # add support for custom telescopes
    factory = TelescopeFactory()
    telescope = factory.create_telescope(config["telescope"]["name"])
    if not isinstance(telescope, BaseTelescope):
        raise TypeError(f"Expected type BaseTelescope, but got {type(telescope)}")
    return telescope


@jaxtyped(typechecker=typechecker)
def get_spatial_bin_edges(config: dict) -> Float[Array, "n_bins"]:
    """
    Get the spatial bin edges based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        The spatial bin edges.
    """
    logger = get_logger(config.get("logger", None))

    logger.info("Calculating spatial bin edges...")

    telescope = get_telescope(config)
    galaxy_dist_z = config["galaxy"]["dist_z"]
    cosmology = get_cosmology(config)
    # Calculate the spatial bin edges
    # TODO: check if we need the spatial bin size somewhere? For now we dont use it
    spatial_bin_edges, spatial_bin_size = calculate_spatial_bin_edges(
        fov=telescope.fov,
        spatial_bins=telescope.sbin,
        dist_z=galaxy_dist_z,
        cosmology=cosmology,
    )

    return spatial_bin_edges


@jaxtyped(typechecker=typechecker)
def get_spaxel_assignment(config: dict) -> Callable:
    """
    Get the spaxel assignment function based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        The spaxel assignment function.

    Example
    -------
    >>> from rubix.core.telescope import get_spaxel_assignment
    >>> bin_particles = get_spaxel_assignment(config)

    >>> rubixdata = bin_particles(rubixdata)

    >>> print(rubixdata.stars.pixel_assignment)
    >>> print(rubixdata.stars.spatial_bin_edges)
    """
    logger = get_logger(config.get("logger", None))

    telescope = get_telescope(config)
    if telescope.pixel_type not in ["square"]:
        raise ValueError(f"Pixel type {telescope.pixel_type} not supported")
    spatial_bin_edges = get_spatial_bin_edges(config)

    @jaxtyped(typechecker=typechecker)
    def spaxel_assignment(rubixdata: RubixData) -> RubixData:
        logger.info("Assigning particles to spaxels...")

        # Assign stars to spaxels
        if rubixdata.stars.coords is not None:
            pixel_assignment = square_spaxel_assignment(
                rubixdata.stars.coords, spatial_bin_edges
            )

            # Use helper function for proper Equinox assignment
            rubixdata = update_stars_batch(rubixdata, pixel_assignment=pixel_assignment)

        # Assign gas to spaxels (if present)
        if rubixdata.gas is not None and rubixdata.gas.coords is not None:
            pixel_assignment = square_spaxel_assignment(
                rubixdata.gas.coords, spatial_bin_edges
            )

            # Use helper function for proper Equinox assignment
            rubixdata = update_gas_batch(rubixdata, pixel_assignment=pixel_assignment)

        return rubixdata

    return spaxel_assignment


@jaxtyped(typechecker=typechecker)
def get_filter_particles(config: dict) -> Callable:
    """
    Creates a function that filters particles using Equinox immutable operations.
    """
    logger = get_logger(config.get("logger", None))

    # Get spatial bin edges
    spatial_bin_edges = get_spatial_bin_edges(config)

    @jaxtyped(typechecker=typechecker)
    def filter_particles(rubixdata: RubixData) -> RubixData:
        logger.info("Filtering particles...")

        # Filter stars
        if rubixdata.stars.coords is not None:
            mask = mask_particles_outside_aperture(
                rubixdata.stars.coords, spatial_bin_edges
            )

            # Apply mask using helper function for proper Equinox assignment
            rubixdata = update_stars_batch(
                rubixdata,
                coords=jnp.where(mask[:, None], rubixdata.stars.coords, 0),
                velocity=jnp.where(mask[:, None], rubixdata.stars.velocity, 0),
                mass=jnp.where(mask, rubixdata.stars.mass, 0),
                age=jnp.where(mask, rubixdata.stars.age, 0),
                metallicity=jnp.where(mask, rubixdata.stars.metallicity, 0),
            )

        # Filter gas (if present)
        if rubixdata.gas is not None and rubixdata.gas.coords is not None:
            mask = mask_particles_outside_aperture(
                rubixdata.gas.coords, spatial_bin_edges
            )

            # Apply mask using helper function for proper Equinox assignment
            rubixdata = update_gas_batch(
                rubixdata,
                coords=jnp.where(mask[:, None], rubixdata.gas.coords, 0),
                mass=jnp.where(mask, rubixdata.gas.mass, 0),
                density=jnp.where(mask, rubixdata.gas.density, 0),
                metallicity=jnp.where(mask, rubixdata.gas.metallicity, 0),
            )

        return rubixdata

    return filter_particles
