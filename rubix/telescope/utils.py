from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rubix.cosmology.base import BaseCosmology


@jaxtyped(typechecker=typechecker)
def calculate_spatial_bin_edges(
    fov: float, spatial_bins: np.int64, dist_z: float, cosmology: BaseCosmology
) -> Tuple[
    Union[Int[Array, "..."], Float[Array, "..."]],
    Union[float, int, Int[Array, "..."], Float[Array, "..."]],
]:
    """
    Calculate the bin edges for the spatial bins.

    Args:
        fov (float): field of view of the telescope.
        spatial_bins (float): number of spatial bins.
        dist_z (float): redshift of the galaxy.
        cosmology (BaseCosmology): cosmology object.

    Returns:
        The bin edges for the spatial bins and the spatial bin size as jnp.array.
    """
    ang_size = cosmology.angular_scale(dist_z)  # kpc/arcsec
    # fov in arcsec
    aperture_size = ang_size * fov
    spatial_bin_size = aperture_size / spatial_bins
    spatial_bin_edges = jnp.arange(
        -aperture_size / 2, aperture_size / 2 + spatial_bin_size, spatial_bin_size
    )
    return spatial_bin_edges, spatial_bin_size


@jaxtyped(typechecker=typechecker)
# TODO: check what the difference is to calculate_wave_bins
def calculate_wave_seq(wave_range: List[float], wave_res: float) -> Float[Array, "..."]:
    """
    Calculate the wavelength sequence for the wavelength bins.

    Args:
        wave_range (Tuple[float, float]): The range of the wavelength bins.
        wave_res (float): The resolution of the wavelength bins.

    Returns:
        The bin edges for the wavelength bins as jnp.array.
    """
    return jnp.arange(wave_range[0], wave_range[1], wave_res)


@jaxtyped(typechecker=typechecker)
def calculate_wave_edges(
    wave_bin_edges: Float[Array, "..."], wave_res: float
) -> Float[Array, "..."]:
    """
    Calculate the bin edges for the wavelength bins.

    Args:
        wave_bin_edges (jnp.array): The bin edges for the wavelength bins.
        wave_res (float): The resolution of the wavelength bins.

    Returns:
        The bin edges for the wavelength bins as jnp.array.
    """

    wave_start = wave_bin_edges[0] - (wave_res / 2)
    wave_end = wave_bin_edges[-1] + (wave_res / 2)
    wave_bins = jnp.arange(wave_start, wave_end, wave_res)
    return wave_bins


@jaxtyped(typechecker=typechecker)
def square_spaxel_assignment(
    coords: Union[Int[Array, "..."], Float[Array, "..."]],
    spatial_bin_edges: Union[Int[Array, "..."], Float[Array, "..."]],
) -> Int[Array, "..."]:
    """
    Bin the particle coordinates into a 2D image with the given bin edges for square pixels.

    This function takes the particle coordinates and bins them into a 2D image with the given bin edges.
    The binning is done by digitizing the x and y coordinates of the particles and then calculating the
    flat indices of the 2D image.

    The returned indexes are the pixel assignments of the particles. Indexing starts at 0.

    Args:
        coords (jnp.array): The particle coordinates.
        spatial_bin_edges (jnp.array): The bin edges for the spatial bins.

    Returns:
        The flat pixel assignments of the particles as jnp.array. Indexing starts at 0.

    Example (Assing two particles to the spatial matching bins)
    -----------------------------------------------------------
    >>> from rubix.telescope.utils import square_spaxel_assignment
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from jaxtyping import Float, Array
    >>> import jax.numpy as jnp
    >>> import numpy as np

    >>> # Define the particle coordinates
    >>> coords = np.array([[0.5, 1.5], [2.5, 3.5]])
    >>> # Define the spatial bin edges
    >>> spatial_bin_edges = np.array([0, 1, 2, 3, 4])

    >>> # Compute the pixel assignments
    >>> pixel_assignments = square_spaxel_assignment(coords, spatial_bin_edges)

    >>> # Plot the results
    ... plt.figure(figsize=(10, 5))
    ... # Plotting the particles with labels
    ... plt.subplot(1, 2, 1)
    ... scatter = plt.scatter(coords[:, 0], coords[:, 1])
    ... plt.colorbar(scatter, ticks=np.arange(0, max_assignment + 1))
    ... plt.title('Particle Coordinates and Pixel Assignments')
    ... plt.xlabel('X Coordinate')
    ... plt.ylabel('Y Coordinate')
    ... plt.xlim(spatial_bin_edges[0], spatial_bin_edges[-1])
    ... plt.ylim(spatial_bin_edges[0], spatial_bin_edges[-1])
    ... # Label each point with its pixel index
    ... for i, (x, y) in enumerate(coords[:, :2]):
    ...     plt.text(x, y, str(pixel_assignments[i]), color='red', fontsize=8)
    ... #create the bins
    ... for edge in spatial_bin_edges:
    ...     plt.axvline(edge, color='k', linestyle='--')
    ...     plt.axhline(edge, color='k', linestyle='--')
    ... plt.tight_layout()
    ... plt.show()
    """

    # Calculate assignment of of x and y coordinates to bins separately
    x_indices = (
        jnp.digitize(coords[:, 0], spatial_bin_edges) - 1
    )  # -1 to start indexing at 0
    y_indices = jnp.digitize(coords[:, 1], spatial_bin_edges) - 1

    number_of_bins = len(spatial_bin_edges) - 1

    # Clip the indices to the valid range
    x_indices = jnp.clip(x_indices, 0, number_of_bins - 1)
    y_indices = jnp.clip(y_indices, 0, number_of_bins - 1)

    # Flatten the 2D indices to 1D indices
    pixel_positions = x_indices + (number_of_bins * y_indices)
    pixel_positions = jnp.array(pixel_positions, dtype=jnp.int32)
    return pixel_positions


@jaxtyped(typechecker=typechecker)
def mask_particles_outside_aperture(
    coords: Union[Float[Array, " * 3"], Int[Array, " * 3"]],
    spatial_bin_edges: Union[Float[Array, "..."], Int[Array, "..."]],
) -> Bool[Array, "..."]:
    """
    Mask the particles that are outside the aperture.

    Args:
        coords (jnp.array): The particle coordinates.
        spatial_bin_edges (jnp.array): The bin edges for the spatial bins.

    Returns:
        A boolean mask as jnp.array that is True for particles that are inside the aperture and False for particles that are outside the aperture.
    """
    min_value = spatial_bin_edges.min()
    max_value = spatial_bin_edges.max()

    mask = (coords[:, 0] >= min_value) & (coords[:, 0] <= max_value)
    mask &= (coords[:, 1] >= min_value) & (coords[:, 1] <= max_value)

    # Filter out all the particles that are outside the aperture

    return mask


# this is implemente in the pipeline
# def filter_particles_outside_aperture(
#     coords: Float[Array, " n_stars 3"],
#     masses: Float[Array, " n_stars"],
#     metallicities: Float[Array, " n_stars"],
#     ages: Float[Array, " n_stars"],
#     spatial_bin_edges: Float[Array, " n_bins"],
# ):
#     """Filter the particles that are outside the aperture by setting their masses, metallicities and ages to zero."""
#     mask = mask_particles_outside_aperture(coords, spatial_bin_edges)
#
#     masses = jnp.where(mask, masses, 0)
#     metallicities = jnp.where(mask, metallicities, 0)
#     ages = jnp.where(mask, ages, 0)
#
#     return masses, metallicities, ages
#

#
# # TODO: there is a better way to to this without loops
# currently not used
# def restructure_data(masses, metallicities, ages, indices, num_pixels):
#     # Calculate the number of particles per pixel
#     # particle_count = jnp.bincount(indices, minlength=num_pixels)
#     particle_count = jnp.bincount(indices, length=num_pixels)
#
#     # Determine the maximum number of particles in any pixel
#     max_particles = particle_count.max().astype(int)
#
#     # Initialize structured arrays
#     masses_structured = jnp.zeros((num_pixels, max_particles))
#     metallicities_structured = jnp.zeros((num_pixels, max_particles))
#     ages_structured = jnp.zeros((num_pixels, max_particles))
#
#     # Process each pixel
#     for i in range(num_pixels):
#         # Find the indices of particles in this pixel
#         particle_indices = jnp.flatnonzero(indices == i)
#         num_particles_in_pixel = particle_indices.size
#
#         # Update structured arrays with data from these particles
#         if num_particles_in_pixel > 0:  # Only update if there are particles
#             masses_structured = masses_structured.at[i, :num_particles_in_pixel].set(
#                 masses[particle_indices]
#             )
#             metallicities_structured = metallicities_structured.at[
#                 i, :num_particles_in_pixel
#             ].set(metallicities[particle_indices])
#             ages_structured = ages_structured.at[i, :num_particles_in_pixel].set(
#                 ages[particle_indices]
#             )
#
#     return masses_structured, metallicities_structured, ages_structured
