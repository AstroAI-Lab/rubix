from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np

from rubix.cosmology.base import BaseCosmology
from rubix.telescope.utils import (
    calculate_spatial_bin_edges,
    mask_particles_outside_aperture,
    square_spaxel_assignment,
)

# enfrce that jax uses cpu only

# Global flag to set a specific platform, must be used at startup.
jax.config.update("jax_platform_name", "cpu")


def test_square_spaxel_assignment():
    # Define a small set of coordinates and bin edges
    coords = jnp.array([[0.5, 1.5], [2.5, 3.5]])
    spatial_bin_edges = jnp.array([0, 1, 2, 3, 4])

    # Expected output for these inputs
    expected_output = jnp.array([4, 14])  # Expected indices in the flattened array

    # expected_output = jnp.array([5, 17])  # Expected indices in the flattened array
    # Call the function with the test inputs
    output = square_spaxel_assignment(coords, spatial_bin_edges)

    # Assert that the output is as expected
    assert jnp.all(
        output == expected_output
    ), "The function failed to assign particles to pixel positions correctly."


def test_no_particles():
    coords = jnp.array([]).reshape(0, 3)
    spatial_bin_edges = jnp.array([0, 1])
    result = mask_particles_outside_aperture(coords, spatial_bin_edges)
    assert len(result) == 0, "Should handle empty coordinate array correctly"


def test_all_particles_inside():
    coords = jnp.array([[0.5, 0.5, 0], [0.2, 0.2, 0]])
    spatial_bin_edges = jnp.array([0, 1])
    mask = mask_particles_outside_aperture(coords, spatial_bin_edges)

    n_particles_inside = jnp.sum(mask)
    assert n_particles_inside == 2, "All particles are inside the aperture"


def test_all_particles_outside():
    coords = jnp.array([[1.5, 1.5, 0], [-0.1, -0.1, 0]])
    spatial_bin_edges = jnp.array([0, 1])
    mask = mask_particles_outside_aperture(coords, spatial_bin_edges)
    n_particles_inside = jnp.sum(mask)
    assert n_particles_inside == 0, "All particles are outside the aperture"


def test_particles_on_boundary():
    coords = jnp.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    spatial_bin_edges = jnp.array([0, 1])
    mask = mask_particles_outside_aperture(coords, spatial_bin_edges)

    n_particles_inside = jnp.sum(mask)
    assert n_particles_inside == 4, "Particles on the boundary should be included"


def test_mixed_particles():
    coords = jnp.array([[0.5, 0.5, 0], [1.5, 1.5, 0], [0, 0, 0], [-0.1, -0.1, 0]])
    spatial_bin_edges = jnp.array([0, 1])
    mask = mask_particles_outside_aperture(coords, spatial_bin_edges)

    n_particles_inside = jnp.sum(mask)
    assert (
        n_particles_inside == 2
    ), "Should include particles inside and on boundary but exclude others"


def test_calculate_spatial_bin_edges():
    fov = 1.0
    spatial_bins = np.int64(10)
    dist_z = 0.5

    # Create a mock cosmology object
    cosmology = MagicMock(spec=BaseCosmology)
    cosmology.angular_scale.return_value = 2.0

    spatial_bin_edges, spatial_bin_size = calculate_spatial_bin_edges(
        fov, spatial_bins, dist_z, cosmology
    )

    # Assertions
    cosmology.angular_scale.assert_called_once_with(dist_z)
    assert spatial_bin_size == 0.2
    expected_edges = jnp.arange(-1.0, 1.0 + 0.2, 0.2)
    assert jnp.allclose(spatial_bin_edges, expected_edges)
