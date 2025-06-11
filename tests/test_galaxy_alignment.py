import jax.numpy as jnp
import numpy as np
import pytest

from rubix.galaxy.alignment import (
    apply_init_rotation,
    apply_rotation,
    center_particles,
    euler_rotation_matrix,
    moment_of_inertia_tensor,
    rotate_galaxy,
    rotation_matrix_from_inertia_tensor,
)


class MockRubixData:
    def __init__(self, galaxy, stars, gas):
        self.galaxy = galaxy
        self.stars = stars
        self.gas = gas


class MockGalaxyData:
    def __init__(self, center):
        self.center = center


class MockStarsData:
    def __init__(self, coords, velocity):
        self.coords = coords
        self.velocity = velocity


class MockGasData:
    def __init__(self, coords, velocity):
        self.coords = coords
        self.velocity = velocity


def test_center_galaxy_center_not_in_bounds():
    stellar_coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    stellar_velocities = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    center = np.array([0, 0, 0])

    mockdata = MockRubixData(
        MockGalaxyData(center=center),
        MockStarsData(coords=stellar_coordinates, velocity=stellar_velocities),
        MockGasData(coords=None, velocity=None),
    )

    with pytest.raises(
        ValueError, match="Center is not within the bounds of the galaxy"
    ):
        center_particles(mockdata, "stars")  # type: ignore


def test_center_galaxy_sucessful():
    stellar_coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    stellar_velocities = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    center = np.array([4, 5, 6])

    mockdata = MockRubixData(
        MockGalaxyData(center=center),
        MockStarsData(coords=stellar_coordinates, velocity=stellar_velocities),
        MockGasData(coords=None, velocity=None),
    )

    result = center_particles(mockdata, "stars")
    assert np.all(result.stars.coords == stellar_coordinates - center)
    assert np.all(
        result.stars.velocity
        == stellar_velocities - np.median(stellar_velocities, axis=0)
    )


def test_moment_of_inertia_tensor():
    """Test the moment_of_inertia_tensor function."""

    # Example positions and masses
    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    masses = jnp.array([1.0, 1.0, 1.0])
    halfmass_radius = 2.0

    expected_tensor = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

    result_tensor = moment_of_inertia_tensor(positions, masses, halfmass_radius)

    assert (
        result_tensor.all() == expected_tensor.all()
    ), f"Test failed. Got other tensor than expected."


def test_rotation_matrix_from_inertia_tensor():
    # Example inertia tensor: simple diagonal tensor with distinct values
    I = jnp.diag(jnp.array([1.0, 2.0, 3.0]))

    rotation_matrix = rotation_matrix_from_inertia_tensor(I)

    # Expected result
    # For a diagonal inertia tensor with distinct values, the rotation matrix
    # should effectively be the identity matrix because the eigenvectors of a
    # diagonal matrix are the standard basis vectors, assuming eigenvalues are
    # already sorted.
    expected_rotation_matrix = jnp.eye(3)

    assert (
        rotation_matrix.all() == expected_rotation_matrix.all()
    ), f"Test failed. Got other rotation matrix than expected."


def test_apply_init_rotation():
    # Example positions
    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Example rotation matrix (90-degree rotation around the z-axis)
    rotation_matrix = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # Expected rotated positions
    expected_rotated_positions = jnp.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Apply rotation using the function
    rotated_positions = apply_init_rotation(positions, rotation_matrix)

    # Verify the function output matches expected values
    assert (
        rotated_positions.all() == expected_rotated_positions.all()
    ), f"Test failed. Initial rotation was not successful."


def test_euler_rotation_matrix():
    alpha = 90.0
    beta = 0.0
    gamma = 0.0

    expected_rotation_matrix = jnp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Get the rotation matrix from the function
    result_rotation_matrix = euler_rotation_matrix(alpha, beta, gamma)

    # Check if the result matches the expected output
    assert (
        result_rotation_matrix.all() == expected_rotation_matrix.all()
    ), f"Test failed. Expected other rotation matrix {expected_rotation_matrix}, got {result_rotation_matrix}."
    # np.testing.assert_array_almost_equal(result_rotation_matrix, expected_rotation_matrix, decimal=5)


def test_apply_rotation():
    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    alpha = 90.0
    beta = 0.0
    gamma = 0.0

    rotation_matrix = jnp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Calculate the expected rotated positions
    expected_rotated_positions = jnp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Apply the rotation using the apply_rotation function
    result_rotated_positions = apply_rotation(positions, alpha, beta, gamma)

    # Verify that the result matches the expected rotated positions
    assert (
        result_rotated_positions.all() == expected_rotated_positions.all()
    ), f"Test failed. Expected other positions."


def test_rotate_galaxy():

    # Example positions, velocities, and masses
    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    velocities = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    masses = jnp.array([1.0, 1.0, 1.0])
    halfmass_radius = jnp.array([2.0])

    #    [1, 0, 0],
    #    [0, 0, -1],
    #    [0, 1, 0]

    expected_rotated_positions = jnp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    expected_rotated_velocities = jnp.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])

    alpha = 90.0
    beta = 0.0
    gamma = 0.0

    rotated_positions, rotated_velocities = rotate_galaxy(
        positions,
        velocities,
        positions,
        masses,
        halfmass_radius,
        alpha,
        beta,
        gamma,
        "IllustrisTNG",
    )

    assert rotated_positions.shape == positions.shape
    assert rotated_velocities.shape == velocities.shape

    assert (
        rotated_positions.all() == expected_rotated_positions.all()
    ), f"Test failed. Expected other positions."
    assert (
        rotated_velocities.all() == expected_rotated_velocities.all()
    ), f"Test failed. Expected other velocities."
