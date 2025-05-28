import h5py
import jax.numpy as jnp

from rubix.debug import (  # Adjust the import based on your actual file structure
    create_dummy_rubix,
    random_data,
)
from rubix.utils import print_hdf5_file_structure


def test_random_data_shape():
    n_particles = 10
    min_val = -50
    max_val = 50
    dimension = 3
    data = random_data(n_particles, min_val, max_val, dimension)
    assert data.shape == (n_particles, dimension), "Incorrect shape for output data"


def test_random_data_bounds():
    n_particles = 10
    min_val = 0
    max_val = 1
    dimension = 1
    data = random_data(n_particles, min_val, max_val, dimension)
    assert jnp.all(data >= min_val) and jnp.all(data <= max_val), "Data out of bounds"


def test_random_data_single_dimension():
    n_particles = 5
    min_val = -1
    max_val = 1
    dimension = 1
    data = random_data(n_particles, min_val, max_val, dimension)
    assert data.shape == (n_particles,), "Output shape should be a flat array for 1D"


def test_create_dummy_rubix_file_creation(tmpdir):
    n_particles = 100
    output_path = tmpdir.join(
        "test_galaxy_output.h5"
    )  # Using pytest's tmpdir fixture to create a path
    create_dummy_rubix(n_particles, str(output_path))  # Ensuring the path is a string

    # Check if file is created
    assert output_path.check(file=1), "Output file was not created"
    print(print_hdf5_file_structure(str(output_path)))
    # Open the file and check the number of particles
    with h5py.File(output_path, "r") as f:
        print("keys:", f.keys())
        assert (
            f["particles/stars/coords"].shape[0] == n_particles
        ), "Incorrect number of particles"
