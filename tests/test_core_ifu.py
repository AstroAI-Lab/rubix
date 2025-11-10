import jax.numpy as jnp
import numpy as np

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.core.ifu import get_calculate_datacube_particlewise, get_telescope

RTOL = 1e-4
ATOL = 1e-6
# Sample input data
sample_inputs = {
    "metallicity": jnp.array([0.1, 0.2]),
    "age": jnp.array([1.0, 2.0]),
    "mass": jnp.array([0.5, 1.0]),
    "velocities": jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    "spectra": jnp.zeros((2, 842)),
}


# reshape sample inputs

print("Sample_inputs:")
for key in sample_inputs:
    print(f"Key: {key}, shape: {sample_inputs[key].shape}")


# Sample configuration
sample_config = {
    "pipeline": {"name": "calc_ifu"},
    "logger": {
        "log_level": "DEBUG",
        "log_file_path": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "telescope": {"name": "MUSE"},
    "cosmology": {"name": "PLANCK15"},
    "galaxy": {"dist_z": 0.1},
    "ssp": {
        "template": {"name": "BruzualCharlot2003"},
    },
}


class MockRubixData:
    def __init__(self, stars, gas):
        self.stars = stars
        self.gas = gas


class MockStarsData:
    def __init__(self, velocity, metallicity, mass, age, spectra=None):
        # self.coords = coords
        self.velocity = velocity
        self.metallicity = metallicity
        self.mass = mass
        self.age = age
        # self.pixel_assignment = pixel_assignment
        self.spectra = spectra


class MockGasData:
    def __init__(self, spectra):
        self.spectra = None


# Sample inputs for testing
initial_spectra = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
initial_wavelengths = jnp.array([[4500.0, 5500.0, 6500.0], [4500.0, 5500.0, 6500.0]])
target_wavelength = jnp.array([4000.0, 5000.0, 6000.0])


def test_get_calculate_datacube_particlewise():
    # Setup config and telescope
    config = {
        "pipeline": {"name": "calc_ifu"},
        "logger": {
            "log_level": "DEBUG",
            "log_file_path": None,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "telescope": {"name": "MUSE"},
        "cosmology": {"name": "PLANCK15"},
        "galaxy": {"dist_z": 0.1},
        "ssp": {"template": {"name": "BruzualCharlot2003"}},
    }
    telescope = get_telescope(config)
    n_spaxels = int(telescope.sbin)
    n_wave_tel = telescope.wave_seq.shape[0]
    n_particles = 3

    # Assign properties for n_particles
    # Use valid values to avoid triggering issues in SSP lookup, resampling, etc.
    metallicity = jnp.array([0.02, 0.01, 0.015])
    age = jnp.array([5.0, 8.0, 10.0])
    mass = jnp.array([1.0, 2.0, 0.5])
    velocity = jnp.array(
        [
            [100.0, 200.0, 300.0],
            [0.0, 50.0, -100.0],
            [1.0, 1.0, 1.0],
        ]
    )
    # Assign each particle to a unique spaxel
    pixel_assignment = jnp.array([0, 1, n_spaxels**2 - 1], dtype=jnp.int32)

    # Build the StarsData and RubixData object
    stars = StarsData()
    stars.metallicity = metallicity
    stars.age = age
    stars.mass = mass
    stars.velocity = velocity
    stars.pixel_assignment = pixel_assignment

    rubixdata = RubixData(galaxy=Galaxy(), stars=stars, gas=GasData())

    # Run the particlewise datacube calculation
    calc_datacube_particlewise = get_calculate_datacube_particlewise(config)
    result = calc_datacube_particlewise(rubixdata)

    # Check output
    assert hasattr(result.stars, "datacube")
    assert result.stars.datacube.shape == (n_spaxels, n_spaxels, n_wave_tel)
    # The cube must be non-negative and not NaN
    assert jnp.all(result.stars.datacube >= 0)
    assert not jnp.isnan(result.stars.datacube).any()
    # Each particle's contribution must end up in the correct spaxel
    # For a full test, you could do a partial "rebuild" as in your get_calculate_datacube test:
    flat_cube = result.stars.datacube.reshape(-1, n_wave_tel)
    # The nonzero spaxels should not be all zero (quick sanity check)
    for pix in pixel_assignment:
        assert jnp.any(flat_cube[pix] != 0)
    # All spaxels not assigned should be exactly zero
    mask = jnp.ones((n_spaxels**2,), dtype=bool)
    mask = mask.at[pixel_assignment].set(False)
    assert jnp.all(flat_cube[mask] == 0)
