import os  # noqa
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from rubix.core.data import (
    Galaxy,
    GasData,
    RubixData,
    StarsData,
)
from rubix.core.pipeline import RubixPipeline
from rubix.spectra.ssp.grid import SSPGrid
from rubix.telescope.base import BaseTelescope


# Dummy data functions
def dummy_get_rubix_data(config):
    return (
        jnp.array([[0, 0, 0]]),  # coords
        jnp.array([[0, 0, 0]]),  # velocities
        jnp.array([0.1]),  # metallicity
        jnp.array([1.0]),  # mass
        jnp.array([1.0]),  # age
        1.0,  # subhalo half mass
    )


@pytest.fixture
def setup_environment(monkeypatch):
    # Monkeypatch the necessary data functions to return dummy data
    monkeypatch.setattr("rubix.core.pipeline.get_rubix_data", dummy_get_rubix_data)


dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, "data/galaxy-id-14.hdf5")
output_path = os.path.join(dir_path, "output")
# Dummy user configuration
user_config = {
    "pipeline": {"name": "calc_ifu"},
    "logger": {
        "log_level": "DEBUG",
        "log_file_path": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "simulation": {
        "name": "IllustrisTNG",
        "args": {
            "path": file_path,
        },
    },
    "data": {
        "subset": {"use_subset": True, "subset_size": 5},
        "args": {"particle_type": ["stars"]},
    },
    "output_path": output_path,
    "telescope": {
        "name": "MUSE",
        "psf": {"name": "gaussian", "size": 5, "sigma": 0.6},
        "lsf": {"sigma": 0.6},
        "noise": {"signal_to_noise": 1, "noise_distribution": "normal"},
    },
    "cosmology": {"name": "PLANCK15"},
    "galaxy": {
        "dist_z": 0.1,
        "rotation": {
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        },
    },
    "ssp": {
        "template": {"name": "BruzualCharlot2003"},
        "dust": {
            "extinction_model": "Cardelli89",  # "Gordon23",
            "dust_to_gas_ratio": 0.01,  # need to check Remyer's paper
            "dust_to_metals_ratio": 0.4,  # do we need this ratio if we set the dust_to_gas_ratio?
            "dust_grain_density": 3.5,  # g/cm^3 #check this value
            "Rv": 3.1,
        },
    },
}


def test_rubix_pipeline_not_implemented(setup_environment):
    config = {"pipeline": {"name": "dummy"}}
    with pytest.raises(
        ValueError, match="Pipeline dummy not found in the configuration"
    ):
        pipeline = RubixPipeline(user_config=config)  # noqa


"""
def test_rubix_pipeline_gradient_not_implemented(setup_environment):
    pipeline = RubixPipeline(user_config=user_config)
    with pytest.raises(
        NotImplementedError, match="Gradient calculation is not implemented yet"
    ):
        pipeline.gradient()
"""

"""
def test_rubix_pipeline_gradient_not_implemented(setup_environment):
    mock_rubix_data = MagicMock()
    mock_rubix_data.stars.coords = jnp.array([[0, 0, 0]])
    mock_rubix_data.stars.velocities = jnp.array([[0, 0, 0]])
    mock_rubix_data.stars.metallicity = jnp.array([0.1])
    mock_rubix_data.stars.mass = jnp.array([1.0])
    mock_rubix_data.stars.age = jnp.array([1.0])

    with patch("rubix.core.pipeline.get_rubix_data", return_value=mock_rubix_data):
        pipeline = RubixPipeline(user_config=user_config)
        with pytest.raises(
            NotImplementedError, match="Gradient calculation is not implemented yet"
        ):
            pipeline.gradient()
"""


def test_rubix_pipeline_run():
    # Mock input data for the function
    input_data = RubixData(
        galaxy=Galaxy(
            redshift=jnp.array([0.1]),
            center=jnp.array([[0.0, 0.0, 0.0]]),
            halfmassrad_stars=jnp.array([1.0]),
        ),
        stars=StarsData(
            coords=jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
            velocity=jnp.array([[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]]),
            metallicity=jnp.array([0.1, 0.2]),
            mass=jnp.array([1000.0, 2000.0]),
            age=jnp.array([4.5, 5.5]),
            pixel_assignment=jnp.array([0, 1]),
        ),
        gas=GasData(velocity=None),
    )

    pipeline = RubixPipeline(user_config=user_config)
    output = pipeline.run(input_data)

    # Check if output is as expected
    assert hasattr(output.stars, "coords")
    assert hasattr(output.stars, "velocity")
    assert hasattr(output.stars, "metallicity")
    assert hasattr(output.stars, "mass")
    assert hasattr(output.stars, "age")
    assert hasattr(output.stars, "spectra")

    assert isinstance(pipeline.telescope, BaseTelescope)
    assert isinstance(pipeline.ssp, SSPGrid)

    spectrum = output.stars.spectra
    print("Spectrum shape: ", spectrum.shape)
    print("Spectrum sum: ", jnp.sum(spectrum, axis=-1))

    # Check if spectrum contains any nan values
    # Only count the numby of NaN values in the spectra
    is_nan = jnp.isnan(spectrum)
    # check whether there are any NaN values in the spectra

    indices_nan = jnp.where(is_nan)

    # Get only the unique index of the spectra with NaN values
    unique_spectra_indices = jnp.unique(indices_nan[-1])
    print("Unique indices of spectra with NaN values: ", unique_spectra_indices)
    print(
        "Masses of the spectra with NaN values: ",
        output.stars.mass[unique_spectra_indices],
    )
    print(
        "Ages of the spectra with NaN values: ",
        output.stars.age[unique_spectra_indices],
    )
    print(
        "Metallicities of the spectra with NaN values: ",
        output.stars.metallicity[unique_spectra_indices],
    )

    ssp = pipeline.ssp
    print("SSP bounds age:", ssp.age.min(), ssp.age.max())
    print("SSP bounds metallicity:", ssp.metallicity.min(), ssp.metallicity.max())

    # assert that the spectra does not contain any NaN values
    assert not jnp.isnan(spectrum).any()


def test_rubix_pipeline_run_sharded():
    # Use the number of devices to set up data that can be sharded
    num_devices = len(jax.devices())
    n_particles = num_devices if num_devices > 1 else 2  # At least two for sanity

    # Mock input data
    input_data = RubixData(
        galaxy=Galaxy(
            redshift=jnp.array([0.1]),
            center=jnp.zeros((1, 3)),
            halfmassrad_stars=jnp.array([1.0]),
        ),
        stars=StarsData(
            coords=jnp.arange(n_particles * 3, dtype=jnp.float32).reshape(n_particles, 3),
            velocity=jnp.arange(n_particles * 3, dtype=jnp.float32).reshape(n_particles, 3),
            metallicity=jnp.linspace(0.01, 0.03, n_particles),
            mass=jnp.ones(n_particles),
            age=jnp.linspace(2.0, 10.0, n_particles),
            pixel_assignment=jnp.arange(n_particles, dtype=jnp.int32),
        ),
        gas=GasData(velocity=None),
    )

    pipeline = RubixPipeline(user_config=user_config)
    output_cube = pipeline.run_sharded(input_data)

    # Output should be a jax array (the datacube)
    assert isinstance(output_cube, jax.Array)
    # Should have 3 dimensions (n_spaxels, n_spaxels, n_wave_tel)
    assert output_cube.ndim == 3
    # Should be non-negative and not NaN
    assert jnp.all(output_cube >= 0)
    assert not jnp.isnan(output_cube).any()
    # The cube should have nonzero values (sanity check)
    assert jnp.any(output_cube != 0)

    print("run_sharded output shape:", output_cube.shape)
    print("run_sharded output sum:", jnp.sum(output_cube))