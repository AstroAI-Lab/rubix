import pytest
import jax.numpy as jnp
from rubix.pipeline import transformer as transformer
from rubix.core.pipeline import RubixPipeline
from rubix.spectra.ssp.grid import SSPGrid
from rubix.telescope.base import BaseTelescope

import os  # noqa


# Dummy data functions
def dummy_get_rubix_data(config):
    return (
        jnp.array([[0, 0, 0]]),  # coords
        jnp.array([[0, 0, 0]]),  # velocities
        jnp.array([0.1]),  # metallicity
        jnp.array([1.0]),  # mass
        jnp.array([1.0]),  # age
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
    "data": {"subset": {"use_subset": True, "subset_size": 5}},
    "output_path": output_path,
    "telescope": {
        "name": "MUSE",
        "psf": {"name": "gaussian", "size": 5, "sigma": 0.6},
    },
    "cosmology": {"name": "PLANCK15"},
    "galaxy": {"dist_z": 0.1},
    "ssp": {
        "template": {"name": "BruzualCharlot2003"},
    },
}


def test_rubix_pipeline_not_implemented(setup_environment):
    config = {"pipeline": {"name": "dummy"}}
    with pytest.raises(
        ValueError, match="Pipeline dummy not found in the configuration"
    ):
        pipeline = RubixPipeline(user_config=config)  # noqa


def test_rubix_pipeline_gradient_not_implemented(setup_environment):
    pipeline = RubixPipeline(user_config=user_config)
    with pytest.raises(
        NotImplementedError, match="Gradient calculation is not implemented yet"
    ):
        pipeline.gradient()


def test_rubix_pipeline_run():
    pipeline = RubixPipeline(user_config=user_config)
    output = pipeline.run()

    # Check if output is as expected
    assert "coords" in output
    assert "velocities" in output
    assert "metallicity" in output
    assert "mass" in output
    assert "age" in output
    assert "spectra" in output

    assert isinstance(pipeline.telescope, BaseTelescope)
    assert isinstance(pipeline.ssp, SSPGrid)

    spectrum = output["spectra"]
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
        output["mass"][unique_spectra_indices],
    )
    print(
        "Ages of the spectra with NaN values: ", output["age"][unique_spectra_indices]
    )
    print(
        "Metallicities of the spectra with NaN values: ",
        output["metallicity"][unique_spectra_indices],
    )

    ssp = pipeline.ssp
    print("SSP bounds age:", ssp.age.min(), ssp.age.max())
    print("SSP bounds metallicity:", ssp.metallicity.min(), ssp.metallicity.max())

    # assert that the spectra does not contain any NaN values
    assert not jnp.isnan(spectrum).any()
