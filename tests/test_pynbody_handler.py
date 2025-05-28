from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rubix.galaxy.input_handler.pynbody import PynbodyHandler


@pytest.fixture
def mock_config():
    """Mocked configuration for PynbodyHandler."""
    return {
        "fields": {
            "stars": {
                "age": "age",
                "mass": "mass",
                "metallicity": "metallicity",
                "coords": "pos",
                "velocity": "vel",
            },
            "gas": {"density": "density", "temperature": "temperature"},
            "dm": {"mass": "mass"},
        },
        "units": {
            "stars": {
                "coords": "kpc",
                "mass": "Msun",
                "age": "Gyr",
                "velocity": "km/s",
                "metallicity": "dimensionless",
            },
            "gas": {"density": "Msun/kpc^3", "temperature": "K"},
            "dm": {"mass": "Msun"},
            "galaxy": {
                "redshift": "dimensionless",
                "center": "kpc",
                "halfmassrad_stars": "kpc",
            },
        },
        "galaxy": {"dist_z": 0.1},
    }


@pytest.fixture
def mock_simulation():
    """Mocked simulation object that mimics a pynbody SimSnap (stars, gas, dm)."""
    mock_sim = MagicMock()

    mock_sim.stars.loadable_keys.return_value = [
        "pos",
        "mass",
        "vel",
        "metallicity",
        "age",
    ]
    mock_sim.gas.loadable_keys.return_value = ["density", "temperature"]
    mock_sim.dm.loadable_keys.return_value = ["mass"]

    star_arrays = {
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        "mass": np.array([1.0, 2.0, 3.0]),
        "vel": np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
        "metallicity": np.array([0.02, 0.03, 0.01]),
        "age": np.array([1.0, 2.0, 3.0]),
    }

    gas_arrays = {
        "density": np.array([0.1, 0.2, 0.3]),
        "temperature": np.array([100.0, 200.0, 300.0]),
    }

    dm_arrays = {"mass": np.array([10.0, 20.0, 30.0])}

    def star_getitem(key):
        return star_arrays[key]

    def gas_getitem(key):
        return gas_arrays[key]

    def dm_getitem(key):
        return dm_arrays[key]

    mock_sim.stars.__getitem__.side_effect = star_getitem
    mock_sim.gas.__getitem__.side_effect = gas_getitem
    mock_sim.dm.__getitem__.side_effect = dm_getitem

    mock_sim.stars.__len__.return_value = len(star_arrays["pos"])
    mock_sim.gas.__len__.return_value = len(gas_arrays["density"])
    mock_sim.dm.__len__.return_value = len(dm_arrays["mass"])

    mock_halos = MagicMock()
    mock_halos.__getitem__.return_value = mock_sim
    mock_sim.halos.return_value = mock_halos

    return mock_sim


@pytest.fixture
def handler_with_mock_data(mock_simulation, mock_config):
    with patch("pynbody.load", return_value=mock_simulation):
        with patch("pynbody.analysis.angmom.faceon", return_value=None):
            handler = PynbodyHandler(
                path="mock_path",
                halo_path="mock_halo_path",
                config=mock_config,
                dist_z=mock_config["galaxy"]["dist_z"],
                halo_id=1,
            )
            return handler


def test_pynbody_handler_initialization(handler_with_mock_data):
    """Test initialization of PynbodyHandler."""
    assert handler_with_mock_data is not None


def test_load_data(handler_with_mock_data):
    """Test if data is loaded correctly."""
    data = handler_with_mock_data.get_particle_data()
    assert "stars" in data


def test_get_galaxy_data(handler_with_mock_data):
    """Test retrieval of galaxy data."""
    galaxy_data = handler_with_mock_data.get_galaxy_data()
    assert galaxy_data is not None, "galaxy_data should not be None."

    expected_redshift = 0.1
    expected_center = [0, 0, 0]

    assert "redshift" in galaxy_data
    assert galaxy_data["redshift"] == expected_redshift
    assert "center" in galaxy_data
    assert galaxy_data["center"] == expected_center
    assert "halfmassrad_stars" in galaxy_data


def test_get_units(handler_with_mock_data):
    """Test if units are correctly returned."""
    units = handler_with_mock_data.get_units()
    assert "stars" in units
    assert "gas" in units
    assert "dm" in units


def test_gas_data_load(handler_with_mock_data):
    """Test loading of gas data."""
    data = handler_with_mock_data.get_particle_data()
    assert "gas" in data
    assert "density" in data["gas"]
    assert "temperature" in data["gas"]


def test_stars_data_load(handler_with_mock_data):
    """Test loading of stars data."""
    data = handler_with_mock_data.get_particle_data()
    assert "stars" in data
    assert "coords" in data["stars"]
    assert "mass" in data["stars"]
