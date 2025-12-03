import copy
from contextlib import ExitStack
from unittest.mock import MagicMock, mock_open, patch

import astropy.units as u
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


def _build_pynbody_handler(mock_simulation, mock_config, **overrides):
    with ExitStack() as stack:
        stack.enter_context(patch("pynbody.load", return_value=mock_simulation))
        stack.enter_context(patch("pynbody.analysis.angmom.faceon", return_value=None))
        stack.enter_context(
            patch(
                "pynbody.analysis.angmom.ang_mom_vec",
                return_value=np.array([0.0, 0.0, 1.0]),
            )
        )
        stack.enter_context(
            patch(
                "pynbody.analysis.angmom.calc_sideon_matrix",
                return_value=np.eye(3),
            )
        )
        handler = PynbodyHandler(
            path="mock_path",
            halo_path=overrides.get("halo_path", "mock_halo_path"),
            rotation_path=overrides.get("rotation_path", "./data"),
            logger=overrides.get("logger"),
            config=mock_config,
            dist_z=overrides.get("dist_z", mock_config["galaxy"]["dist_z"]),
            halo_id=overrides.get("halo_id", 1),
        )
        return handler


@pytest.fixture
def handler_with_mock_data(mock_simulation, mock_config, tmp_path):
    rotation_path = tmp_path / "rotation"
    return _build_pynbody_handler(
        mock_simulation,
        mock_config,
        rotation_path=str(rotation_path),
    )


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


def test_load_config_uses_env_path():
    handler = object.__new__(PynbodyHandler)
    handler.logger = MagicMock()
    env_path = "/tmp/mock_config.yml"
    config_content = "fields: {}"
    with patch.dict(
        "os.environ",
        {"RUBIX_PYNBODY_CONFIG": env_path},
        clear=True,
    ):
        with (
            patch(
                "rubix.galaxy.input_handler.pynbody.os.path.exists",
                return_value=True,
            ),
            patch("builtins.open", mock_open(read_data=config_content)),
        ):
            config = handler._load_config()
    handler.logger.info.assert_called_with(
        f"Using environment-specified config path: {env_path}"
    )
    assert config == {"fields": {}}


def test_load_config_default_missing():
    handler = object.__new__(PynbodyHandler)
    handler.logger = MagicMock()
    with patch.dict("os.environ", {}, clear=True):
        with (
            patch(
                "rubix.galaxy.input_handler.pynbody.os.path.exists",
                return_value=False,
            ),
            pytest.raises(FileNotFoundError),
        ):
            handler._load_config()


def test_rotation_matrix_saved(mock_simulation, mock_config, tmp_path):
    logger = MagicMock()
    rotation_path = tmp_path / "rotation_saved"
    rotation_path.mkdir()
    with (
        patch(
            "rubix.galaxy.input_handler.pynbody.os.path.exists",
            return_value=True,
        ),
        patch("rubix.galaxy.input_handler.pynbody.np.save") as mock_save,
    ):
        handler = _build_pynbody_handler(
            mock_simulation,
            mock_config,
            rotation_path=str(rotation_path),
            logger=logger,
        )
    assert handler is not None
    mock_save.assert_called_once()
    logger.info.assert_any_call(
        "Rotation matrix calculated and saved to "
        f"'{rotation_path}/rotation_matrix.npy'."
    )


def test_rotation_matrix_not_saved(mock_simulation, mock_config, tmp_path):
    logger = MagicMock()
    rotation_path = tmp_path / "rotation_nosave"
    with (
        patch(
            "rubix.galaxy.input_handler.pynbody.os.path.exists",
            return_value=False,
        ),
        patch("rubix.galaxy.input_handler.pynbody.np.save") as mock_save,
    ):
        handler = _build_pynbody_handler(
            mock_simulation,
            mock_config,
            rotation_path=str(rotation_path),
            logger=logger,
        )
    assert handler is not None
    mock_save.assert_not_called()
    logger.info.assert_any_call("Rotation matrix calculated and not saved.")


def test_get_halo_data_without_halo_path(mock_simulation, mock_config):
    logger = MagicMock()
    handler = _build_pynbody_handler(
        mock_simulation,
        mock_config,
        halo_path=None,
        logger=logger,
    )
    handler.logger.warning.reset_mock()
    result = handler.get_halo_data()
    assert result is None
    handler.logger.warning.assert_called_once_with("No halo file provided or found.")


def test_get_halo_data_default_index(mock_simulation, mock_config):
    handler = _build_pynbody_handler(mock_simulation, mock_config)
    handler.sim.halos.reset_mock()
    handler.sim.halos.return_value.__getitem__.reset_mock()
    result = handler.get_halo_data(halo_id=None)
    handler.sim.halos.assert_called_once()
    handler.sim.halos.return_value.__getitem__.assert_called_once_with(0)
    assert result == handler.sim.halos.return_value.__getitem__.return_value


def test_get_galaxy_data_without_stars(mock_simulation, mock_config):
    logger = MagicMock()
    handler = _build_pynbody_handler(
        mock_simulation,
        mock_config,
        logger=logger,
    )
    handler.data.pop("stars", None)
    handler.logger.warning.reset_mock()
    galaxy_data = handler.get_galaxy_data()
    assert galaxy_data["halfmassrad_stars"] is None
    handler.logger.warning.assert_called_once_with(
        "No star data available to calculate the half-mass radius."
    )


def test_get_simulation_metadata_returns_expected_values(mock_simulation, mock_config):
    handler = _build_pynbody_handler(mock_simulation, mock_config)
    metadata = handler.get_simulation_metadata()
    assert metadata["path"] == "mock_path"
    assert metadata["halo_path"] == "mock_halo_path"
    assert "logger" in metadata


def test_calculate_halfmass_radius_handles_1d_positions():
    handler = object.__new__(PynbodyHandler)
    positions = np.array([1.0, 2.0, 3.0])
    masses = np.array([1.0, 1.0, 1.0])
    radius = handler.calculate_halfmass_radius(positions, masses)
    assert radius == 2.0


def test_get_units_warns_for_unknown_unit(mock_simulation, mock_config, tmp_path):
    logger = MagicMock()
    bad_config = copy.deepcopy(mock_config)
    bad_config["units"]["stars"]["coords"] = "NotAUnit"
    rotation_path = tmp_path / "rotation_units"
    handler = _build_pynbody_handler(
        mock_simulation,
        bad_config,
        rotation_path=str(rotation_path),
        logger=logger,
    )
    handler.logger.warning.reset_mock()
    units = handler.get_units()
    assert units["stars"]["coords"] == u.dimensionless_unscaled
    handler.logger.warning.assert_called_with(
        "Unit 'NotAUnit' for 'stars.coords' not recognized. " "Using dimensionless."
    )
