from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rubix.galaxy import IllustrisHandler
from rubix.utils import SFTtoAge

#
# @patch("os.path.exists")
# def test_init(mock_exists):
#     # Mock the os.path.exists method to always return True
#     mock_exists.return_value = True
#
#     # Mock the h5py.File method to return a MagicMock
#     with patch("h5py.File", return_value=MagicMock()) as mock_file:
#         # Test initialization
#         handler = IllustrisHandler("path", "output_path")
#         mock_file.assert_called_once_with("path", "r")
#


def create_mock_hdf5(mock_data):
    def create_mock_data():
        conversion_factors = {"a_scaling": 0.0, "h_scaling": 0.0, "to_cgs": 0}
        stars_dataset = MagicMock()
        stars_dataset.attrs = conversion_factors
        stars_dataset.__getitem__.return_value = mock_data
        return stars_dataset

    coordindates = create_mock_data()
    masses = create_mock_data()
    metallicity = create_mock_data()
    age = create_mock_data()
    velocity = create_mock_data()

    header_mock = MagicMock()
    header_mock.keys.return_value = ["Time", "HubbleParam"]
    header_mock.attrs = {
        "Time": 0.5,
        "HubbleParam": 0.7,
        "SimulationName": "MockSimulation",
        "SnapshotNumber": 1,
        "Redshift": 0.5,
        "CutoutID": "MockCutoutID",
        "CutoutRequest": "MockRequest",
    }
    # Coordinates', 'Masses', 'GFM_Metallicity', 'Velocities', 'GFM_StellarFormationTime'
    data = {
        "Header": header_mock,
        "PartType4": {
            "GFM_InitialMass": masses,
            "Coordinates": coordindates,
            "GFM_Metallicity": metallicity,
            "GFM_StellarFormationTime": age,
            "Velocities": velocity,
        },
        "SubhaloData": {
            "pos_x": np.array(1.0),
            "pos_y": np.array(2.0),
            "pos_z": np.array(3.0),
            "halfmassrad_stars": np.array(1.5),
        },
    }
    return data


def test_wrong_path():
    with pytest.raises(FileNotFoundError):
        handler = IllustrisHandler("path")


@patch("os.path.exists")
@patch("h5py.File")
def test_load_data(mock_file, mock_exists):
    # Mock the os.path.exists method to always return True
    mock_exists.return_value = True

    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)
    mock_file.return_value.__enter__.return_value = data

    # Mock the h5py.File method to return a MagicMock

    # Test _load_data method
    handler = IllustrisHandler("path")

    handler_particle_data = handler.get_particle_data()
    handler_galaxy_data = handler.get_galaxy_data()
    handler_simulation_metadata = handler.get_simulation_metadata()

    assert "stars" in handler_particle_data

    for key in handler_particle_data:
        for subkey, value in handler_particle_data[key].items():
            print(subkey, value)
            if subkey == "age":
                assert np.array_equal(value, SFTtoAge(mock_data))
            else:
                assert np.array_equal(value, mock_data)

    assert "center" in handler_galaxy_data
    assert (handler_galaxy_data["center"] == np.array([1.0, 2.0, 3.0])).all()

    print("Handler_units:", handler.get_units())
    assert handler.get_units() == {
        "stars": {
            "coords": "cm",
            "mass": "g",
            "metallicity": "",
            "velocity": "cm/s",
            "age": "Gyr",
        },
        "gas": {
            "coords": "cm",
            "density": "g/cm^3",
            "mass": "g",
            "metallicity": "",
            "metals": "",
            "sfr": "Msun/yr",
            "internal_energy": "erg/g",
            "velocity": "cm/s",
            "electron_abundance": "",
        },
        "galaxy": {
            "center": "cm",
            "halfmassrad_stars": "cm",
            "redshift": "",
        },
    }


@patch("os.path.exists")
@patch("h5py.File")
def test_wrong_fields(mock_file, mock_exists):
    mock_exists.return_value = True
    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)
    mock_file.return_value.__enter__.return_value = data
    data.pop("Header")
    with pytest.raises(ValueError) as e:
        handler = IllustrisHandler("path")
    assert "Header" in str(e.value)


@patch("os.path.exists")
@patch("h5py.File")
def test_attrs_missing_in_header(mock_file, mock_exists):
    mock_exists.return_value = True
    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)
    mock_file.return_value.__enter__.return_value = data
    data["Header"].attrs.pop("Time")
    with pytest.raises(ValueError) as e:
        handler = IllustrisHandler("path")
    assert "Time" in str(e.value)


@patch("os.path.exists")
@patch("h5py.File")
def test_hubble_attrs_missing_in_header(mock_file, mock_exists):
    mock_exists.return_value = True
    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)
    mock_file.return_value.__enter__.return_value = data

    data["Header"].attrs.pop("HubbleParam")
    with pytest.raises(ValueError) as f:
        handler = IllustrisHandler("path")
    assert "HubbleParam" in str(f.value)


@patch("os.path.exists")
@patch("h5py.File")
def test_unsupported_particle_type(mock_file, mock_exists):
    mock_exists.return_value = True
    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)

    data["PartTypeErr"] = {}
    mock_file.return_value.__enter__.return_value = data
    print(data)
    with pytest.raises(NotImplementedError) as f:
        handler = IllustrisHandler("path")
        print(handler.PARTICLE_KEYS)
    assert "PartTypeErr" in str(f.value)


@patch("os.path.exists")
@patch("h5py.File")
def test_unsupported_particle_field(mock_file, mock_exists):
    mock_exists.return_value = True
    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)

    data["PartType4"]["Unsupported"] = {}
    mock_file.return_value.__enter__.return_value = data
    print(data)
    with pytest.raises(NotImplementedError) as f:
        handler = IllustrisHandler("path")
    assert "Unsupported" in str(f.value)


@patch("os.path.exists")
@patch("h5py.File")
def test_missing_field(mock_file, mock_exists):
    mock_exists.return_value = True
    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)

    data["PartType4"].pop("Coordinates")
    mock_file.return_value.__enter__.return_value = data
    print(data)
    with pytest.raises(ValueError) as f:
        handler = IllustrisHandler("path")  #
    assert "Unsupported", "Missing" in str(f.value)


@patch("os.path.exists")
@patch("h5py.File")
def test_set_logger(mock_file, mock_exists):
    mock_exists.return_value = True
    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)
    import logging

    logger = logging.getLogger("test")
    mock_file.return_value.__enter__.return_value = data
    handler = IllustrisHandler("path", logger=logger)  #
    assert handler._logger == logger


@patch("os.path.exists")
@patch("h5py.File")
def test_load_data_without_GFM_stellarformation_time(mock_file, mock_exists):
    # Mock the os.path.exists method to always return True
    mock_exists.return_value = True

    mock_data = np.array([0.5, 0.5])
    data = create_mock_hdf5(mock_data)
    mock_file.return_value.__enter__.return_value = data

    # Update the config
    config = {
        "MAPPED_FIELDS": {
            "PartType4": {
                "Coordinates": "coordinates",
                "GFM_InitialMass": "initial_mass",
                "GFM_Metallicity": "metallicity",
                "Velocities": "velocities",
            },
            "test_part": {
                "Coordinates": "coordinates",
                "GFM_InitialMass": "initial_mass",
                "GFM_Metallicity": "metallicity",
                "Velocities": "velocities",
            },
        },
        "MAPPED_PARTICLE_KEYS": {"PartType4": "stars", "test_part": "test_particles"},
        "SIMULATION_META_KEYS": {},
        "GALAXY_SUBHALO_KEYS": {},
        "UNITS": {},
        "ILLUSTRIS_DATA": ["PartType4", "SubhaloData"],
    }

    # Test _load_data method
    handler = IllustrisHandler("path")
    handler.MAPPED_FIELDS = config["MAPPED_FIELDS"]
    handler.MAPPED_PARTICLE_KEYS = config["MAPPED_PARTICLE_KEYS"]

    data["PartType4"].pop("GFM_StellarFormationTime")
    data["test_part"] = data["PartType4"]
    data = handler._get_particle_data(data, "test_part")
    assert "coordinates" in data


def _make_stub_handler():
    handler = object.__new__(IllustrisHandler)
    handler._logger = MagicMock()
    return handler


def test_check_fields_missing_expected():
    handler = _make_stub_handler()
    with pytest.raises(ValueError) as exc:
        handler._check_fields({"random": {}})
    assert "No expected fields" in str(exc.value)


def test_check_fields_unexpected_extra_field():
    handler = _make_stub_handler()
    fake_data = {
        "Header": {},
        "SubhaloData": {},
        "PartType4": {},
        "Random": {},
    }
    with pytest.raises(ValueError) as exc:
        handler._check_fields(fake_data)
    assert "Unexpected fields" in str(exc.value)


def test_check_fields_unsupported_part_type():
    handler = _make_stub_handler()
    fake_data = {
        "Header": {},
        "SubhaloData": {},
        "PartType4": {},
        "PartType99": {},
    }
    with pytest.raises(NotImplementedError) as exc:
        handler._check_fields(fake_data)
    assert "PartType99" in str(exc.value)


def test_check_particle_data_requires_mapped_fields():
    handler = _make_stub_handler()
    valid_data = {"stars": {"coords": np.array([0.0])}}
    with pytest.raises(ValueError):
        handler._check_particle_data(valid_data, {})


def test_get_particle_keys_unsupported_type():
    handler = _make_stub_handler()
    handler.MAPPED_PARTICLE_KEYS = {"PartType4": "stars"}
    handler.ILLUSTRIS_DATA = [
        "Header",
        "SubhaloData",
        "PartType4",
        "PartTypeX",
    ]
    fake_file = {
        "Header": {},
        "SubhaloData": {},
        "PartType4": {},
        "PartTypeX": {},
    }
    with pytest.raises(NotImplementedError) as exc:
        handler._get_particle_keys(fake_file)
    assert "PartTypeX" in str(exc.value)


def test_check_particle_data_no_matching_fields():
    handler = _make_stub_handler()
    with pytest.raises(ValueError) as exc:
        handler._check_particle_data({"unexpected": {}}, {})
    assert "No expected fields" in str(exc.value)


def test_check_particle_data_extra_parttype_field_raises_not_implemented():
    handler = _make_stub_handler()
    handler.MAPPED_PARTICLE_KEYS = {"PartType4": "stars"}
    handler.MAPPED_FIELDS = {"PartType4": {"Coordinates": "coords"}}
    particle_data = {
        "stars": {"coords": np.array([0.0])},
        "PartType99": {},
    }
    with pytest.raises(NotImplementedError) as exc:
        handler._check_particle_data(particle_data, {})
    assert "PartType99" in str(exc.value)


def test_check_particle_data_extra_field_raises_value_error():
    handler = _make_stub_handler()
    handler.MAPPED_PARTICLE_KEYS = {"PartType4": "stars"}
    handler.MAPPED_FIELDS = {"PartType4": {"Coordinates": "coords"}}
    particle_data = {
        "stars": {"coords": np.array([0.0])},
        "extra": {},
    }
    with pytest.raises(ValueError) as exc:
        handler._check_particle_data(particle_data, {})
    assert "Unexpected fields" in str(exc.value)


def test_halfmassrad_stars_requires_coordinates():
    handler = _make_stub_handler()
    handler.TIME = 1.0
    handler.HUBBLE_PARAM = 0.5
    fake_file = {
        "SubhaloData": {"halfmassrad_stars": np.array(1.0)},
        "PartType4": {},
    }
    with pytest.raises(ValueError) as exc:
        handler._get_halfmassrad_stars(fake_file)
    assert "Coordinates" in str(exc.value)
