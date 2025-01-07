from unittest.mock import MagicMock, Mock, patch, call

import jax
import jax.numpy as jnp
from rubix.core.data import (
    convert_to_rubix,
    prepare_input,
    reshape_array,
    get_rubix_data,
    get_reshape_data,
)

# Mock configuration for tests
config_dict = {
    "data": {
        "name": "IllustrisAPI",
        "args": {"api_key": "your_api_key", "particle_type": ["stars"]},
        "load_galaxy_args": {"snapshot": "latest"},
    },
    "galaxy": {"dist_z": 0.1, "rotation": {"type": "face-on"}},
    "output_path": "/path/to/output",
    "logger": {"log_level": "DEBUG", "log_file_path": None},
}

config_path = "path/to/config.yaml"


class RubixData:
    def __init__(self, galaxy, stars, gas):
        self.galaxy = galaxy
        self.stars = stars
        self.gas = gas


class Galaxy:
    def __init__(self):
        self.redshift = None
        self.center = None
        self.halfmassrad_stars = None


class StarsData:
    def __init__(self):
        self.coords = None
        self.velocity = None
        self.metallicity = None
        self.mass = None
        self.age = None


class GasData:
    def __init__(self):
        self.coords = None
        self.velocity = None
        self.mass = None
        self.density = None
        self.internal_energy = None
        self.metallicity = None
        self.sfr = None
        self.electron_abundance = None


# Test convert_to_rubix function
@patch("rubix.core.data.read_yaml")
@patch("rubix.core.data.get_logger")
@patch("rubix.core.data.get_input_handler")
@patch("rubix.core.data.IllustrisAPI")
def test_convert_to_rubix(
    mock_illustris_api, mock_input_handler, mock_logger, mock_read_yaml
):
    mock_read_yaml.return_value = config_dict
    mock_logger.return_value = MagicMock()
    mock_input_handler.return_value = MagicMock()
    mock_input_handler.return_value.to_rubix.return_value = None
    mock_illustris_api.return_value = MagicMock()
    mock_illustris_api.return_value.load_galaxy.return_value = None

    output_path = convert_to_rubix(config_path)
    assert output_path == config_dict["output_path"]
    mock_read_yaml.assert_called_once_with(config_path)
    mock_logger.assert_called_once()
    mock_input_handler.assert_called_once()
    mock_illustris_api.assert_called_once()


def test_rubix_file_already_exists():
    # Mock configuration for the test
    config = {
        "output_path": "/fake/path",
        "data": {"name": "IllustrisAPI", "args": {}, "load_galaxy_args": {}},
    }

    # Create a mock logger that does nothing
    mock_logger = Mock()

    with patch("rubix.core.data.os.path.exists", return_value=True) as mock_exists:
        with patch(
            "rubix.core.data.get_logger", return_value=mock_logger
        ) as mock_get_logger:
            # Call the function under test
            result = convert_to_rubix(config)

            # Check that the file existence check was performed correctly
            mock_exists.assert_called_once_with("/fake/path/rubix_galaxy.h5")

            # Check that the logger was created
            mock_get_logger.assert_called_once_with(None)

            # Ensure the function logs the right message and skips conversion
            mock_logger.info.assert_called_with(
                "Rubix galaxy file already exists, skipping conversion"
            )

            # Verify that the function returns the expected path without performing further actions
            assert (
                result == "/fake/path"
            ), "Function should return the output path when file exists"


@patch("rubix.core.data.os.path.join")
@patch("rubix.core.data.center_particles")
def test_prepare_input(mock_center_particles, mock_path_join):
    mock_path_join.return_value = "/path/to/output/rubix_galaxy.h5"
    particle_data = {
        "particle_data": {
            "stars": {
                "coords": [[1, 2, 3]],
                "velocity": [[4, 5, 6]],
                "metallicity": [0.1],
                "mass": [1000],
                "age": [4.5],
            },
        },
        "subhalo_center": [0, 0, 0],
        "subhalo_halfmassrad_stars": 1,
        "redshift": 0.1,
    }
    units = {
        "galaxy": {"center": "kpc", "halfmassrad_stars": "kpc", "redshift": ""},
        "stars": {
            "coords": "kpc",
            "velocity": "km/s",
            "metallicity": "",
            "mass": "Msun",
            "age": "Gyr",
        },
    }
    mock_load_galaxy_data = (particle_data, units)
    with patch("rubix.core.data.load_galaxy_data", return_value=mock_load_galaxy_data):
        # mock_center_particles.return_value = ([[1, 2, 3]], [[4, 5, 6]])
        mock_center_particles.return_value = RubixData(Galaxy(), StarsData(), GasData())
        mock_center_particles.return_value.stars.coords = [[1, 2, 3]]
        mock_center_particles.return_value.stars.velocity = [[4, 5, 6]]
        mock_center_particles.return_value.stars.metallicity = [0.1]
        mock_center_particles.return_value.stars.mass = [1000]
        mock_center_particles.return_value.stars.age = [4.5]
        mock_center_particles.return_value.galaxy.halfmassrad_stars = 1

        rubixdata = prepare_input(config_dict)

        assert rubixdata.stars.coords == [[1, 2, 3]]
        assert rubixdata.stars.velocity == [[4, 5, 6]]
        assert rubixdata.stars.metallicity == [0.1]
        assert rubixdata.stars.mass == [1000]
        assert rubixdata.stars.age == [4.5]
        assert rubixdata.galaxy.halfmassrad_stars == 1

        print(mock_path_join.call_args_list)  # Print all calls to os.path.join
        # Check if the specific call is in the list of calls
        assert (
            call(config_dict["output_path"], "rubix_galaxy.h5")
            in mock_path_join.call_args_list
        )


@patch("rubix.core.data.os.path.join")
@patch("rubix.core.data.center_particles")
@patch("rubix.core.data.get_logger")
@patch("rubix.core.data.load_galaxy_data")
def test_prepare_input_subset_case(
    mock_load_galaxy_data, mock_get_logger, mock_center_particles, mock_path_join
):
    mock_path_join.return_value = "/path/to/output/rubix_galaxy.h5"
    particle_data = {
        "particle_data": {
            "stars": {
                "coords": jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "velocity": jnp.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                "metallicity": jnp.array([0.1, 0.2, 0.3]),
                "mass": jnp.array([1000, 2000, 3000]),
                "age": jnp.array([4.5, 5.5, 6.5]),
            },
        },
        "subhalo_center": jnp.array([0, 0, 0]),
        "subhalo_halfmassrad_stars": 1,
        "redshift": 0.1,
    }
    units = {
        "galaxy": {"center": "kpc", "halfmassrad_stars": "kpc", "redshift": ""},
        "stars": {
            "coords": "kpc",
            "velocity": "km/s",
            "metallicity": "",
            "mass": "Msun",
            "age": "Gyr",
        },
    }

    mock_load_galaxy_data.return_value = (particle_data, units)
    mock_center_particles.return_value = RubixData(Galaxy(), StarsData(), GasData())
    mock_center_particles.return_value.stars.coords = jnp.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    mock_center_particles.return_value.stars.velocity = jnp.array(
        [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
    )
    mock_center_particles.return_value.stars.metallicity = jnp.array([0.1, 0.2, 0.3])
    mock_center_particles.return_value.stars.mass = jnp.array([1000, 2000, 3000])
    mock_center_particles.return_value.stars.age = jnp.array([4.5, 5.5, 6.5])
    mock_center_particles.return_value.galaxy.halfmassrad_stars = 1

    config_dict = {
        "data": {
            "name": "IllustrisAPI",
            "args": {"api_key": "your_api_key", "particle_type": ["stars"]},
            "load_galaxy_args": {"snapshot": "latest"},
            "subset": {"use_subset": True, "subset_size": 2},
        },
        "galaxy": {"dist_z": 0.1, "rotation": {"type": "face-on"}},
        "output_path": "/path/to/output",
        "logger": {"log_level": "DEBUG", "log_file_path": None},
    }

    rubixdata = prepare_input(config_dict)

    assert rubixdata.stars.coords.shape[0] == 2
    assert rubixdata.stars.velocity.shape[0] == 2
    assert rubixdata.stars.metallicity.shape[0] == 2
    assert rubixdata.stars.mass.shape[0] == 2
    assert rubixdata.stars.age.shape[0] == 2
    assert rubixdata.galaxy.halfmassrad_stars == 1

    assert (
        call(config_dict["output_path"], "rubix_galaxy.h5")
        in mock_path_join.call_args_list
    )


def test_reshape_array_single_gpu(monkeypatch):
    monkeypatch.setattr(jax, "device_count", lambda: 1)
    arr = jnp.array([[1, 2], [3, 4]])
    result = reshape_array(arr)
    expected = arr.reshape(1, 2, 2)
    assert jnp.array_equal(result, expected)


def test_reshape_array_multiple_gpus(monkeypatch):
    monkeypatch.setattr(jax, "device_count", lambda: 2)
    arr = jnp.array([[1, 2], [3, 4], [5, 6]])
    result = reshape_array(arr)
    expected = jnp.array([[[1, 2], [3, 4]], [[5, 6], [0, 0]]])
    assert jnp.array_equal(result, expected)


def test_reshape_array_padding(monkeypatch):
    monkeypatch.setattr(jax, "device_count", lambda: 3)
    arr = jnp.array([[1, 2], [3, 4]])
    result = reshape_array(arr)
    expected = jnp.array([[[1, 2]], [[3, 4]], [[0, 0]]])
    assert jnp.array_equal(result, expected)


@patch("rubix.core.data.convert_to_rubix")
@patch("rubix.core.data.prepare_input")
def test_get_rubix_data(mock_prepare_input, mock_convert_to_rubix):
    config = {"output_path": "/path/to/output"}
    mock_prepare_input.return_value = "prepared_data"

    result = get_rubix_data(config)

    mock_convert_to_rubix.assert_called_once_with(config)
    mock_prepare_input.assert_called_once_with(config)
    assert result == "prepared_data"


"""
@patch("rubix.core.data.reshape_array")
def test_get_reshape_data(mock_reshape_array):
    config = {}
    reshape_func = get_reshape_data(config)

    input_data = {
        "coords": jnp.array([[1, 2], [3, 4]]),
        "velocities": jnp.array([[5, 6], [7, 8]]),
        "metallicity": jnp.array([0.1, 0.2]),
        "mass": jnp.array([1000, 2000]),
        "age": jnp.array([4.5, 5.5]),
        "pixel_assignment": jnp.array([0, 1]),
    }

    reshaped_data = {
        "coords": jnp.array([[1, 2]]),
        "velocities": jnp.array([[5, 6]]),
        "metallicity": jnp.array([0.1]),
        "mass": jnp.array([1000]),
        "age": jnp.array([4.5]),
        "pixel_assignment": jnp.array([0]),
    }

    mock_reshape_array.side_effect = lambda x: reshaped_data[
        [k for k, v in input_data.items() if jnp.array_equal(v, x)][0]
    ]

    result = reshape_func(input_data)

    for key in input_data:
        assert jnp.array_equal(result[key], reshaped_data[key])

    # Print calls to mock_reshape_array
    print("Calls to mock_reshape_array:")
    for calls in mock_reshape_array.call_args_list:
        print(calls)
"""


class MockRubixData:
    def __init__(self, stars, gas):
        self.stars = stars
        self.gas = gas


class MockStarsData:
    def __init__(self, coords, velocity, metallicity, mass, age, pixel_assignment):
        self.coords = coords
        self.velocity = velocity
        self.metallicity = metallicity
        self.mass = mass
        self.age = age
        self.pixel_assignment = pixel_assignment


class MockGasData:
    def __init__(self, coords=None):
        self.coords = coords


@patch("rubix.core.data.reshape_array")
def test_get_reshape_data(mock_reshape_array):
    config = {}
    reshape_func = get_reshape_data(config)

    input_data = MockRubixData(
        MockStarsData(
            coords=jnp.array([[1, 2], [3, 4]]),
            velocity=jnp.array([[5, 6], [7, 8]]),
            metallicity=jnp.array([0.1, 0.2]),
            mass=jnp.array([1000, 2000]),
            age=jnp.array([4.5, 5.5]),
            pixel_assignment=jnp.array([0, 1]),
        ),
        MockGasData(coords=None),
    )

    reshaped_data = MockRubixData(
        MockStarsData(
            coords=jnp.array([[1, 2]]),
            velocity=jnp.array([[5, 6]]),
            metallicity=jnp.array([0.1]),
            mass=jnp.array([1000]),
            age=jnp.array([4.5]),
            pixel_assignment=jnp.array([0]),
        ),
        MockGasData(coords=None),
    )

    def side_effect(x):
        for k, v in input_data.stars.__dict__.items():
            if jnp.array_equal(v, x):
                return reshaped_data.stars.__dict__[k]
        for k, v in input_data.gas.__dict__.items():
            if jnp.array_equal(v, x):
                return reshaped_data.gas.__dict__[k]
        return None

    mock_reshape_array.side_effect = side_effect

    result = reshape_func(input_data)

    assert jnp.array_equal(result.stars.coords, reshaped_data.stars.coords)
    assert jnp.array_equal(result.stars.velocity, reshaped_data.stars.velocity)
    assert jnp.array_equal(result.stars.metallicity, reshaped_data.stars.metallicity)
    assert jnp.array_equal(result.stars.mass, reshaped_data.stars.mass)
    assert jnp.array_equal(result.stars.age, reshaped_data.stars.age)
    assert jnp.array_equal(
        result.stars.pixel_assignment, reshaped_data.stars.pixel_assignment
    )


# # Test prepare_input function
# @patch("rubix.core.data.os.path.join")
# @patch("rubix.core.data.center_particles")
# def test_prepare_input(mock_center_particles, mock_path_join):
#     mock_path_join.return_value = "/path/to/output/rubix_galaxy.h5"
#     particle_data = {
#         "particle_data": {
#             "stars": {
#                 "coords": [[1, 2, 3]],
#                 "velocity": [[4, 5, 6]],
#                 "metallicity": [0.1],
#                 "mass": [1000],
#                 "age": [4.5],
#             },
#         },
#         "subhalo_center": [0, 0, 0],
#     }
#     units = {
#         "galaxy": {"center": "kpc", "halfmassrad_stars": "kpc", "redshift": ""},
#         "stars": {"mass": "Msun"},
#     }
#     mock_load_galaxy_data = (particle_data, units)
#     with patch("rubix.core.data.load_galaxy_data", return_value=mock_load_galaxy_data):
#         mock_center_particles.return_value = ([[1, 2, 3]], [[4, 5, 6]])
#
#         coords, velocities, metallicity, mass, age = prepare_input(config_dict)
#
#         assert coords == [[1, 2, 3]]
#         assert velocities == [[4, 5, 6]]
#         assert metallicity == [0.1]
#         assert mass == [1000]
#         assert age == [4.5]
#
#         mock_path_join.assert_called_once_with(
#             config_dict["output_path"], "rubix_galaxy.h5"
#         )
#         mock_center_particles.assert_called_once()
