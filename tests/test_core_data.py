import tempfile
from unittest.mock import MagicMock, Mock, call, patch

import jax
import jax.numpy as jnp
import pytest

from rubix.core.data import (
    Galaxy,
    GasData,
    RubixData,
    StarsData,
    convert_to_rubix,
    get_reshape_data,
    get_rubix_data,
    prepare_input,
    reshape_array,
)

# Mock configuration for tests
config_dict = {
    "data": {
        "name": "IllustrisAPI",
        "args": {"api_key": "your_api_key", "particle_type": ["stars", "gas"]},
        "load_galaxy_args": {"snapshot": "latest"},
    },
    "galaxy": {"dist_z": 0.1, "rotation": {"type": "face-on"}},
    "output_path": "/path/to/output",
    "logger": {"log_level": "DEBUG", "log_file_path": None},
}

config_path = "path/to/config.yaml"


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
            "gas": {
                "coords": [[7, 8, 9]],
                "velocity": [[10, 11, 12]],
                "metallicity": [0.2],
                "mass": [2000],
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
        "gas": {
            "coords": "kpc",
            "velocity": "km/s",
            "metallicity": "",
            "mass": "Msun",
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
        mock_center_particles.return_value.gas.coords = [[7, 8, 9]]
        mock_center_particles.return_value.gas.velocity = [[10, 11, 12]]
        mock_center_particles.return_value.gas.metallicity = [0.2]
        mock_center_particles.return_value.gas.mass = [2000]
        mock_center_particles.return_value.galaxy.halfmassrad_stars = 1

        rubixdata = prepare_input(config_dict)

    assert jnp.array_equal(rubixdata.stars.coords, jnp.array([[1, 2, 3]]))
    assert jnp.array_equal(rubixdata.stars.velocity, jnp.array([[4, 5, 6]]))
    assert jnp.array_equal(rubixdata.stars.metallicity, jnp.array([0.1]))
    assert jnp.array_equal(rubixdata.stars.mass, jnp.array([1000]))
    assert jnp.array_equal(rubixdata.stars.age, jnp.array([4.5]))
    assert jnp.array_equal(rubixdata.gas.coords, jnp.array([[7, 8, 9]]))
    assert jnp.array_equal(rubixdata.gas.velocity, jnp.array([[10, 11, 12]]))
    assert jnp.array_equal(rubixdata.gas.metallicity, jnp.array([0.2]))
    assert jnp.array_equal(rubixdata.gas.mass, jnp.array([2000]))
    assert rubixdata.galaxy.halfmassrad_stars == 1

    print(mock_path_join.call_args_list)  # Print all calls to os.path.join
    # Check if the specific call is in the list of calls
    assert (
        call(config_dict["output_path"], "rubix_galaxy.h5")
        in mock_path_join.call_args_list
    )


def test_convert_to_rubix_raises_on_unknown_data_source():
    mock_logger = Mock()

    # Mock configuration for the test
    minimal_config = {
        "output_path": "/fake/path",
        "data": {"name": "UnknownAPI", "args": {}, "load_galaxy_args": {}},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        minimal_config["output_path"] = tmpdir
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ValueError, match="Unknown data source: UnknownAPI."):
                result = convert_to_rubix(minimal_config)


@patch("rubix.core.data.os.path.join")
@patch("rubix.core.data.center_particles")
@patch("rubix.core.data.get_logger")
@patch("rubix.core.data.load_galaxy_data")
def test_prepare_input_subset_case(
    mock_load_galaxy_data, mock_get_logger, mock_center_particles, mock_path_join
):
    # Mock output path
    mock_path_join.return_value = "/path/to/output/rubix_galaxy.h5"

    # Mock particle and galaxy data
    particle_data = {
        "particle_data": {
            "stars": {
                "coords": jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "velocity": jnp.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                "metallicity": jnp.array([0.1, 0.2, 0.3]),
                "mass": jnp.array([1000, 2000, 3000]),
                "age": jnp.array([4.5, 5.5, 6.5]),
            },
            "gas": {
                "coords": jnp.array([[7, 8, 9]]),
                "velocity": jnp.array([[10, 11, 12]]),
                "metallicity": jnp.array([0.2]),
                "mass": jnp.array([2000]),
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
        "gas": {
            "coords": "kpc",
            "velocity": "km/s",
            "metallicity": "",
            "mass": "Msun",
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
    mock_center_particles.return_value.gas.coords = jnp.array([[7, 8, 9]])
    mock_center_particles.return_value.gas.velocity = jnp.array([[10, 11, 12]])
    mock_center_particles.return_value.gas.metallicity = jnp.array([0.2])
    mock_center_particles.return_value.gas.mass = jnp.array([2000])
    mock_center_particles.return_value.galaxy.halfmassrad_stars = 1

    # Config with subset enabled
    config_dict = {
        "data": {
            "name": "IllustrisAPI",
            "args": {"api_key": "your_api_key", "particle_type": ["stars", "gas"]},
            "load_galaxy_args": {"snapshot": "latest"},
            "subset": {"use_subset": True, "subset_size": 2},
        },
        "galaxy": {"dist_z": 0.1, "rotation": {"type": "face-on"}},
        "output_path": "/path/to/output",
        "logger": {"log_level": "DEBUG", "log_file_path": None},
    }

    # Call prepare_input
    rubixdata = prepare_input(config_dict)

    # Check that only 2 entries are present due to subset
    assert rubixdata.stars.coords.shape[0] == 2
    assert rubixdata.stars.velocity.shape[0] == 2
    assert rubixdata.stars.metallicity.shape[0] == 2
    assert rubixdata.stars.mass.shape[0] == 2
    assert rubixdata.stars.age.shape[0] == 2
    assert rubixdata.gas.coords.shape[0] == 2
    assert rubixdata.gas.velocity.shape[0] == 2
    assert rubixdata.gas.metallicity.shape[0] == 2
    assert rubixdata.gas.mass.shape[0] == 2
    assert rubixdata.galaxy.halfmassrad_stars == 1

    # Verify path join was called with correct arguments
    assert (
        call(config_dict["output_path"], "rubix_galaxy.h5")
        in mock_path_join.call_args_list
    )


@patch("rubix.core.data.os.path.join")
@patch("rubix.core.data.center_particles")
@patch("rubix.core.data.load_galaxy_data")
def test_prepare_input_subset_gas_only(
    mock_load_galaxy_data, mock_center_particles, mock_path_join
):
    """Ensure the branch where stars are missing and gas is present is exercised.

    This hits the `elif rubixdata.gas.coords is not None:` branch in
    `prepare_input` that selects indices from gas.coords when stars.coords are
    not available.
    """
    # Mock output path
    mock_path_join.return_value = "/path/to/output/rubix_galaxy.h5"

    # Mock particle and galaxy data (only gas present)
    particle_data = {
        "particle_data": {
            "gas": {
                "coords": [[7, 8, 9], [10, 11, 12], [13, 14, 15]],
                "velocity": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "metallicity": [0.1, 0.2, 0.3],
                "mass": [100, 200, 300],
            }
        },
        "subhalo_center": [0, 0, 0],
        "subhalo_halfmassrad_stars": 1,
        "redshift": 0.1,
    }
    units = {
        "galaxy": {"center": "kpc", "halfmassrad_stars": "kpc", "redshift": ""},
        "gas": {"coords": "kpc", "velocity": "km/s", "metallicity": "", "mass": "Msun"},
    }

    mock_load_galaxy_data.return_value = (particle_data, units)

    # center_particles should return a RubixData where stars.coords stays None
    mock_center_particles.return_value = RubixData(Galaxy(), StarsData(), GasData())
    mock_center_particles.return_value.gas.coords = jnp.array(
        [[7, 8, 9], [10, 11, 12], [13, 14, 15]]
    )
    mock_center_particles.return_value.gas.velocity = jnp.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    mock_center_particles.return_value.gas.metallicity = jnp.array([0.1, 0.2, 0.3])
    mock_center_particles.return_value.gas.mass = jnp.array([100, 200, 300])
    mock_center_particles.return_value.galaxy.halfmassrad_stars = 1

    # Config with subset enabled
    cfg = {
        "data": {
            "name": "IllustrisAPI",
            "args": {"api_key": "your_api_key", "particle_type": ["stars", "gas"]},
            "load_galaxy_args": {"snapshot": "latest"},
            "subset": {"use_subset": True, "subset_size": 2},
        },
        "galaxy": {"dist_z": 0.1, "rotation": {"type": "face-on"}},
        "output_path": "/path/to/output",
        "logger": {"log_level": "DEBUG", "log_file_path": None},
    }

    # Call prepare_input
    rubixdata = prepare_input(cfg)

    # stars were never provided so should remain None
    assert rubixdata.stars.coords is None

    # gas should have been subset to the requested size
    assert rubixdata.gas.coords.shape[0] == 2
    assert rubixdata.gas.velocity.shape[0] == 2
    assert rubixdata.gas.metallicity.shape[0] == 2
    assert rubixdata.gas.mass.shape[0] == 2

    # Verify path join was called with correct arguments
    assert call(cfg["output_path"], "rubix_galaxy.h5") in mock_path_join.call_args_list


@patch("rubix.core.data.load_galaxy_data")
@patch("rubix.core.data.center_particles")
def test_prepare_input_subset_no_coords_raises(
    mock_center_particles, mock_load_galaxy_data
):
    """Raise when neither stars nor gas coords are available.

    Exercises the ValueError in the subset-selection branch where both
    `rubixdata.stars.coords` and `rubixdata.gas.coords` are None.
    """
    particle_data = {
        "particle_data": {"stars": {"coords": [[1, 2, 3]]}},
        "subhalo_center": [0, 0, 0],
        "subhalo_halfmassrad_stars": 1,
        "redshift": 0.1,
    }
    units = {
        "galaxy": {"center": "", "halfmassrad_stars": "", "redshift": ""},
        "stars": {"coords": ""},
    }

    mock_load_galaxy_data.return_value = (particle_data, units)

    # center_particles returns RubixData with no coords for either part
    mock_center_particles.return_value = RubixData(Galaxy(), StarsData(), GasData())

    cfg = {
        "data": {
            "name": "IllustrisAPI",
            "args": {"api_key": "x", "particle_type": ["stars"]},
            "load_galaxy_args": {},
            "subset": {"use_subset": True, "subset_size": 1},
        },
        "output_path": ".",
    }

    with pytest.raises(
        ValueError, match="Neither stars nor gas coordinates are available."
    ):
        prepare_input(cfg)


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

    # Mock the prepare_input function to return a RubixData instance
    mock_rubix_data = RubixData(
        galaxy=Galaxy(),
        stars=StarsData(),
        gas=GasData(),
    )
    mock_prepare_input.return_value = mock_rubix_data

    # Call the function
    result = get_rubix_data(config)

    # Assert that convert_to_rubix and prepare_input are called correctly
    mock_convert_to_rubix.assert_called_once_with(config)
    mock_prepare_input.assert_called_once_with(config)

    # Assert that the result is the mocked RubixData object
    assert result == mock_rubix_data


@patch("rubix.core.data.reshape_array")
def test_get_reshape_data(mock_reshape_array):
    # Configuration (if required)
    config = {}
    reshape_func = get_reshape_data(config)

    # Mock input data for the function
    input_data = RubixData(
        stars=StarsData(
            coords=jnp.array([[1, 2], [3, 4]]),
            velocity=jnp.array([[5, 6], [7, 8]]),
            metallicity=jnp.array([0.1, 0.2]),
            mass=jnp.array([1000, 2000]),
            age=jnp.array([4.5, 5.5]),
            pixel_assignment=jnp.array([0, 1]),
        ),
        gas=GasData(velocity=None),
    )

    # Expected reshaped data
    reshaped_data = RubixData(
        stars=StarsData(
            coords=jnp.array([[1, 2]]),
            velocity=jnp.array([[5, 6]]),
            metallicity=jnp.array([0.1]),
            mass=jnp.array([1000]),
            age=jnp.array([4.5]),
            pixel_assignment=jnp.array([0]),
        ),
        gas=GasData(velocity=None),
    )

    # Define the side effect for the mock to simulate reshaping
    def side_effect(x):
        # Match the input field with the reshaped data's equivalent field
        for field, value in input_data.stars.__dict__.items():
            if jnp.array_equal(value, x):
                return getattr(reshaped_data.stars, field)
        for field, value in input_data.gas.__dict__.items():
            if jnp.array_equal(value, x):
                return getattr(reshaped_data.gas, field)
        return None

    mock_reshape_array.side_effect = side_effect

    # Call the reshape function
    result = reshape_func(input_data)

    # Assertions to verify correctness
    assert jnp.array_equal(result.stars.coords, reshaped_data.stars.coords)
    assert jnp.array_equal(result.stars.velocity, reshaped_data.stars.velocity)
    assert jnp.array_equal(result.stars.metallicity, reshaped_data.stars.metallicity)
    assert jnp.array_equal(result.stars.mass, reshaped_data.stars.mass)
    assert jnp.array_equal(result.stars.age, reshaped_data.stars.age)
    assert jnp.array_equal(
        result.stars.pixel_assignment, reshaped_data.stars.pixel_assignment
    )
    assert result.gas.velocity == reshaped_data.gas.velocity


@pytest.fixture
def config():
    # Minimal config for logger
    return {
        "logger": {
            "log_level": "WARNING",
            "log_file_path": None,
            "format": "%(message)s",
        }
    }


def make_rubixdata(n_particles=8, n_features=3):
    # Create dummy data for stars and gas
    coords = jnp.arange(n_particles * n_features).reshape(n_particles, n_features)
    velocity = jnp.ones((n_particles, n_features))
    mass = jnp.arange(n_particles)
    metallicity = jnp.linspace(0, 1, n_particles)
    age = jnp.arange(n_particles)
    # Only fill a few attributes for brevity
    stars = StarsData(
        coords=coords,
        velocity=velocity,
        mass=mass,
        metallicity=metallicity,
        age=age,
    )
    gas = GasData(
        coords=coords + 100,
        velocity=velocity + 2,
        mass=mass + 10,
        metallicity=metallicity + 0.5,
    )
    galaxy = Galaxy(redshift=0.1, center=jnp.zeros(3), halfmassrad_stars=5.0)
    return RubixData(galaxy=galaxy, stars=stars, gas=gas)


def test_get_reshape_data_returns_callable(config):
    reshape_data = get_reshape_data(config)
    assert callable(reshape_data)


def test_reshape_data_shapes(monkeypatch, config):
    n_gpus = 3
    monkeypatch.setattr(jax, "device_count", lambda: n_gpus)
    n_particles = 7  # Not divisible by n_gpus to test padding
    n_features = 3
    rubixdata = make_rubixdata(n_particles=n_particles, n_features=n_features)
    reshape_data = get_reshape_data(config)
    reshaped = reshape_data(rubixdata)

    # Check stars.coords shape
    expected_particles_per_gpu = (n_particles + n_gpus - 1) // n_gpus
    assert reshaped.stars.coords.shape == (
        n_gpus,
        expected_particles_per_gpu,
        n_features,
    )
    assert reshaped.stars.velocity.shape == (
        n_gpus,
        expected_particles_per_gpu,
        n_features,
    )
    assert reshaped.stars.mass.shape == (n_gpus, expected_particles_per_gpu)
    assert reshaped.stars.metallicity.shape == (n_gpus, expected_particles_per_gpu)
    assert reshaped.stars.age.shape == (n_gpus, expected_particles_per_gpu)

    # Check gas.coords shape
    assert reshaped.gas.coords.shape == (n_gpus, expected_particles_per_gpu, n_features)
    assert reshaped.gas.velocity.shape == (
        n_gpus,
        expected_particles_per_gpu,
        n_features,
    )
    assert reshaped.gas.mass.shape == (n_gpus, expected_particles_per_gpu)
    assert reshaped.gas.metallicity.shape == (n_gpus, expected_particles_per_gpu)


def test_reshape_data_handles_none_attributes(config):
    # Only coords is set, others are None
    stars = StarsData(coords=jnp.ones((4, 3)))
    gas = GasData(coords=jnp.ones((4, 3)))
    rubixdata = RubixData(galaxy=Galaxy(), stars=stars, gas=gas)
    reshape_data = get_reshape_data(config)
    reshaped = reshape_data(rubixdata)
    assert reshaped.stars.coords.shape[0] == jax.device_count()
    assert reshaped.gas.coords.shape[0] == jax.device_count()


def test_reshape_data_skips_if_coords_none(config):
    # coords is None, should not raise
    stars = StarsData(coords=None)
    gas = GasData(coords=None)
    rubixdata = RubixData(galaxy=Galaxy(), stars=stars, gas=gas)
    reshape_data = get_reshape_data(config)
    reshaped = reshape_data(rubixdata)
    assert reshaped.stars.coords is None
    assert reshaped.gas.coords is None
