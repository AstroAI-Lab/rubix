from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from rubix.telescope.apertures import (
    CIRCULAR_APERTURE,
    HEXAGONAL_APERTURE,
    SQUARE_APERTURE,
)
from rubix.telescope.base import BaseTelescope
from rubix.telescope.factory import (
    TelescopeFactory,
)

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def sample_telescope_config():
    return {
        "telescope1": {
            "fov": 100.0,
            "spatial_res": 10,
            "wave_range": [400.0, 700.0],
            "wave_res": 0.5,
            "lsf_fwhm": 0.1,
            "signal_to_noise": 100.0,
            "aperture_type": "square",
            "pixel_type": "square",
        }
    }


@patch("rubix.utils.read_yaml", return_value={})
def test_telescope_factory_no_config_warning(mock_read_yaml):
    with pytest.warns(Warning, match="No telescope config provided, using default"):
        factory = TelescopeFactory()  # noqa


def test_telescope_factory_with_file_path(tmpdir):
    # Mock the read_yaml function to return a predefined configuration
    # Create a factory instance with a string argument simulating a path to a config file
    dummy_data = {
        "telescope1": {
            "fov": 100.0,
            "spatial_res": 10,
            "aperture_type": "square",
            "wave_range": [400.0, 700.0],
            "wave_res": 0.5,
            "lsf_fwhm": 0.1,
            "signal_to_noise": 100.0,
            "pixel_type": "square",
        }
    }

    # save the dummy data to a file
    config_file = tmpdir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(dummy_data, f)
    factory = TelescopeFactory(telescopes_config=str(config_file))

    # Create a telescope using the factory and check if it is correctly configured
    telescope = factory.create_telescope("telescope1")
    assert telescope.fov == 100.0
    assert telescope.spatial_res == 10
    assert telescope.wave_range == [400.0, 700.0]
    assert telescope.wave_res == 0.5
    assert telescope.lsf_fwhm == 0.1
    assert telescope.signal_to_noise == 100
    # Check if the aperture region is correctly initialized
    assert np.all(telescope.aperture_region == SQUARE_APERTURE(np.int64(100 / 10)))


def test_create_telescope_with_valid_config(sample_telescope_config):
    factory = TelescopeFactory(telescopes_config=sample_telescope_config)
    telescope = factory.create_telescope("telescope1")
    assert isinstance(telescope, BaseTelescope)


def test_create_telescope_with_invalid_name(sample_telescope_config):
    factory = TelescopeFactory(telescopes_config=sample_telescope_config)
    with pytest.raises(ValueError, match="Telescope telescope2 not found in config"):
        factory.create_telescope("telescope2")


def test_create_telescope_unknown_aperture_type(sample_telescope_config):
    # Modify config to have an unknown aperture type
    config = sample_telescope_config.copy()
    config["telescope1"]["aperture_type"] = "unknown"

    factory = TelescopeFactory(telescopes_config=config)
    with pytest.raises(ValueError, match="Unknown aperture type: unknown"):
        factory.create_telescope("telescope1")


@pytest.mark.parametrize(
    "aperture_type, aperture_function",
    [
        ("square", SQUARE_APERTURE),
        ("circular", CIRCULAR_APERTURE),
        ("hexagonal", HEXAGONAL_APERTURE),
    ],
)
def test_create_telescope_with_different_apertures(
    sample_telescope_config, aperture_type, aperture_function
):
    config = sample_telescope_config.copy()
    config["telescope1"]["aperture_type"] = aperture_type

    with patch(
        f"rubix.telescope.apertures.{aperture_type.upper()}_APERTURE",
        MagicMock(return_value=np.array("some_aperture_representation")),
    ) as mocked_aperture:  # noqa
        factory = TelescopeFactory(telescopes_config=config)
        telescope = factory.create_telescope("telescope1")
        sbin = np.int64(
            config["telescope1"]["fov"] / config["telescope1"]["spatial_res"]
        )
        expected_aperture = aperture_function(sbin)
        actual_aperture = getattr(telescope, "aperture_region")

        # Comparing using numpy's array_equal for simplicity
        assert np.array_equal(
            actual_aperture, expected_aperture
        ), "Aperture region mismatch"
