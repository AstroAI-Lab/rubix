from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.core.rotation import get_galaxy_rotation


def _build_rubix_data():
    galaxy = Galaxy(center=np.zeros(3), halfmassrad_stars=1.0)
    stars = StarsData(
        coords=np.zeros((1, 3)), velocity=np.zeros((1, 3)), mass=np.ones(1)
    )
    gas = GasData(coords=np.ones((1, 3)), velocity=np.ones((1, 3)))
    return RubixData(galaxy=galaxy, stars=stars, gas=gas)


def _base_config(particle_types):
    return {
        "galaxy": {"rotation": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}},
        "simulation": {"name": "mock"},
        "data": {"args": {"particle_type": particle_types}},
    }


def _get_data():
    return {
        "coords": None,
        "velocities": None,
        "mass": None,
        "halfmassrad_stars": None,
    }


def test_rotation_info_not_provided():
    config = {"galaxy": {}}
    with pytest.raises(
        ValueError, match="Rotation information not provided in galaxy config"
    ):
        get_galaxy_rotation(config)


def test_alpha_not_provided():
    config = {"galaxy": {"rotation": {"beta": 0, "gamma": 0}}}
    with pytest.raises(ValueError, match="alpha not provided in rotation information"):
        get_galaxy_rotation(config)


def test_beta_not_provided():
    config = {"galaxy": {"rotation": {"alpha": 0, "gamma": 0}}}
    with pytest.raises(ValueError, match="beta not provided in rotation information"):
        get_galaxy_rotation(config)


def test_invalid_rotation_type():
    config = {"galaxy": {"rotation": {"type": "invalid"}}}
    with pytest.raises(
        ValueError, match="Invalid type provided in rotation information"
    ):
        get_galaxy_rotation(config)


def test_gamma_not_provided():
    config = {"galaxy": {"rotation": {"alpha": 0, "beta": 0}}}
    with pytest.raises(ValueError, match="gamma not provided in rotation information"):
        get_galaxy_rotation(config)


def test_face_on_rotation():
    config = {"galaxy": {"rotation": {"type": "face-on"}}}
    rotate_galaxy = get_galaxy_rotation(config)
    assert callable(rotate_galaxy)


def test_edge_on_rotation():
    config = {"galaxy": {"rotation": {"type": "edge-on"}}}
    rotate_galaxy = get_galaxy_rotation(config)
    assert callable(rotate_galaxy)


def test_custom_rotation():
    config = {"galaxy": {"rotation": {"alpha": 45, "beta": 30, "gamma": 15}}}
    rotate_galaxy = get_galaxy_rotation(config)
    assert callable(rotate_galaxy)


@patch("rubix.core.rotation.rotate_galaxy_core")
@patch("rubix.core.rotation.get_logger")
def test_rotation_applies_to_gas_and_stars(mock_get_logger, mock_rotate_core):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_rotate_core.side_effect = lambda **kwargs: (
        kwargs["positions"] + 1,
        kwargs["velocities"] + 2,
    )

    rubixdata = _build_rubix_data()
    config = _base_config(["stars", "gas"])
    rotate = get_galaxy_rotation(config)

    rotated = rotate(rubixdata)

    assert np.all(rotated.gas.coords == np.ones((1, 3)) + 1)
    assert np.all(rotated.gas.velocity == np.ones((1, 3)) + 2)
    assert np.all(rotated.stars.coords == np.zeros((1, 3)) + 1)
    assert np.all(rotated.stars.velocity == np.zeros((1, 3)) + 2)
    assert mock_rotate_core.call_count == 2


@patch("rubix.core.rotation.rotate_galaxy_core")
@patch("rubix.core.rotation.get_logger")
def test_rotation_warns_when_gas_missing(mock_get_logger, mock_rotate_core):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_rotate_core.side_effect = lambda **kwargs: (
        kwargs["positions"] + 5,
        kwargs["velocities"] + 5,
    )

    rubixdata = _build_rubix_data()
    config = _base_config(["stars"])
    rotate = get_galaxy_rotation(config)
    rotate(rubixdata)

    mock_logger.warning.assert_called_with(
        "Gas not found in particle_type, only rotating stellar component."
    )
    assert mock_rotate_core.call_count == 1
