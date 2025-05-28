import pytest

from rubix.core.rotation import get_galaxy_rotation


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
