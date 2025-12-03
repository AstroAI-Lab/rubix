from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.core.telescope import (
    get_filter_particles,
    get_spatial_bin_edges,
    get_spaxel_assignment,
    get_telescope,
)
from rubix.telescope.base import BaseTelescope


class MockRubixData:
    def __init__(self, stars, gas):
        self.stars = stars
        self.gas = gas


class MockStarsData:
    def __init__(self, coords):
        self.coords = coords


class MockGasData:
    def __init__(self, coords):
        self.coords = coords


@patch("rubix.core.telescope.TelescopeFactory")
def test_get_telescope(mock_factory):
    config = {"telescope": {"name": "MUSE"}}
    # Create a mock with spec of BaseTelescope
    mock_telescope = MagicMock(spec=BaseTelescope)

    # Set the return value of the mock factory method
    mock_factory.return_value.create_telescope.return_value = mock_telescope

    # Call the function under test
    result = get_telescope(config)

    # Assertions
    mock_factory.return_value.create_telescope.assert_called_once_with("MUSE")
    assert result == mock_telescope


def test_get_spaxel_assignment():
    config = {
        "telescope": {
            "name": "MUSE",
        },
        "galaxy": {"dist_z": 0.5},
        "cosmology": {"name": "PLANCK15"},
    }

    with (
        patch("rubix.core.telescope.get_telescope") as mock_get_telescope,
        patch(
            "rubix.core.telescope.get_spatial_bin_edges"
        ) as mock_get_spatial_bin_edges,
        patch(
            "rubix.core.telescope.square_spaxel_assignment"
        ) as mock_square_spaxel_assignment,
    ):

        mock_get_telescope.return_value = MagicMock(pixel_type="square")
        mock_get_spatial_bin_edges.return_value = "spatial_bin_edges"
        mock_square_spaxel_assignment.return_value = "pixel_assignment"

        spaxel_assignment = get_spaxel_assignment(config)

        assert callable(spaxel_assignment)

        # input_data = {"coords": "coords"}
        input_data = MockRubixData(
            MockStarsData(
                coords="coords",
            ),
            MockGasData(
                coords=None,
            ),
        )
        result = spaxel_assignment(input_data)

        assert result.stars.pixel_assignment == "pixel_assignment"
        assert result.stars.spatial_bin_edges == "spatial_bin_edges"
        assert result.stars.coords == "coords"

    # Test for unsupported pixel type
    with patch("rubix.core.telescope.get_telescope") as mock_get_telescope:
        mock_get_telescope.return_value.pixel_type = "unsupported"

        with pytest.raises(ValueError):
            get_spaxel_assignment(config)


@patch("rubix.core.telescope.calculate_spatial_bin_edges")
@patch("rubix.core.telescope.get_cosmology")
@patch("rubix.core.telescope.get_telescope")
def test_get_spatial_bin_edges(
    mock_get_telescope, mock_get_cosmology, mock_calculate_spatial_bin_edges
):
    config = {
        "telescope": {"name": "MUSE"},
        "galaxy": {"dist_z": 0.5},
        "cosmology": {"name": "PLANCK15"},
    }

    mock_telescope = MagicMock(fov=1.0, sbin=10)
    mock_get_telescope.return_value = mock_telescope
    mock_get_cosmology.return_value = "cosmology"
    mock_calculate_spatial_bin_edges.return_value = (
        jnp.array([0.0, 1.0, 2.0]),  # Mocked spatial bin edges
        jnp.array([1.0, 1.0, 1.0]),  # spatial_bin_size
    )

    result = get_spatial_bin_edges(config)

    mock_get_telescope.assert_called_once_with(config)
    mock_get_cosmology.assert_called_once_with(config)
    mock_calculate_spatial_bin_edges.assert_called_once_with(
        fov=1.0,
        spatial_bins=10,
        dist_z=0.5,
        cosmology="cosmology",
    )
    # Assertions
    assert isinstance(result, jnp.ndarray)  # Ensure the return type matches
    assert result.shape == (3,)  # Check the shape of spatial_bin_edges


@patch("rubix.core.telescope.TelescopeFactory")
def test_get_telescope_type_error(mock_factory):
    config = {"telescope": {"name": "MUSE"}}
    mock_factory.return_value.create_telescope.return_value = MagicMock()

    with pytest.raises(TypeError, match="Expected type BaseTelescope"):
        get_telescope(config)


def test_spaxel_assignment_handles_stars_and_gas():
    config = {
        "telescope": {"name": "MUSE"},
        "galaxy": {"dist_z": 0.5},
        "cosmology": {"name": "PLANCK15"},
    }

    with (
        patch("rubix.core.telescope.get_telescope") as mock_get_telescope,
        patch("rubix.core.telescope.get_spatial_bin_edges") as mock_get_spatial,
        patch("rubix.core.telescope.square_spaxel_assignment") as mock_assignment,
    ):
        mock_get_telescope.return_value = MagicMock(pixel_type="square")
        mock_get_spatial.return_value = jnp.array([0.0, 1.0])
        mock_assignment.side_effect = ["star-pa", "gas-pa"]

        spaxel_assignment = get_spaxel_assignment(config)

        stars = StarsData(coords=jnp.zeros((1, 3)))
        gas = GasData(coords=jnp.ones((1, 3)))
        data = RubixData(galaxy=Galaxy(), stars=stars, gas=gas)

        result = spaxel_assignment(data)

        assert result.stars.pixel_assignment == "star-pa"
        assert result.stars.spatial_bin_edges is mock_get_spatial.return_value
        assert result.gas.pixel_assignment == "gas-pa"
        assert result.gas.spatial_bin_edges is mock_get_spatial.return_value
        assert mock_assignment.call_count == 2


@patch("rubix.core.telescope.mask_particles_outside_aperture")
@patch("rubix.core.telescope.get_spatial_bin_edges")
def test_filter_particles_masks_stars_and_gas(mock_get_edges, mock_mask_particles):
    config = {
        "telescope": {"name": "MUSE"},
        "galaxy": {"dist_z": 0.5},
        "cosmology": {"name": "PLANCK15"},
        "data": {"args": {"particle_type": ["stars", "gas"]}},
    }

    mock_get_edges.return_value = jnp.array([0.0, 1.0, 2.0])
    star_mask = jnp.array([True, False, True])
    gas_mask = jnp.array([False, True, True])
    mock_mask_particles.side_effect = [star_mask, gas_mask]

    stars = StarsData(
        coords=jnp.zeros((3, 3)),
        velocity=jnp.zeros((3, 3)),
        mass=jnp.array([1.0, 2.0, 3.0]),
        age=jnp.array([1.0, 2.0, 3.0]),
        metallicity=jnp.array([1.0, 2.0, 3.0]),
    )
    gas = GasData(
        coords=jnp.zeros((3, 3)),
        velocity=jnp.zeros((3, 3)),
        mass=jnp.array([4.0, 5.0, 6.0]),
        density=jnp.array([4.0, 5.0, 6.0]),
        internal_energy=jnp.array([4.0, 5.0, 6.0]),
        metallicity=jnp.array([4.0, 5.0, 6.0]),
    )
    data = RubixData(galaxy=Galaxy(), stars=stars, gas=gas)

    filter_fn = get_filter_particles(config)

    result = filter_fn(data)

    assert jnp.array_equal(result.stars.mass, jnp.array([1.0, 0.0, 3.0]))
    assert jnp.array_equal(result.stars.age, jnp.array([1.0, 0.0, 3.0]))
    assert jnp.array_equal(result.stars.metallicity, jnp.array([1.0, 0.0, 3.0]))
    assert jnp.array_equal(result.stars.mask, star_mask)
    assert jnp.array_equal(result.gas.mass, jnp.array([0.0, 5.0, 6.0]))
    assert jnp.array_equal(result.gas.density, jnp.array([0.0, 5.0, 6.0]))
    assert jnp.array_equal(result.gas.internal_energy, jnp.array([0.0, 5.0, 6.0]))
    assert jnp.array_equal(result.gas.mask, gas_mask)
    assert mock_mask_particles.call_count == 2
