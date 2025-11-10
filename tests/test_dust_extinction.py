from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from rubix.core.data import RubixData
from rubix.core.dust import get_extinction
from rubix.spectra.dust.dust_extinction import (
    apply_spaxel_extinction,
    calculate_dust_to_gas_ratio,
)
from rubix.spectra.dust.helpers import poly_map_domain


@pytest.fixture
def mock_config():
    return {
        "logger": None,
        "constants": {
            "MASS_OF_PROTON": 1.6726219e-24,
            "MSUN_TO_GRAMS": 1.989e33,
            "KPC_TO_CM": 3.086e21,
        },
        "ssp": {
            "dust": {
                "dust_grain_density": 3.0,
                "extinction_model": "Cardelli89",
                "Rv": 3.1,
                "dust_to_gas_model": "power law slope free",
                "Xco": "MW",
            }
        },
    }


@pytest.fixture
def mock_rubixdata():
    class MockGas:
        def __init__(self):
            self.coords = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            self.pixel_assignment = jnp.array([[0, 1]])
            self.metals = jnp.array(
                [[[0.01, 0.02, 0.03, 0.04, 0.05], [0.06, 0.07, 0.08, 0.09, 0.1]]]
            )
            self.mass = jnp.array([[1.0, 2.0]])

    class MockStars:
        def __init__(self):
            self.coords = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            self.pixel_assignment = jnp.array([[0, 1]])
            self.mass = jnp.array([[1.0, 2.0]])
            self.spectra = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])

    class MockRubixData(RubixData):
        def __init__(self):
            self.gas = MockGas()
            self.stars = MockStars()

    return MockRubixData()


def test_spaxel_extinction_Cardelli(mock_config, mock_rubixdata):
    wavelength = jnp.array([5000.0, 6000.0])
    n_spaxels = 2
    spaxel_area = jnp.array([1.0, 1.0])

    result = apply_spaxel_extinction(
        mock_config, mock_rubixdata, wavelength, n_spaxels, spaxel_area
    )

    assert result.shape == (1, 2, 2)
    assert jnp.all(result >= 0)


@pytest.fixture
def mock_config():
    return {
        "logger": None,
        "constants": {
            "MASS_OF_PROTON": 1.6726219e-24,
            "MSUN_TO_GRAMS": 1.989e33,
            "KPC_TO_CM": 3.086e21,
        },
        "ssp": {
            "dust": {
                "dust_grain_density": 3.0,
                "extinction_model": "Gordon23",
                "Rv": 3.1,
                "dust_to_gas_model": "power law slope free",
                "Xco": "MW",
            }
        },
    }


def test_spaxel_extinction_Gordon(mock_config, mock_rubixdata):
    wavelength = jnp.array([5000.0, 6000.0])
    n_spaxels = 2
    spaxel_area = jnp.array([1.0, 1.0])

    result = apply_spaxel_extinction(
        mock_config, mock_rubixdata, wavelength, n_spaxels, spaxel_area
    )

    assert result.shape == (1, 2, 2)
    assert jnp.all(result >= 0)


@pytest.fixture
def mock_config():
    return {
        "logger": None,
        "constants": {
            "MASS_OF_PROTON": 1.6726219e-24,
            "MSUN_TO_GRAMS": 1.989e33,
            "KPC_TO_CM": 3.086e21,
        },
        "ssp": {
            "dust": {
                "dust_grain_density": 3.0,
                "extinction_model": "Cardelli89",
                "Rv": 3.1,
                "dust_to_gas_model": "power law slope free",
                "Xco": "MW",
            }
        },
    }


@pytest.fixture
def mock_rubixdata():
    class MockGas:
        def __init__(self):
            self.coords = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            self.pixel_assignment = jnp.array([[0, 1]])
            self.metals = jnp.array(
                [[[0.01, 0.02, 0.03, 0.04, 0.05], [0.06, 0.07, 0.08, 0.09, 0.1]]]
            )
            self.mass = jnp.array([[1.0, 2.0]])

    class MockStars:
        def __init__(self):
            self.coords = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            self.pixel_assignment = jnp.array([[0, 1]])
            self.mass = jnp.array([[1.0, 2.0]])
            self.spectra = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])

    class MockRubixData(RubixData):
        def __init__(self):
            self.gas = MockGas()
            self.stars = MockStars()

    return MockRubixData()


def test_calculate_dust_to_gas_ratio_power_law_slope_free_MW():
    gas_metallicity = jnp.array([8.0, 8.5])
    model = "power law slope free"
    Xco = "MW"
    result = calculate_dust_to_gas_ratio(gas_metallicity, model, Xco)
    assert result.shape == (2,)
    assert jnp.all(result >= 0)


def test_calculate_dust_to_gas_ratio_broken_power_law_fit_MW():
    gas_metallicity = jnp.array([8.0, 8.5])
    model = "broken power law fit"
    Xco = "MW"
    result = calculate_dust_to_gas_ratio(gas_metallicity, model, Xco)
    assert result.shape == (2,)
    assert jnp.all(result >= 0)


def test_calculate_dust_to_gas_ratio_power_law_slope_free_Z():
    gas_metallicity = jnp.array([8.0, 8.5])
    model = "power law slope free"
    Xco = "Z"
    result = calculate_dust_to_gas_ratio(gas_metallicity, model, Xco)
    assert result.shape == (2,)
    assert jnp.all(result >= 0)


def test_calculate_dust_to_gas_ratio_broken_power_law_fit_Z():
    gas_metallicity = jnp.array([8.0, 8.5])
    model = "broken power law fit"
    Xco = "Z"
    result = calculate_dust_to_gas_ratio(gas_metallicity, model, Xco)
    assert result.shape == (2,)
    assert jnp.all(result >= 0)


def test_invalid_extinction_model(mock_config, mock_rubixdata):
    # Modify the config to use an invalid extinction model
    mock_config["ssp"]["dust"]["extinction_model"] = "InvalidModel"

    wavelength = jnp.array([5000.0, 6000.0])
    n_spaxels = 2
    spaxel_area = jnp.array([1.0, 1.0])

    with pytest.raises(
        ValueError,
        match="Extinction model 'InvalidModel' is not available. Choose from",
    ):
        apply_spaxel_extinction(
            mock_config, mock_rubixdata, wavelength, n_spaxels, spaxel_area
        )


def test_get_extinction_raises_value_error_for_missing_dust_key():
    config = {"ssp": {}, "galaxy": {"dist_z": 0.1}}
    with pytest.raises(
        ValueError, match="Dust configuration not found in config file."
    ):
        get_extinction(config)


def test_get_extinction_raises_value_error_for_missing_extinction_model():
    config = {"ssp": {"dust": {}}, "galaxy": {"dist_z": 0.1}}
    with pytest.raises(
        ValueError, match="Extinction model not found in dust configuration."
    ):
        get_extinction(config)


@patch("rubix.core.dust.get_telescope")
@patch("rubix.core.dust.get_cosmology")
@patch("rubix.core.dust.calculate_spatial_bin_edges")
@patch("rubix.core.dust.apply_spaxel_extinction")
@patch("rubix.core.dust.get_logger")
def test_get_extinction_applies_dust_extinction(
    mock_get_logger,
    mock_apply_spaxel_extinction,
    mock_calculate_spatial_bin_edges,
    mock_get_cosmology,
    mock_get_telescope,
):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    mock_telescope = MagicMock()
    mock_telescope.sbin = 2
    mock_telescope.wave_seq = [5000, 6000, 7000]
    mock_get_telescope.return_value = mock_telescope

    mock_calculate_spatial_bin_edges.return_value = (None, 1.0)

    config = {
        "ssp": {"dust": {"extinction_model": "some_model"}},
        "galaxy": {"dist_z": 0.1},
    }

    rubixdata = MagicMock(spec=RubixData)
    rubixdata.stars.spectra = [1, 2, 3]

    calculate_extinction = get_extinction(config)
    result = calculate_extinction(rubixdata)

    mock_logger.info.assert_called_with(
        "Applying dust extinction to the spaxel data..."
    )
    mock_apply_spaxel_extinction.assert_called_with(
        config, rubixdata, [5000, 6000, 7000], 4, 1.0
    )
    assert result == rubixdata
