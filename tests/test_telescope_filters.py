import pytest
from unittest.mock import MagicMock, patch, call
from rubix.telescope.filters.filters import (
    Filter,
    FilterCurves,
    convolve_filter_with_spectra,
    load_filter,
    save_filters,
    print_filter_list,
    print_filter_list_info,
    print_filter_property,
    _load_filter_list_for_instrument,
)

import os
from astropy.table import Table

import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

# Use the Agg backend for testing to avoid opening a figure window
matplotlib.use("Agg")


def test_filter_initialization():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    assert jnp.all(filt.wavelength == wavelength)
    assert jnp.all(filt.response == response)
    assert filt.name == name
    assert str(filt) == name


def test_filter_plot():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    # Test with provided axes
    fig, ax = plt.subplots()
    filt.plot(ax)
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == name
    plt.close(fig)

    # Test with no axes provided
    fig = plt.figure()
    filt.plot()  # This should use plt.gca()
    ax = plt.gca()
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == name
    plt.close(fig)


def test_filter_call():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    new_wavelengths = jnp.array([450, 550])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    interpolated_response = filt(new_wavelengths)

    assert jnp.allclose(interpolated_response, jnp.array([0.3, 0.7]), atol=1e-2)


def test_filter_curves_initialization():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)
    filt_curves = FilterCurves([filt])

    assert len(filt_curves.filters) == 1
    assert filt_curves.filters[0] == filt
    assert len(filt_curves) == 1
    assert filt_curves[0] == filt


def test_filter_curves_plot():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)
    filt_curves = FilterCurves([filt])

    # filt_curves.plot()


def test_apply_filter_curves():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    spectra = jnp.array([1.0, 2.0, 3.0])
    cube = jnp.ones((2, 2, 3))
    name = "Test Filter"

    filt = Filter(wavelength, response, name)
    filt_curves = FilterCurves([filt])

    images = filt_curves.apply_filter_curves(cube, wavelength)

    assert len(images["image"]) == 1
    assert images["filter"][0] == name
    assert images["image"][0].shape == (2, 2)


def test_convolve_filter_with_spectra_single():
    # Define a simple filter and spectrum where the result of convolution is known
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([1.0, 1.0, 1.0])  # Flat response for simplicity
    spectra = jnp.array([2.0, 2.0, 2.0])  # Uniform spectra
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    # The convolved flux should be the sum of the spectra values times the spacing between wavelengths
    # Here, the spacing between each wavelength is 100, so the result should be 2*100 + 2*100 = 400
    convolved_flux = convolve_filter_with_spectra(filt, spectra, wavelength)

    expected_flux = 2.0 * 100 + 2.0 * 100  # Using trapezoidal integration

    assert jnp.isclose(convolved_flux, expected_flux, atol=1e-2)


def test_convolve_filter_with_spectra_cube():
    # Define a simple filter and spectrum cube where the result of convolution is known
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([1.0, 1.0, 1.0])  # Flat response for simplicity
    cube = jnp.ones((2, 2, 3))  # Uniform cube of spectra, all values are 1
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    # The convolved image should be the sum of the spectra values times the spacing between wavelengths
    # Here, the spacing between each wavelength is 100, and each spectrum sum is 1*100 + 1*100 = 200
    # Thus, each pixel in the convolved image should be 1*100 + 1*100 = 200
    convolved_image = convolve_filter_with_spectra(filt, cube, wavelength)

    expected_image_value = 1.0 * 100 + 1.0 * 100  # Using trapezoidal integration

    assert convolved_image.shape == (2, 2)
    assert jnp.allclose(convolved_image, expected_image_value, atol=1e-2)


def test_load_filter():
    facility = "SLOAN"
    instrument = "SDSS"
    filter_name = "r"
    filters_path = "/path/to/filters"

    with (
        patch("os.path.exists") as mock_exists,
        patch("astropy.table.Table.read") as mock_read,
        patch(
            "rubix.telescope.filters.filters._load_filter_list_for_instrument"
        ) as mock_load_filter,
    ):

        mock_exists.return_value = True
        mock_read.return_value = MagicMock()
        mock_load_filter.return_value = MagicMock()

        filter_curves = load_filter(facility, instrument, filter_name, filters_path)

        mock_exists.assert_called_once_with(os.path.join(filters_path, facility))
        mock_read.assert_called_once_with(
            f"{os.path.join(filters_path, facility)}/{facility}.csv"
        )

    assert isinstance(filter_curves, FilterCurves)


def test_load_filter_with_instrument_and_filter_name():
    facility = "SLOAN"
    instrument = "SDSS"
    filter_name = "r"
    filters_path = "/path/to/filters"

    with (
        patch("os.path.exists") as mock_exists,
        patch("astropy.table.Table.read") as mock_read,
        patch(
            "rubix.telescope.filters.filters._load_filter_list_for_instrument"
        ) as mock_load_filter,
    ):

        mock_exists.return_value = True
        mock_read.return_value = MagicMock()

        mock_filter = MagicMock()
        mock_filter.name = "SLOAN/SDSS.r"
        mock_load_filter.return_value = [mock_filter]

        filter_curves = load_filter(facility, instrument, filter_name, filters_path)

        assert isinstance(filter_curves, FilterCurves)
        assert len(filter_curves.filters) == 1
        assert filter_curves.filters[0].name == "SLOAN/SDSS.r"


def test_load_filter_with_instrument_and_filter_name_list():
    facility = "SLOAN"
    instrument = "SDSS"
    filter_name = ["r", "g", "b"]
    filters_path = "/path/to/filters"

    with (
        patch("os.path.exists") as mock_exists,
        patch("astropy.table.Table.read") as mock_read,
        patch(
            "rubix.telescope.filters.filters._load_filter_list_for_instrument"
        ) as mock_load_filter,
    ):

        mock_exists.return_value = True
        mock_read.return_value = MagicMock()

        mock_filter1 = MagicMock()
        mock_filter1.name = "SLOAN/SDSS.r"
        mock_filter2 = MagicMock()
        mock_filter2.name = "SLOAN/SDSS.g"
        mock_filter3 = MagicMock()
        mock_filter3.name = "SLOAN/SDSS.b"
        mock_load_filter.return_value = [mock_filter1, mock_filter2, mock_filter3]

        filter_curves = load_filter(facility, instrument, filter_name, filters_path)

        assert isinstance(filter_curves, FilterCurves)
        assert len(filter_curves.filters) == 3
        assert filter_curves.filters[0].name == "SLOAN/SDSS.r"
        assert filter_curves.filters[1].name == "SLOAN/SDSS.g"
        assert filter_curves.filters[2].name == "SLOAN/SDSS.b"


def test_load_filter_with_instrument_list():
    facility = "JWST"
    instrument = ["MIRI", "NIRCAM"]
    filters_path = "/path/to/filters"

    with (
        patch("os.path.exists") as mock_exists,
        patch("astropy.table.Table.read") as mock_read,
        patch(
            "rubix.telescope.filters.filters._load_filter_list_for_instrument"
        ) as mock_load_filter,
    ):

        mock_exists.return_value = True
        mock_table = MagicMock()
        mock_table["filterID"] = ["JWST/MIRI.F560W", "JWST/NIRCAM.F070W"]
        mock_read.return_value = mock_table

        def side_effect_function(*args, **kwargs):
            mock_filter = MagicMock()
            filt_ID = {
                "JWST/MIRI": "JWST/MIRI.F560W",
                "JWST/NIRCAM": "JWST/NIRCAM.F070W",
            }
            mock_filter.name = filt_ID[args[1]]
            return [mock_filter]

        mock_load_filter.side_effect = side_effect_function

        filter_curves = load_filter(facility, instrument, filters_path=filters_path)

        assert isinstance(filter_curves, FilterCurves)
        assert len(filter_curves.filters) == 2
        assert all(filter.name.startswith("JWST/") for filter in filter_curves.filters)


def test_load_filter_with_instrument_none():
    facility = "JWST"
    filters_path = "/path/to/filters"

    with (
        patch("os.path.exists") as mock_exists,
        patch("astropy.table.Table.read") as mock_read,
        patch(
            "rubix.telescope.filters.filters._load_filter_list_for_instrument"
        ) as mock_load_filter,
    ):

        mock_exists.return_value = True
        mock_read.return_value = MagicMock()

        mock_filter1 = MagicMock()
        mock_filter1.name = "JWST/MIRI.F560W"
        mock_filter2 = MagicMock()
        mock_filter2.name = "JWST/NIRCAM.F070W"
        mock_load_filter.return_value = [mock_filter1, mock_filter2]

        filter_curves = load_filter(facility, filters_path=filters_path)

        assert isinstance(filter_curves, FilterCurves)
        assert len(filter_curves.filters) == 2
        assert filter_curves.filters[0].name == "JWST/MIRI.F560W"
        assert filter_curves.filters[1].name == "JWST/NIRCAM.F070W"


def test_load_filter_with_no_instrument_and_filter_name():
    facility = "SLOAN"
    filter_name = "r"
    filters_path = "/path/to/filters"

    with pytest.raises(ValueError) as e:
        filter_curves = load_filter(
            facility, filter_name=filter_name, filters_path=filters_path
        )
        assert filter_curves is None
    assert (
        str(e.value)
        == "Cannot specify a filter_name without instrument. To avoid consfusion, please specify the instrument as well. Or if you like to load all filters for that instrument, set filter_name=None."
    )


def test_load_filter_with_no_instrument_and_no_filter_name():
    facility = "SDSS"
    filters_path = "/path/to/filters"

    with (
        patch("os.path.exists") as mock_exists,
        patch("astropy.table.Table.read") as mock_read,
        patch(
            "rubix.telescope.filters.filters._load_filter_list_for_instrument"
        ) as mock_load_filter,
    ):

        mock_exists.return_value = True
        mock_read.return_value = MagicMock()

        mock_filter = MagicMock()
        mock_filter.name = "SLOAN/SDSS.r"
        mock_load_filter.return_value = [mock_filter]

        filter_curves = load_filter(facility, filters_path)

        assert isinstance(filter_curves, FilterCurves)
        assert len(filter_curves.filters) == 1
        assert filter_curves.filters[0].name == "SLOAN/SDSS.r"


def test_load_filter_with_invalid_filter_name():
    facility = "SLOAN"
    instrument = "SDSS"
    filter_name = "invalid_filter"
    filters_path = "/path/to/filters"

    with (
        patch("os.path.exists") as mock_exists,
        patch("astropy.table.Table.read") as mock_read,
        patch(
            "rubix.telescope.filters.filters._load_filter_list_for_instrument"
        ) as mock_load_filter,
    ):

        try:
            filter_curves = load_filter(facility, instrument, filter_name, filters_path)
        except ValueError as e:
            assert str(e) == "Invalid filter_name specified."


def test_load_filter_with_nonexistent_filters_path():
    facility = "SLOAN"
    instrument = "SDSS"
    filters_path = "/nonexistent/path"

    with (
        patch("os.path.exists") as mock_exists,
        patch("rubix.telescope.filters.filters.save_filters") as mock_save_filters,
    ):

        mock_exists.return_value = False

        filter_curves = load_filter(facility, instrument, filters_path=filters_path)

        assert mock_save_filters.called_once_with(facility, filters_path)


@patch("os.makedirs")
@patch("rubix.telescope.filters.filters.SvoFps.get_transmission_data")
@patch("rubix.telescope.filters.filters.SvoFps.get_filter_list")
def test_save_filters(
    mock_get_filter_list, mock_get_transmission_data, mock_os_makedirs
):

    facility = "SLOAN"
    filters_path = "/path/to/filters"

    mock_filter_list = Table(names=("filterID",), data=[("SLOAN/SDSS.r",)])
    mock_get_filter_list.return_value = mock_filter_list
    mock_filter_data = MagicMock()
    mock_get_transmission_data.return_value = mock_filter_data

    with (
        patch("astropy.table.Table.write") as mock_table_write,
        patch("os.path.isdir") as mock_os_path_isdir,
    ):

        mock_os_path_isdir.return_value = False

        result = save_filters(facility, filters_path)

        mock_os_makedirs.assert_called_once_with(os.path.join(filters_path, facility))
        mock_get_filter_list.assert_called_once_with(facility=facility)
        mock_table_write.assert_any_call(
            f"{os.path.join(filters_path, facility)}/{facility}.csv", format="csv"
        )
        mock_get_transmission_data.assert_called_once_with("SLOAN/SDSS.r")
        mock_filter_data.write.assert_any_call(
            f"{os.path.join(filters_path, facility)}/SLOAN/SDSS.r.csv", format="csv"
        )
        assert result == mock_filter_list


@patch("rubix.telescope.filters.filters.Filter")
@patch("astropy.table.Table.read")
def test_load_filter_list_for_instrument(mock_table_read, mock_filter):
    # Arrange
    filter_table = Table(
        names=("filterID",),
        data=[("facility/instrument.filter1", "facility/instrument.filter2")],
    )
    filter_prefix = "facility/instrument"
    filter_name = ["filter1"]
    filter_dir = "/path/to/filters"

    mock_transmissivity = MagicMock()
    mock_transmissivity["Wavelength"].filled.return_value = [1, 2, 3]
    mock_transmissivity["Transmission"].filled.return_value = [0.1, 0.2, 0.3]
    mock_table_read.return_value = mock_transmissivity

    mock_filter_instance = MagicMock(spec=Filter)
    mock_filter.return_value = mock_filter_instance

    # Act
    filter_list = _load_filter_list_for_instrument(
        filter_table, filter_prefix, filter_name, filter_dir
    )

    # Assert
    mock_table_read.assert_called_once_with(
        f"{filter_dir}/{filter_prefix}.{filter_name[0]}.csv"
    )
    mock_filter.assert_called()
    assert mock_filter.called_once_with(
        mock_transmissivity["Wavelength"].filled.return_value,
        mock_transmissivity["Transmission"].filled.return_value,
        f"{filter_prefix}.{filter_name[0]}",
    )
    assert len(filter_list) == 1
    assert filter_list[0] == mock_filter_instance


@patch("rubix.telescope.filters.filters.SvoFps.get_filter_list")
@patch("builtins.print")
def test_print_filter_list(mock_print, mock_get_filter_list):

    facility = "SLOAN"
    instrument = "SDSS"

    mock_filter_list = Table(names=("filterID",), data=[("SLOAN/SDSS.r",)])
    mock_get_filter_list.return_value = mock_filter_list

    print_filter_list(facility, instrument)

    mock_get_filter_list.assert_called_once_with(
        facility=facility, instrument=instrument
    )
    mock_print.assert_called_once_with(mock_filter_list["filterID"])


@patch("rubix.telescope.filters.filters.SvoFps.get_filter_list")
@patch("builtins.print")
def test_print_filter_list_info(mock_print, mock_get_filter_list):

    facility = "SLOAN"
    instrument = "SDSS"

    mock_filter_list = MagicMock()
    mock_filter_list.info = "Filter list info"
    mock_get_filter_list.return_value = mock_filter_list

    print_filter_list_info(facility, instrument)

    mock_get_filter_list.assert_called_once_with(
        facility=facility, instrument=instrument
    )
    mock_print.assert_called_once_with(mock_filter_list.info)


@patch("rubix.telescope.filters.filters.SvoFps.get_filter_list")
@patch("builtins.print")
def test_print_filter_property(mock_print, mock_get_filter_list):
    # Arrange
    facility = "SLOAN"
    instrument = "SDSS"
    filter_name = "r"

    mock_filter_list = MagicMock()
    mock_filter_info = MagicMock()
    mock_filter_list.loc.__getitem__.return_value = mock_filter_info
    mock_get_filter_list.return_value = mock_filter_list

    # Act
    filter_info = print_filter_property(facility, instrument, filter_name)

    # Assert
    mock_get_filter_list.assert_called_once_with(
        facility=facility, instrument=instrument
    )
    mock_filter_list.loc.__getitem__.assert_called_once_with(
        f"{facility}/{instrument}.{filter_name}"
    )
    mock_print.assert_called_once_with(mock_filter_info)
    assert filter_info == mock_filter_info
