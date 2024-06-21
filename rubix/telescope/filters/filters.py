import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from typing import List, Union, Optional
from rubix.paths import FILTERS_PATH
from rubix.logger import get_logger
from astropy.table import Table
from astroquery.svo_fps import SvoFps
import os

_logger = get_logger()


class Filter(eqx.Module):
    """
    A class representing a single filter with wavelength and response data.

    Attributes
    ----------
    wavelength : Float[Array, "n_wavelengths"]
        The wavelengths at which the filter response is defined.
    response : Float[Array, "n_wavelengths"]
        The filter response at the corresponding wavelengths.
    name : str
        The name of the filter.
    """

    wavelength: Float[Array, " n_wavelengths"]
    response: Float[Array, " n_wavelengths"]
    name: str

    def __init__(self, wavelength, response, name: str):
        """
        Initialize the Filter with given wavelength, response, and name.

        Parameters
        ----------
        wavelength : array-like
            The wavelengths at which the filter response is defined.
        response : array-like
            The filter response at the corresponding wavelengths.
        name : str
            The name of the filter.
        """
        self.wavelength = jnp.array(wavelength)
        self.response = jnp.array(response)
        self.name = name

    def plot(self, ax=None):
        """
        Plot the filter response.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, the current axes (`plt.gca()`) will be used.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.wavelength, self.response, label=self.name)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Response")
        ax.set_title("Filter Responses")
        ax.legend()

    def __call__(self, new_wavelengths):
        """
        Interpolate the filter response at new wavelengths.

        Parameters
        ----------
        new_wavelengths : array-like
            The new wavelengths at which to interpolate the filter response.

        Returns
        -------
        jax.numpy.ndarray
            The interpolated filter response at the new wavelengths.
        """
        new_response = jnp.interp(new_wavelengths, self.wavelength, self.response)
        return new_response

    def __str__(self):
        """
        Return the name of the filter.

        Returns
        -------
        str
            The name of the filter.
        """
        return self.name

    def __repr__(self):
        """
        Return the name of the filter for representation.

        Returns
        -------
        str
            The name of the filter.
        """
        return self.name


class FilterCurves(eqx.Module):
    """
    A class representing a collection of filter curves.

    Attributes
    ----------
    filters : List[Filter]
        The list of filter objects.
    """

    filters: List[Filter]

    def __init__(self, filters):
        """
        Initialize the FilterCurves with a list of filters.

        Parameters
        ----------
        filters : List[Filter]
            The list of filter objects.
        """
        self.filters = filters

    def plot(self):
        """
        Plot all filter responses on the same figure.
        """
        fig, ax = plt.subplots()
        for filter in self.filters:
            filter.plot(ax)
        plt.show()

    def apply_filter_curves(self, cube, wavelengths):
        """
        Get the images of a cube of spectra through all filters.
        Parameters
        ----------
        cube : jax.numpy.ndarray
            The cube of spectra.
        wavelengths : jax.numpy.ndarray
            The wavelengths of the cube.
        Returns
        -------
        List[jax.numpy.ndarray]
            The list of images through each filter.
        """
        images = {"filter": [], "image": []}
        for filter in self.filters:
            images["filter"].append(filter.name)
            images["image"].append(
                convolve_filter_with_spectra(filter, cube, wavelengths)
            )

        return images

    def __getitem__(self, key):
        """
        Get a filter by index.

        Parameters
        ----------
        key : int
            The index of the filter to retrieve.

        Returns
        -------
        Filter
            The filter at the specified index.
        """
        return self.filters[key]

    def __len__(self):
        """
        Get the number of filters.

        Returns
        -------
        int
            The number of filters.
        """
        return len(self.filters)


def load_filter(
    filters_path: str,
    facility: str,
    instrument: Optional[Union[str, List[str]]] = None,
    filter_name: Optional[Union[str, List[str]]] = None,
):
    """
    Load a single filter or all filters of a given facility and instrument as Filter objects.
    If filters are locally present we load them from the specified path, otherwise we download them from the SVO Filter Profile Service (http://svo2.cab.inta-csic.es/theory/fps/index.php).
    Filters are implicitly stored in the format of SVO: 'facilty/instrument.filter.csv'

    Parameters
    ----------
    filters_path : str
        Path to load the filters from if present on disk, or to save the filters to if downloaded.

    facility : str
        Name of the facility. e.g 'SLOAN' for SDSS.

    instrument : str or list of str
        optional: default=None
        Name of the instrument/s. e.g 'SDSS' for 'SLOAN'.
        If None, all instruments are loaded.

    filter_name : str or list of str
        optional: default=None
        Name of the specific filter/s to load. e.g 'r' for 'SDSS.r' which loads only the SDSS r-band filter.
        If None, all filters of the facility and instrument are loaded.

    Returns
    -------
    FilterCurves
        FilterCurves object containing the Filter objects.
    """

    # some sanity checks...
    # Check if the filter_name is specified without the instrument
    # we could allow for this, but this will make the code more complex.
    if instrument is None and filter_name is not None:
        raise ValueError(
            "Cannot specify a filter_name without instrument. To avoid consfusion, please specify the instrument as well. Or if you like to load all filters for that instrument, set filter_name=None."
        )

    # Try Loading the filters data from the FILTERS_PATH
    filter_dir = os.path.join(filters_path, facility)
    if os.path.exists(filter_dir):
        filter_table = Table.read(f"{filter_dir}/{facility}.csv")
    else:
        filter_table = save_filters(filters_path, facility)

    # make table searchable by filterID
    filter_table.add_index("filterID")

    # check if one specific filter is requested and create a list of lenght 1 so we can use the same logic as for multiple filters
    if isinstance(filter_name, str):
        filter_name = [filter_name]

    filter_curves = []
    if isinstance(instrument, str):
        # we have a single instrument
        filter_ID = f"{facility}.{instrument}"
        filter_curves.extend(
            _load_filter_list_for_instrument(filter_dir, filter_ID, filter_name)
        )

    elif isinstance(instrument, list):
        for inst in instrument:
            filter_ID = f"{facility}.{inst}"
            filter_curves.extend(
                _load_filter_list_for_instrument(filter_dir, filter_ID, filter_name)
            )

    elif instrument is None:
        # all instruments of this facility are requested
        # since we checked above that in this case also filter_name is None, we can directly load all filters for the facility.
        for ID in filter_table["filterID"]:
            if ID.startswith(facility):
                filter_curves.extend(
                    _load_filter_list_for_instrument(filter_dir, filter_ID, filter_name)
                )

    # Create a list of Filter objects
    # Filter object has wavelength and response attributes

    return FilterCurves(filter_curves)


def _load_filter_list_for_instrument(
    filter_dir: str, filter_prefix: str, filter_name: Optional[List[str]] = None
):
    """
    Load the filter list from the specified path.

    Parameters
    ----------
    filter_dir : str
        Path to load the filter list from.

    filter_prefix : str
        The filter prefix ID in the format of SVO: 'facilty/instrument'.

    filter_name : list of str
        optional: default=None
        Name of the specific filters to load. e.g 'r' for 'SDSS.r' which loads only the SDSS r-band filter.
        If None, all filters are loaded.

    Returns
    -------
    List[Filter]
        List of Filter objects containing the transmission curve.
    """

    filter_list = []
    if filter_name is None:
        # all filters for the instrument are requested
        for ID in filter_table["filterID"]:
            if ID.startswith(filter_ID):
                # filter_data = filter_table.loc[ID]
                transmissivity = Table.read(f"{filter_dir}/{ID}.csv")
                filter_list.append(
                    Filter(
                        jnp.asarray(transmissivity["Wavelength"].filled()),
                        jnp.asarray(transmissivity["Transmission"].filled()),
                        filter_ID,
                    )
                )
    elif isinstance(filter_name, list):
        # multiple specific filters are requested
        for f_name in filter_name:
            filter_ID = f"{filter_ID}.{f_name}"
            # filter_data = filter_table.loc[f_name]
            transmissivity = Table.read(f"{filter_dir}/{filter_ID}.csv")
            filter_list.append(
                Filter(
                    jnp.asarray(transmissivity["Wavelength"].filled()),
                    jnp.asarray(transmissivity["Transmission"].filled()),
                    filter_ID,
                )
            )
    else:
        _logger.error("Invalid filter_name type. Please provide a valid filter_name.")

    return FilterCurves(filter_list)


def save_filters(filters_path: str, facility: str):
    """
    Download all filters of a given facility from the Filter Profile Service of the Spanisch Virtual Observatory (http://svo2.cab.inta-csic.es/theory/fps/index.php) and save them as csv file to the specified path.

    Parameters
    ----------
    filters_path : str
        Path to save the filters as csv files.

    facility : str
        Name of the facility. e.g 'SLOAN' for SDSS.

    Returns
    -------
    Table
        Table containing the filter list.
    """

    _logger.info(f"Downloading telescope filter files for {facility}.")

    filter_dir = os.path.join(filters_path, facility)
    if not os.path.isdir(filter_dir):
        os.makedirs(filter_dir)

    filter_list = SvoFps.get_filter_list(facility=facility)
    filter_list.write(f"{filter_dir}/{facility}.csv", format="csv")

    for filter_name in filter_list["filterID"]:
        # Filter ID in the format SVO: 'facilty/instrument.filter'
        save_name = filter_name.split("/")[-1]
        filter_data = SvoFps.get_transmission_data(filter_name)
        filter_data.write(f"{filter_dir}/{filter_name}.csv", format="csv")

    _logger.info(f"Filter files for {facility} successfully downloaded!")
    _logger.info(f"File {save_name} saved to {filter_dir}.")

    return filter_list


def print_filter_list(facility: str, instrument=None):
    """
    Print the list of filters available for a given facility and instrument.
    If you want to see the list of all facilities and instruments, follow the link below:
    http://svo2.cab.inta-csic.es/theory/fps/index.php

    Parameters
    ----------
    facility : str
        Name of the facility. e.g 'SLOAN' for SDSS.

        instrument : str
        optional: default=None
        Name of the instrument. e.g 'NIRC2' for 'Keck'.

    Returns
    -------
    None
    """

    # TODO: for some facilities we might want to add a mapping from the fps names to more common names, e.g. 'SDSS' instead of 'SLOAN'.
    filter_list = SvoFps.get_filter_list(facility=facility, instrument=instrument)
    print(filter_list["filterID"])


def print_filter_list_info(facility: str, instrument=None):
    """
    Print the information of a filter list available for a given facility and instrument.
    If you want to see the list of all facilities and instruments, follow the link below:
    http://svo2.cab.inta-csic.es/theory/fps/index.php

    Parameters
    ----------
    facility : str
        Name of the facility. e.g 'SLOAN' for SDSS.

    instrument : str
    optinal: default=None
        Name of the instrument. e.g 'NIRC2' for 'Keck'.

    Returns
    -------
    None
    """
    # TODO: for some facilities we might want to add a mapping from the fps names to more common names, e.g. 'SDSS' instead of 'SLOAN'.
    filter_list = SvoFps.get_filter_list(facility=facility, instrument=instrument)
    print(filter_list.info)


def convolve_filter_with_spectra(
    filter: Filter,
    spectra: Union[
        Float[Array, " n_wavelengths"], Float[Array, " n_x n_y n_wavelengths"]
    ],
    wavelengths: Float[Array, " n_wavelengths"],
) -> Union[Float[Array, "1"], Float[Array, " n_x n_y"]]:
    """
    Convolves a single filter with a single spectrum or a cube of spectra.

    Parameters
    ----------
    filter : Filter
        The filter to convolve with the spectrum or cube.
    spectrum_or_cube : jax.numpy.ndarray
        The spectrum or cube of spectra.
    wavelengths : jax.numpy.ndarray
        The wavelengths of the spectrum or cube.

    Returns
    -------
    jax.numpy.ndarray
        The convolved flux value for a single spectrum or the convolved image for a cube of spectra.
    """
    # Interpolate the filter response to the wavelengths
    filter_response = filter(wavelengths)

    if spectra.ndim == 1:
        # Single spectrum case
        convolved_flux = jnp.trapezoid(spectra * filter_response, wavelengths)
        return convolved_flux
    elif spectra.ndim == 3:
        # Cube of spectra case
        convolved_image = jnp.trapezoid(spectra * filter_response, wavelengths, axis=-1)
        return convolved_image
    else:
        raise ValueError("Input array must be 1D (spectrum) or 3D (cube of spectra).")
