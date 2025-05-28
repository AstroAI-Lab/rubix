import os
from dataclasses import dataclass, fields
from typing import List, Tuple, Union

import equinox as eqx
import h5py
import jax.numpy as jnp
import requests
from astropy import units as u
from astropy.io import fits
from beartype import beartype as typechecker
from interpax import interp2d
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int, jaxtyped

from rubix import config as rubix_config
from rubix.logger import get_logger

SSP_UNITS = rubix_config["ssp"]["units"]


@dataclass
class SSPGrid:
    """
    Base class for all SSP models.
    """

    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, "metallicity_bins age_bins wavelength_bins"]
    # This does not work with jax.jit, gives error that str is not valid Jax type
    # units: Dict[str, str] = eqx.field(default_factory=dict)

    def __init__(self, age, metallicity, wavelength, flux, _logger=None):
        self.age = jnp.asarray(age)
        self.metallicity = jnp.asarray(metallicity)
        self.wavelength = jnp.asarray(wavelength)
        self.flux = jnp.asarray(flux)
        # self.units = SSP_UNITS

    @jaxtyped(typechecker=typechecker)
    def keys(self) -> List[str]:
        """
        Returns the keys of the dataclass.

        Parameters
        ----------
        Args:
            None

        Returns:
            List of keys of the dataclass.
        """
        return [f.name for f in fields(self)]

    def __iter__(self):
        yield from (getattr(self, field.name) for field in fields(self))

    @jaxtyped(typechecker=typechecker)
    def get_lookup_interpolation(
        self, method: str = "cubic", extrap: int = 0
    ) -> Partial:
        """
        Returns a 2D interpolation function for the SSP grid.

        The function can be called with metallicity and age as arguments to get the flux at that metallicity and age.

        Parameters
        ----------
        method : str
            The method to use for interpolation. Default is "cubic".
        extrap: float, bool or tuple
            The value to return for points outside the interpolation domain. Default is 0.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.Interpolator2D.html#interpax.Interpolator2D

        Returns:
            The 2D interpolation function `Interp2D`.

        Example 1
        ----------
        >>> grid = SSPGrid(...)
        >>> lookup = grid.get_lookup_interpolation()
        >>> metallicity = 0.02
        >>> age = 1e9
        >>> flux = lookup(metallicity, age)

        Example 2
        ----------
        >>> import matplotlib.pyplot as plt
        >>> from rubix.spectra.ssp.templates import BruzualCharlot2003
        >>> from jax import jit

        >>> ssp = BruzualCharlot2003
        >>> wave = ssp.wavelength

        >>> age_index = 0
        >>> met_index = 3
        >>> target_age = ssp.age[age_index] + 0.5*(ssp.age[age_index+1] - ssp.age[age_index])
        >>> target_met = ssp.metallicity[met_index] + 0.5*(ssp.metallicity[met_index+1] - ssp.metallicity[met_index])

        >>> lookup = ssp.get_lookup_interpolation()
        >>> spec_calc = lookup(target_met, target_age)
        >>> spec_true = ssp.flux[met_index, age_index, :]

        >>> plt.plot(wave, spec_calc, label='calc')
        >>> plt.plot(wave, spec_true, label='true')
        >>> plt.legend()
        >>> plt.yscale('log')
        """

        # Bind the SSP grid to the interpolation function
        interp = Partial(
            interp2d,
            method=method,
            x=self.metallicity,
            y=self.age,
            f=self.flux,
            extrap=extrap,
        )
        interp.__doc__ = (
            "Interpolation function for SSP grid, args: f(metallicity, age)"
        )
        return interp

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def convert_units(
        data: Union[Float[Array, "..."], Int[Array, "..."]],
        from_units: str,
        to_units: str,
    ) -> Float[Array, "..."]:
        """
        Convert the units of the data from `from_units` to `to_units`.

        Parameters
        ----------
        Args:
            data (array-like): The data to convert.
            from_units (str): The units of the data.
            to_units (str): The units to convert to.

        Returns:
            The data converted to the new units.
        """
        quantity = u.Quantity(data, from_units)
        return jnp.array(quantity.to(to_units).value, dtype=jnp.float32)

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def checkout_SSP_template(config: dict, file_location: str) -> str:
        """
        Check if the SSP template exists on disk, if not download it
        from the given URL in the configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        file_location : str
            Location to save the template file.

        Returns:
            The path to the file as str.
        """

        _logger = get_logger()
        file_path = os.path.join(file_location, config["file_name"])
        source = config["source"]
        if not config["source"].endswith("/"):
            source += "/"

        if not os.path.exists(file_path):
            _logger.info(
                f'[SSPModels] File {file_path} not found. Downloading it from {config["source"]}'
            )
            try:
                response = requests.get(source + config["file_name"])
            except requests.exceptions.SSLError as SSLerr:
                _logger.warning(f"[SSPModels] Error: {SSLerr}")
                _logger.warning(
                    f"[SSPModels] Trying to download file {config['file_name']} from url {source} without SSL verification."
                )
                try:
                    response = requests.get(source + config["file_name"], verify=False)
                except requests.exceptions.RequestException as err:
                    _logger.error(f"[SSPModels] Error: {err}")
                    # except requests.exceptions.HTTPError as errh:
                    #    print("Http Error:",errh)
                    raise FileNotFoundError(
                        f"Could not download file {config['file_name']} from url {source}."
                    )
            except requests.exceptions.RequestException as err:
                _logger.error(f"[SSPModels] Error: {err}")
                # except requests.exceptions.HTTPError as errh:
                #    print("Http Error:",errh)
                raise FileNotFoundError(
                    f"Could not download file {config['file_name']} from url {source}."
                )

            response.raise_for_status()

            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                    _logger.info(
                        f'[SSPModels] File {config["file_name"]} downloaded successfully!'
                    )
                return file_path
            else:
                raise FileNotFoundError(
                    f"Could not download file {config['file_name']} from url {source}."
                )
        else:
            return file_path

    @jaxtyped(typechecker=typechecker)
    @classmethod
    def from_file(cls, config: dict, file_location: str):
        """
        Template function to load a SSP grid from a file.

        Parameters
        ----------
            config (dict): Configuration dictionary.
            file_location (str): Location of the file.

        Returns:
            The SSP grid SSPGrid in the correct units.
        """

        # Initialize an empty zero length array for each field
        # in the SSP configuration.
        # Actual loading of templates needs to be implemented in the subclasses.

        ssp_data = {}
        for field_name, field_info in config["fields"].items():
            ssp_data[field_info["name"]] = jnp.empty(0)

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid


class HDF5SSPGrid(SSPGrid):
    """
    Class for SSP models stored in HDF5 format.
    Mainly used for custom collection of Bruzual & Charlot 2003 models and MILES models.

    Example
    -------
    >>> config = {
    ...     "name": "Bruzual & Charlot (2003)",
    ...     "format": "HDF5",
    ...     "source": "https://www.bruzual.org/bc03/",
    ...     "file_name": "BC03lr.h5",
    ...     "fields": {
    ...         "age": {
    ...             "name": "age",
    ...             "units": "Gyr",
    ...             "in_log": False
    ...         },
    ...         "metallicity": {
    ...             "name": "metallicity",
    ...             "units": "",
    ...             "in_log": False
    ...         },
    ...         "wavelength": {
    ...             "name": "wavelength",
    ...             "units": "Angstrom",
    ...             "in_log": False
    ...         },
    ...         "flux": {
    ...             "name": "flux",
    ...             "units": "Lsun/Angstrom",
    ...             "in_log": False
    ...         }
    ...     }
    ... }

    >>> from rubix.spectra.ssp.grid import HDF5SSPGrid
    >>> ssp = HDF5SSPGrid.from_file(config, file_location="../rubix/spectra/ssp/templates")

    >>> ssp.age.shape
    >>> ssp.metallicity.shape
    >>> ssp.wavelength.shape
    >>> ssp.flux.shape
    """

    # Do we need this again or is this taken care of by inheriting from SSPGrid?
    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, "metallicity_bins age_bins wavelength_bins"]
    # This does not work with jax.jit, gives error that str is not valid Jax type
    # units: Dict[str, str] = eqx.field(default_factory=dict)

    def __init__(self, age, metallicity, wavelength, flux):
        super().__init__(age, metallicity, wavelength, flux)

    @jaxtyped(typechecker=typechecker)
    @classmethod
    def from_file(cls, config: dict, file_location: str) -> SSPGrid:
        """
        Load a SSP grid from a HDF5 file.

        Parameters
        ----------
            config (dict): Configuration dictionary.
            file_location (str): Location of the file.

        Returns:
            The SSP grid `SSPGrid` in the correct units.
        """

        if config.get("format", "").lower() not in ["hdf5", "fsps"]:
            raise ValueError("Configured file format is not HDF5.")

        file_path = cls.checkout_SSP_template(config, file_location)

        ssp_data = {}
        with h5py.File(file_path, "r") as f:
            for field_name, field_info in config["fields"].items():
                data = f[field_info["name"]][:]  # type: ignore
                data = jnp.power(10, data) if field_info["in_log"] else data  # type: ignore
                data = jnp.array(data, dtype=jnp.float32)
                data = cls.convert_units(
                    data, field_info["units"], SSP_UNITS[field_name]
                )
                ssp_data[field_name] = data

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid


class pyPipe3DSSPGrid(SSPGrid):
    """
    Class for all SSP models supported by the pyPipe3D project.
    See http://ifs.astroscu.unam.mx/pyPipe3D/templates/ for more information.

    Example
    -------
    >>> config = {
    ...     "name": "Mastar Charlot & Bruzual (2019)",
    ...     "format": "pyPipe3D",
    ...     "source": "https://ifs.astroscu.unam.mx/pyPipe3D/templates/",
    ...     "file_name": "MaStar_CB19.slog_1_5.fits.gz",
    ...     "fields": {
    ...         "age": {
    ...             "name": "age",
    ...             "units": "Gyr",
    ...             "in_log": False
    ...         },
    ...         "metallicity": {
    ...             "name": "metallicity",
    ...             "units": "",
    ...             "in_log": False
    ...         },
    ...         "wavelength": {
    ...             "name": "wavelength",
    ...             "units": "Angstrom",
    ...             "in_log": False
    ...         },
    ...         "flux": {
    ...             "name": "flux",
    ...             "units": "Lsun/Angstrom",
    ...             "in_log": False
    ...         }
    ...     }
    ... }

    >>> from rubix.spectra.ssp.grid import pyPipe3DSSPGrid
    >>> ssp = pyPipe3DSSPGrid.from_file(config, file_location="../rubix/spectra/ssp/templates")
    """

    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, "metallicity_bins age_bins wavelength_bins"]
    # This does not work with jax.jit, gives error that str is not valid Jax type
    # units: Dict[str, str] = eqx.field(default_factory=dict)

    def __init__(self, age, metallicity, wavelength, flux):
        super().__init__(age, metallicity, wavelength, flux)

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def get_wavelength_from_header(header, wave_axis=None) -> Array:
        """
        Generates a wavelength array using `header`, a :class:`astropy.io.fits.header.Header`
        instance, at axis `wave_axis`.

        wavelengths = CRVAL + CDELT*([0, 1, ..., NAXIS] + 1 - CRPIX)

        adapted from https://github.com/reginasar/TNG_MaNGA_mocks/blob/3229dd47b441aef380ef7dbfdf110f39e5c5a77c/sin_ifu_clean.py#L1466

        Parameters
        ----------
        header : :class:`astropy.io.fits.header.Header`
            FITS header with spectral data.

        wave_axis : int, optional
            The axis where the wavelength information is stored in `header`,
            (CRVAL, CDELT, NAXIS, CRPIX).
            Defaults to 1.

        Returns:
            Wavelengths array: wavelengths = CRVAL + CDELT*([0, 1, ..., NAXIS] + 1 - CRPIX)
        """
        if wave_axis is None:
            wave_axis = 1
        h = header

        crval = h[f"CRVAL{wave_axis}"]
        cdelt = h[f"CDELT{wave_axis}"]
        naxis = h[f"NAXIS{wave_axis}"]
        crpix = h[f"CRPIX{wave_axis}"]
        if not cdelt:
            cdelt = 1
        return crval + cdelt * (jnp.arange(naxis) + 1 - crpix)

    # @staticmethod
    # def get_normalization_wavelength(header, wavelength, flux_models, n_models):
    #    """
    #    Search for the normalization wavelength at the FITS header.
    #    If the key WAVENORM does not exists in the header, sweeps all the
    #    models looking for the wavelengths where the flux is closer to 1,
    #    calculates the median of those wavelengths and returns it.
    #
    #    TODO: defines a better normalization wavelength if it's not present
    #    in the header.
    #
    #    adapted from https://github.com/reginasar/TNG_MaNGA_mocks/blob/3229dd47b441aef380ef7dbfdf110f39e5c5a77c/sin_ifu_clean.py#L1466
    #
    #    Parameters
    #    ----------
    #    header : :class:`astropy.io.fits.header.Header`
    #        FITS header with spectral data.
    #
    #    wavelength : array like, wavelength of the model SSPs.
    #
    #    flux_models : array like, flux of the model SSPs.
    #
    #    n_models : int, number of models in the SSP grid.
    #
    #    Returns
    #    -------
    #    float
    #        The normalization wavelength.
    #    """
    #    try:
    #        wave_norm = header['WAVENORM']
    #    except Exception as ex:
    #        _closer = 1e-6
    #        probable_wavenorms = jnp.hstack([wavelength[(jnp.abs(flux_models[i] - 1) < _closer)]
    #                                    for i in range(n_models)])
    #        wave_norm = jnp.median(probable_wavenorms)
    #        print(f'[SSPModels] {ex}')
    #        print(f'[SSPModels] setting normalization wavelength to {wave_norm} A')
    #    return wave_norm

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def get_tZ_models(
        header, n_models: int
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """
        Reads the values of age, metallicity and mass-to-light at the
        normalization flux from the SSP models FITS file.

        adapted from https://github.com/reginasar/TNG_MaNGA_mocks/blob/3229dd47b441aef380ef7dbfdf110f39e5c5a77c/sin_ifu_clean.py#L1466

        Parameters
        ----------
        header : :class:`astropy.io.fits.header.Header`
            FITS header with spectral data.
        n_models : int, number of models in the SSP grid.

        Returns:
        array like
            Ages, in Gyr, in the sequence as they appear in FITS data.

        array like
            Metallicities in the sequence as they appear in FITS data.

        array like
            Mass-to-light value at the normalization wavelength.
        """
        ages = jnp.zeros(n_models, dtype=jnp.float32)
        Zs = jnp.zeros(n_models, dtype=jnp.float32)
        mtol = jnp.zeros(n_models, dtype=jnp.float32)
        for i in range(n_models):
            mult = {"Gyr": 1, "Myr": 1 / 1000}
            name_read_split = header[f"NAME{i}"].split("_")
            # removes 'spec_ssp_' from the name
            name_read_split = name_read_split[2:]
            _age = name_read_split[0]
            if "yr" in _age:
                mult = mult[_age[-3:]]  # Gyr or Myr
                _age = _age[:-3]
            else:
                mult = 1  # Gyr
            age = mult * jnp.float32(_age)
            _Z = name_read_split[1].split(".")[0]
            Z = jnp.float32(_Z.replace("z", "0."))
            ages = ages.at[i].set(age)
            Zs = Zs.at[i].set(Z)
            if jnp.float32(header[f"NORM{i}"]) != 0:
                mtol = mtol.at[i].set(1 / jnp.float32(header[f"NORM{i}"]))
            else:
                mtol = mtol.at[i].set(1)
        return jnp.unique(ages), jnp.unique(Zs), mtol

    @jaxtyped(typechecker=typechecker)
    @classmethod
    def from_file(cls, config: dict, file_location: str) -> SSPGrid:
        """
        Load a SSP grid from a fits file in pyPipe3D format.

        Parameters
        ----------
            config (dict): Configuration dictionary.
            file_location (str): Location of the file.

        Returns:
            The SSP grid SSPGrid in the correct units.
        """
        if config.get("format", "").lower() != "pypipe3d":
            raise ValueError("Configured file format is not fits.")

        file_path = cls.checkout_SSP_template(config, file_location)

        ssp_data = {}
        with fits.open(file_path) as f:
            _header = f[0].header
            # n_wave = _header['NAXIS1']
            n_models = _header["NAXIS2"]
            # pyPIPE3D uses the key WAVENORM to store the normalization wavelength
            # not sure what this is actually used for in the end.
            # Here we enable reading it, but we should make sure we understand what it is used for.
            # normalization_wavelength = get_normalization_wavelength(_header, wavelength, flux_models, n_models)
            ages, metallicities, m2l = cls.get_tZ_models(_header, n_models)
            wavelength = cls.get_wavelength_from_header(_header)

            # read in the flux of the models and multiply by the mass-to-light ratio to get the flux in Lsun/Msun
            # see also eq. A1 here https://arxiv.org/pdf/1811.04856.pdf
            template_flux = jnp.array(f[0].data, dtype=jnp.float32) / m2l[:, None]
            # reshape and bring into the correct order of metallcity, age, wavelength
            # to conform with the SSPGrid dataclass
            flux_models = template_flux.reshape(
                len(metallicities), len(ages), len(wavelength)
            )

            flux_models = template_flux.reshape(
                len(metallicities), len(ages), len(wavelength)
            )

            for field_name, field_info in config["fields"].items():
                if field_name == "flux":
                    data = flux_models
                elif field_name == "wavelength":
                    data = wavelength
                elif field_name == "age":
                    data = ages
                elif field_name == "metallicity":
                    data = metallicities
                else:
                    raise ValueError(f"Field {field_name} not recognized")

                data = jnp.power(10, data) if field_info["in_log"] else data  # type: ignore
                data = cls.convert_units(
                    data, field_info["units"], SSP_UNITS[field_name]
                )
                ssp_data[field_name] = data

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid


# TODO: build another class that handles eMILES, sMILES templates that are also used by the GECKOS survey.
# those will also have alpha enhancement and not only metallicity dependence. might need some changes to the
# interpolation function further down the pipeline...
