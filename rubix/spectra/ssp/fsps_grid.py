"""Use python-fsps to retrieve a block of Simple Stellar Population (SSP) data
adapted from https://github.com/ArgonneCPAC/dsps/blob/main/dsps/data_loaders/retrieve_fsps_data.py"""

import numpy as np
from rubix.logger import get_logger
from rubix import config as rubix_config
from rubix.paths import TEMPLATE_PATH
import h5py
import os
import importlib
from .grid import SSPGrid
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# Setup a logger based on the config
logger = get_logger()

HAS_FSPS = importlib.util.find_spec("fsps") is not None
if HAS_FSPS:
    import fsps
else:
    logger.warning(
        "python-fsps is not installed. Please install it to use this function. Install using pip install fsps and check the installation page: https://dfm.io/python-fsps/current/installation/ for more details. Especially, make sure to set all necessary environment variables."
    )


@jaxtyped(typechecker=typechecker)
def retrieve_ssp_data_from_fsps(
    add_neb_emission: bool = True,
    imf_type: int = 2,
    zmet=None,
    tage: float = 0.0,
    peraa: bool = True,
    **kwargs,
) -> "SSPGrid":
    """Use python-fsps to populate arrays and matrices of data
    for the default simple stellar populations (SSPs) in the shapes expected by DSPS
    adapted from https://github.com/ArgonneCPAC/dsps/blob/main/dsps/data_loaders/retrieve_fsps_data.py

    Parameters
    ----------
    add_neb_emission : bool, optional
        Argument passed to fsps.StellarPopulation. Default is True.

    imf_type : int, optional
        Argument passed to fsps.StellarPopulation to specify the IMF type. Default is 2 and specifies Chabrier (2003).
        See https://dfm.io/python-fsps/current/stellarpop_api/#example for more details.

    zmet : int, optional
        Argument passed to fsps.StellarPopulation to specify the metallicity index. Default is None.

    tage : float, optional
        Argument passed to fsps.StellarPopulation to specify the age of the SSP. Default is 0.0.

    peraa : bool, optional
        Argument passed to fsps.StellarPopulation to specify whether the spectrum should be returned in Lsun/Angstrom (True) or Lsun/Hz (False). Default is True.

    kwargs : optional
        Any keyword arguments passed to the retrieve_ssp_data_from_fsps function will be
        passed on to fsps.StellarPopulation.

    Returns
    -------
    ssp_lgmet : ndarray of shape (n_met, )
        Array of log10(Z) of the SSP templates
        where dimensionless Z is the mass fraction of elements heavier than He

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    ssp_wave : ndarray of shape (n_wave, )

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    Notes
    -----
    The retrieve_ssp_data_from_fsps function is just a wrapper around
    python-fsps without any other dependencies. This standalone function
    should be straightforward to modify to use python-fsps to build
    alternate SSP data blocks.

    All DSPS functions operate on plain ndarrays, so user-supplied data
    storing alternate SSP models is supported. You will just need to
    pack your SSP data into arrays with shapes matching the shapes of
    the arrays returned by this function.

    """
    assert HAS_FSPS, "Must have python-fsps installed to use this function"
    import fsps

    config = rubix_config["ssp"]["templates"]["FSPS"]

    sp = fsps.StellarPopulation(zcontinuous=0, imf_type=imf_type)
    ssp_lgmet = np.log10(sp.zlegend)
    nzmet = ssp_lgmet.size
    ssp_lg_age_gyr = sp.log_age - 9.0
    spectrum_collector = []
    for zmet_indx in range(1, ssp_lgmet.size + 1):
        print("...retrieving zmet = {0} of {1}".format(zmet_indx, nzmet))
        sp = fsps.StellarPopulation(
            zcontinuous=0,
            zmet=zmet_indx,
            add_neb_emission=add_neb_emission,
            imf_type=imf_type,
            **kwargs,
        )
        _wave, _fluxes = sp.get_spectrum(zmet=zmet, tage=tage, peraa=peraa)
        spectrum_collector.append(_fluxes)
    ssp_wave = np.array(_wave)
    ssp_wave_centered = ssp_wave - 1.5
    ssp_flux = np.array(spectrum_collector)

    grid = SSPGrid(ssp_lg_age_gyr, ssp_lgmet, ssp_wave_centered, ssp_flux)
    grid.__class__.__name__ = config["name"]
    return grid


@jaxtyped(typechecker=typechecker)
def write_fsps_data_to_disk(
    outname: str,
    file_location=TEMPLATE_PATH,
    add_neb_emission: bool = True,
    imf_type: int = 2,
    peraa: bool = True,
    **kwargs,
):
    """
    Write FSPS ssp template data to disk in HDF5 format.
    adapted from https://github.com/ArgonneCPAC/dsps/blob/main/scripts/write_fsps_data_to_disk.py

    Args:
        outname (str): The name of the output file.
        file_location (str, optional): The location where the file will be saved. Defaults to TEMPLATE_PATH.

    Returns:
        None
    """

    ssp_data = retrieve_ssp_data_from_fsps(
        add_neb_emission=True, imf_type=2, peraa=True, **kwargs
    )
    file_path = os.path.join(file_location, outname)

    logger.info(
        f"Writing created FSPS data to disk under the following path: {file_path}."
    )
    with h5py.File(file_path, "w") as hdf:
        for key, arr in zip(ssp_data.keys(), ssp_data):
            hdf[key] = arr
