"""Use python-fsps to retrieve a block of Simple Stellar Population data.

Adapted from the DSPS ``retrieve_fsps_data.py`` workflow.
"""

import importlib
import os

import h5py
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix import config as rubix_config
from rubix.logger import get_logger
from rubix.paths import TEMPLATE_PATH

from .grid import SSPGrid

# Setup a logger based on the config
logger = get_logger()

HAS_FSPS = importlib.util.find_spec("fsps") is not None
if not HAS_FSPS:
    logger.warning(
        "python-fsps is not installed. Install it via `pip install fsps` "
        "and see https://dfm.io/python-fsps/current/installation/ for "
        "details, including required environment variables."
    )


@jaxtyped(typechecker=typechecker)
def retrieve_ssp_data_from_fsps(
    add_neb_emission: bool = True,
    imf_type: int = 2,
    zmet: int | None = None,
    tage: float = 0.0,
    peraa: bool = True,
    **kwargs,
) -> "SSPGrid":
    r"""
    Use python-fsps to populate arrays and matrices of data
    for the default simple stellar populations (SSPs) in the shapes expected by DSPS
    adapted from https://github.com/ArgonneCPAC/dsps/blob/main/dsps/data_loaders/retrieve_fsps_data.py

    Args:
        add_neb_emission (bool, optional): Whether to enable nebular emission
            in ``fsps.StellarPopulation``. Defaults to ``True``.
        imf_type (int, optional): IMF type identifier passed to
            ``fsps.StellarPopulation``. Defaults to ``2`` (Chabrier 2003).
            See https://dfm.io/python-fsps/current/stellarpop_api/#example for more details.
        zmet (int | None, optional): Metallicity index for
            ``fsps.StellarPopulation``. When ``None`` the default FSPS grid is
            used.
        tage (float, optional): SSP age in Gyr. Defaults to ``0.0``.
        peraa (bool, optional): If ``True`` return spectra in
            L$\_\{\odot\}$/\AA, otherwise L$\_\{\odot\}$/Hz. Defaults to
            ``True``.
        **kwargs: Additional keyword arguments forwarded to
            ``fsps.StellarPopulation``.

    Returns:
        SSPGrid: Grid containing age, metallicity, wavelength, and flux arrays
            in the shapes expected by downstream DSPS consumers.

    Raises:
        AssertionError: If ``python-fsps`` is not installed.

    Notes:
        The retrieve_ssp_data_from_fsps function is just a wrapper around
        python-fsps without any other dependencies. This standalone function
        should be straightforward to modify to use python-fsps to build
        alternate SSP data blocks.

        All DSPS functions operate on plain ndarrays, so user-supplied data
        storing alternate SSP models is supported. You will just need to
        pack your SSP data into arrays with shapes matching the shapes of
        the arrays returned by this function.
    """

    if not HAS_FSPS:
        raise AssertionError(
            "python-fsps is required to retrieve SSP data. Install the "
            "`fsps` package and configure the necessary environment "
            "variables."
        )
    fsps = importlib.import_module("fsps")

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
    # Adjust the wavelength grid to the bin centers. The offset equals half
    # the spacing between the first two wavelength samples, so it adapts to
    # the spectrum resolution (e.g., 1.5 Å when the spacing is 3 Å). This
    # keeps known lines such as Hα at 6563 Å centered after interpolation.
    offset = (_wave[1] - _wave[0]) / 2.0
    ssp_wave_centered = ssp_wave - offset
    ssp_flux = np.array(spectrum_collector)

    grid = SSPGrid(ssp_lg_age_gyr, ssp_lgmet, ssp_wave_centered, ssp_flux)
    grid.__class__.__name__ = config["name"]
    return grid


@jaxtyped(typechecker=typechecker)
def write_fsps_data_to_disk(
    outname: str,
    file_location: str | os.PathLike = TEMPLATE_PATH,
    add_neb_emission: bool = True,
    imf_type: int = 2,
    peraa: bool = True,
    **kwargs,
) -> None:
    """
    Write FSPS ssp template data to disk in HDF5 format.
    adapted from https://github.com/ArgonneCPAC/dsps/blob/main/scripts/write_fsps_data_to_disk.py

    Args:
        outname (str): Output filename, relative to ``file_location``.
        file_location (str | os.PathLike, optional): Directory for the generated file.
            Defaults to ``TEMPLATE_PATH``.
        add_neb_emission (bool, optional): Passed through to
            :func:`retrieve_ssp_data_from_fsps`. Defaults to ``True``.
        imf_type (int, optional): IMF type forwarded to
            :func:`retrieve_ssp_data_from_fsps`. Defaults to ``2``.
        peraa (bool, optional): Spectral units flag forwarded to
            :func:`retrieve_ssp_data_from_fsps`. Defaults to ``True``.
        **kwargs: Additional parameters forwarded to
            :func:`retrieve_ssp_data_from_fsps`.

    Returns:
        None
    """

    ssp_data = retrieve_ssp_data_from_fsps(
        add_neb_emission=add_neb_emission,
        imf_type=imf_type,
        peraa=peraa,
        **kwargs,
    )
    file_path = os.path.join(str(file_location), outname)

    logger.info(
        f"Writing created FSPS data to disk under the following path: {file_path}."
    )
    with h5py.File(file_path, "w") as hdf:
        for key, arr in zip(ssp_data.keys(), ssp_data):
            hdf[key] = arr
