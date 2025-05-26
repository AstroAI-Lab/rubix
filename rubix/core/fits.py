import numpy as np
from astropy.io import fits
from rubix.core.telescope import get_telescope
from rubix.logger import get_logger
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def store_fits(config, data, filepath):
    """
    Store the datacube in a FITS file.

    Parameters:
        config (dict): The configuration dictionary
        data (dict): The data dictionary
        filepath (str): The path to save the FITS file

    Returns:
        None
    """
    logger_config = config.get("logger", None)
    logger = get_logger(logger_config)

    """
    if "cube_type" not in config["data"]["args"]:
        datacube = data.stars.datacube
        parttype = "stars"
    elif config["data"]["args"]["cube_type"] == "stars":
        datacube = data.stars.datacube
        parttype = "stars"
    elif config["data"]["args"]["cube_type"] == "gas":
        datacube = data.gas.datacube
        parttype = "gas"
    """
    datacube = data

    telescope = get_telescope(config)

    hdr = fits.Header()
    hdr["SIMPLE"] = "T /conforms to FITS standard"
    hdr["PIPELINE"] = config["pipeline"]["name"]
    hdr["DIST_z"] = config["galaxy"]["dist_z"]
    hdr["ROTATION"] = config["galaxy"]["rotation"]["type"]
    hdr["SIM"] = config["simulation"]["name"]

    #For Illustris and NIHAO
    galaxy_id = config["data"]["load_galaxy_args"]["id"]
    snapshot = config["data"]["args"]["snapshot"]

    hdr["GALAXYID"] = galaxy_id
    object_name = f"{config['simulation']['name']} {galaxy_id}"
    hdr["SNAPSHOT"] = snapshot

    hdr["SUBSET"] = config["data"]["subset"]["use_subset"]
    hdr["SSP"] = config["ssp"]["template"]["name"]
    hdr["INSTR"] = config["telescope"]["name"]
    hdr["PSF"] = config["telescope"]["psf"]["name"]
    hdr["PSF_SIZE"] = config["telescope"]["psf"]["size"]
    hdr["PSFSIGMA"] = config["telescope"]["psf"]["sigma"]
    hdr["LSF"] = config["telescope"]["lsf"]["sigma"]
    hdr["S_TO_N"] = config["telescope"]["noise"]["signal_to_noise"]
    hdr["N_DISTR"] = config["telescope"]["noise"]["noise_distribution"]
    hdr["COSMO"] = config["cosmology"]["name"]

    hdr1 = fits.Header()
    hdr1["EXTNAME"] = "DATA"
    hdr1["OBJECT"] = object_name
    hdr1["BUNIT"] = "erg/(s*cm^2*A)"  # flux unit per Angstrom
    hdr1["CRPIX1"] = (datacube.shape[0] - 1) / 2
    hdr1["CRPIX2"] = (datacube.shape[1] - 1) / 2
    hdr1["CD1_1"] = telescope.spatial_res / 3600  # convert arcsec to deg
    hdr1["CD1_2"] = 0
    hdr1["CD2_1"] = 0
    hdr1["CD2_2"] = telescope.spatial_res / 3600  # convert arcsec to deg
    hdr1["CUNIT1"] = "deg"
    hdr1["CUNIT2"] = "deg"
    hdr1["CTYPE1"] = "RA---TAN"
    hdr1["CTYPE2"] = "DEC--TAN"
    hdr1["CTYPE3"] = "AWAV"
    hdr1["CUNIT3"] = "Angstrom"
    hdr1["CD3_3"] = telescope.wave_res
    hdr1["CRPIX3"] = 1
    hdr1["CRVAL3"] = telescope.wave_range[0]
    hdr1["CD1_3"] = 0
    hdr1["CD2_3"] = 0
    hdr1["CD3_1"] = 0
    hdr1["CD3_2"] = 0

    empty_primary = fits.PrimaryHDU(header=hdr)
    image_hdu1 = fits.ImageHDU(datacube.T, header=hdr1)

    output_filename = (
        f"{filepath}{config['simulation']['name']}_id{galaxy_id}_snap{snapshot}_"
        f"subset{config['data']['subset']['use_subset']}.fits"
    )

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    hdul = fits.HDUList([empty_primary, image_hdu1])
    hdul.writeto(output_filename, overwrite=True)
    logger.info(f"Datacube saved to {output_filename}")

def load_fits(filepath):
    """
    Load a FITS file and return the datacube.

    Parameters:
        filepath (str): The path to the FITS file

    Returns:
        The cube object from mpdaf
    """
    cube = Cube(filename=filepath)
    return cube
