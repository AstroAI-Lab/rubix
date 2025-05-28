from typing import Callable

from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.core.cosmology import get_cosmology
from rubix.logger import get_logger
from rubix.spectra.dust.dust_extinction import apply_spaxel_extinction
from rubix.telescope.utils import calculate_spatial_bin_edges

from .data import RubixData
from .telescope import get_telescope


@jaxtyped(typechecker=typechecker)
def get_extinction(config: dict) -> Callable:
    """
    Get the function to apply the dust extinction to the spaxel data.

    Parameters
    ----------
    config : dict
        The configuration dictionary.

    Returns
    -------
    Callable
        The function to apply the dust extinction to the spaxel data.
    """
    logger = get_logger(config.get("logger", None))

    # check if dust key exists in config file to ensure we really want to apply dust extinction
    if "dust" not in config["ssp"]:
        raise ValueError("Dust configuration not found in config file.")
    if "extinction_model" not in config["ssp"]["dust"]:
        raise ValueError("Extinction model not found in dust configuration.")

    # Get the telescope wavelength and spaxel number
    telescope = get_telescope(config)
    n_spaxels = int(telescope.sbin**2)
    wavelength = telescope.wave_seq

    galaxy_dist_z = config["galaxy"]["dist_z"]
    cosmology = get_cosmology(config)
    # Calculate the spatial bin edges
    _, spatial_bin_size = calculate_spatial_bin_edges(
        fov=telescope.fov,
        spatial_bins=telescope.sbin,
        dist_z=galaxy_dist_z,
        cosmology=cosmology,
    )

    spaxel_area = spatial_bin_size**2

    def calculate_extinction(rubixdata: RubixData) -> RubixData:
        """Apply the dust extinction to the spaxel data."""
        logger.info("Applying dust extinction to the spaxel data...")

        rubixdata.stars.spectra = apply_spaxel_extinction(
            config, rubixdata, wavelength, n_spaxels, spaxel_area
        )

        return rubixdata

    return calculate_extinction
