'''Module: flux.py
Core utility for converting luminosity datacubes into observed, normalized flux cubes
for inclusion in the Rubix pipeline.
'''
from typing import Callable

from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.spectra.ifu import convert_luminoisty_to_flux  # original spelling
from rubix.cosmology import PLANCK15

from .data import RubixData

@jaxtyped(typechecker=typechecker)
def get_converted_flux(config: dict) -> Callable[[RubixData], RubixData]:
    """
    Build a function converting `rubixdata.stars.datacube` from luminosity to flux,
    then normalizing by 1e-20.

    Parameters
    ----------
    config : dict
        Must contain `galaxy.dist_z` (float). Optionally `ifu.pixel_size` (float, default 1.0).

    Returns
    -------
    Callable[[RubixData], RubixData]
        Function that applies conversion in-place on `stars.datacube`.
    """
    # extract redshift
    try:
        redshift = config['galaxy']['dist_z']
    except KeyError:
        raise ValueError('Missing galaxy.dist_z in config')

    # compute luminosity distance
    lum_dist = PLANCK15.luminosity_distance_to_z(redshift)

    # pixel size for IFU conversion
    pixel_size = config.get('ifu', {}).get('pixel_size', 1.0)

    def convert_flux(rubixdata: RubixData) -> RubixData:
        # perform luminosity->flux conversion
        flux_cube = convert_luminoisty_to_flux(
            rubixdata.stars.datacube,
            lum_dist,
            redshift,
            pixel_size,
        )
        # apply fixed normalization
        rubixdata.stars.datacube = flux_cube / 1e-20
        return rubixdata

    return convert_flux
