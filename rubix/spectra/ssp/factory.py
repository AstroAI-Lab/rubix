from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix import config as rubix_config
from rubix.logger import get_logger
from rubix.paths import TEMPLATE_PATH
from rubix.spectra.ssp.fsps_grid import write_fsps_data_to_disk
from rubix.spectra.ssp.grid import HDF5SSPGrid, SSPGrid, pyPipe3DSSPGrid


@jaxtyped(typechecker=typechecker)
def get_ssp_template(template: str) -> SSPGrid:
    """Return a configured SSP template grid.

    Args:
        template (str): Template key defined in ``config["ssp"]["templates"]``.

    Returns:
        SSPGrid: Loaded stellar population grid.

    Raises:
        ValueError: If the template name or source format is not supported.

    Example:

    >>> from rubix.spectra.ssp.factory import get_ssp_template
    >>> ssp = get_ssp_template("FSPS")
    >>> ssp.age.shape
    """

    config = rubix_config["ssp"]["templates"]

    # Setup a logger based on the config
    logger = get_logger()

    # Check if the template exists in config
    if template not in config:
        raise ValueError(
            f"SSP template {template} not found in the supported " "configuration file."
        )

    if config[template]["format"].lower() == "hdf5":
        return HDF5SSPGrid.from_file(
            config[template],
            file_location=TEMPLATE_PATH,
        )
    elif config[template]["format"].lower() == "pypipe3d":
        return pyPipe3DSSPGrid.from_file(
            config[template],
            file_location=TEMPLATE_PATH,
        )
    elif config[template]["format"].lower() == "fsps":
        if config[template]["source"] == "load_from_file":
            try:
                return HDF5SSPGrid.from_file(
                    config[template], file_location=TEMPLATE_PATH
                )
            except FileNotFoundError:
                logger.warning(
                    "The FSPS SSP template file is not found. Running FSPS "
                    "to generate SSP templates."
                )
                write_fsps_data_to_disk(
                    config[template]["file_name"], file_location=TEMPLATE_PATH
                )
                return HDF5SSPGrid.from_file(
                    config[template], file_location=TEMPLATE_PATH
                )
        elif config[template]["source"] == "rerun_from_scratch":
            logger.info(
                "Running FSPS to generate SSP templates. This may take a " "while."
            )
            write_fsps_data_to_disk(
                config[template]["file_name"], file_location=TEMPLATE_PATH
            )
            return HDF5SSPGrid.from_file(
                config[template],
                file_location=TEMPLATE_PATH,
            )
        else:
            raise ValueError(
                f"The source {config[template]['source']} of the FSPS SSP "
                "template is not supported."
            )
    else:
        raise ValueError(
            "Currently only HDF5 format and fits files in the format of "
            "pyPipe3D format are supported for SSP templates."
        )
