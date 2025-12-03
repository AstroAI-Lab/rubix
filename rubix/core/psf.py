from beartype import beartype as typechecker
from beartype.typing import Callable
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.telescope.psf.psf import apply_psf, get_psf_kernel

from .data import RubixData


# TODO: add option to disable PSF convolution
@jaxtyped(typechecker=typechecker)
def get_convolve_psf(config: dict) -> Callable:
    """Return a callable that applies the configured PSF kernel.

    Args:
        config (dict): Pipeline configuration that must include ``telescope``
            settings. The ``telescope.psf`` block requires ``name`` and, when
            using the Gaussian kernel, ``size`` and ``sigma`` fields to define
            the kernel dimensions and width.

    Returns:
        Callable[[RubixData], RubixData]: Callable that convolves the stars
            datacube with the generated PSF kernel.

    Raises:
        ValueError: When the PSF settings are missing or reference an unknown
            kernel type.

    Example:

            >>> config = {
            ...     ...
            ...     "telescope": {
            ...         "name": "MUSE",
            ...         "psf": {"name": "gaussian", "size": 5, "sigma": 0.6},
            ...         "lsf": {"sigma": 0.5},
            ...         "noise": {
            ...             "signal_to_noise": 1,
            ...             "noise_distribution": "normal",
            ...         },
            ...    },
            ...     ...
            ... }

            >>> from rubix.core.psf import get_convolve_psf
            >>> convolve_psf = get_convolve_psf(config)
            >>> rubixdata = convolve_psf(rubixdata)
    """

    logger = get_logger(config.get("logger", None))

    # Check if key exists in config file
    if "psf" not in config["telescope"]:
        raise ValueError("PSF configuration not found in telescope configuration")
    if "name" not in config["telescope"]["psf"]:
        raise ValueError("PSF name not found in telescope configuration")

    # Get the PSF kernel based on the configuration
    if config["telescope"]["psf"]["name"] == "gaussian":
        # Check if the PSF size and sigma are defined
        if "size" not in config["telescope"]["psf"]:
            raise ValueError("PSF size not found in telescope configuration")
        if "sigma" not in config["telescope"]["psf"]:
            raise ValueError("PSF sigma not found in telescope configuration")

        size = config["telescope"]["psf"]["size"]
        m, n = size, size
        sigma = config["telescope"]["psf"]["sigma"]
        psf_kernel = get_psf_kernel("gaussian", m, n, sigma=sigma)

    else:
        raise ValueError(
            f"Unknown PSF kernel name: {config['telescope']['psf']['name']}"
        )

    # Define the function to convolve the datacube with the PSF kernel
    def convolve_psf(rubixdata: RubixData) -> RubixData:
        """Convolve the stars datacube with the predetermined PSF kernel.

        Args:
            rubixdata (RubixData): Dataset whose ``stars.datacube`` field is
                convolved in-place.

        Returns:
            RubixData: The same dataset with ``stars.datacube`` replaced by the
                convolved result.
        """
        logger.info("Convolving with PSF...")
        datacube = rubixdata.stars.datacube
        rubixdata.stars.datacube = apply_psf(datacube, psf_kernel)
        return rubixdata

    return convolve_psf
