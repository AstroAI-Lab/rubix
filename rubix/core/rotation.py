from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.galaxy.alignment import rotate_galaxy as rotate_galaxy_core
from rubix.logger import get_logger

from .data import RubixData


@jaxtyped(typechecker=typechecker)
def get_galaxy_rotation(config: dict):
    """Return a rotation function configured for the provided galaxy settings.

    Args:
        config (dict): Pipeline configuration containing
            ``galaxy.rotation`` with either ``type`` (``face-on``, ``edge-on``,
            ``matrix``) or explicit ``alpha``, ``beta``, ``gamma`` angles.

    Returns:
        Callable[[RubixData], RubixData]:
            Function that applies the requested rotation.

    Raises:
        ValueError:
            When the rotation configuration or required fields are invalid
            or missing.

    Example:

            >>> config = {
            ...     ...
            ...     "galaxy": {
            ...         "dist_z": 0.1,
            ...         "rotation": {"type": "edge-on"},
            ...     },
            ...     ...
            ... }

            >>> from rubix.core.rotation import get_galaxy_rotation
            >>> rotate_galaxy = get_galaxy_rotation(config)
            >>> rubixdata = rotate_galaxy(rubixdata)
    """

    # Check if rotation information is provided under galaxy config
    if "rotation" not in config["galaxy"]:
        raise ValueError("Rotation information not provided in galaxy config")

    logger = get_logger()
    # Check if type is provided
    if "type" in config["galaxy"]["rotation"]:
        valid_rotation_types = ("face-on", "edge-on", "matrix")
        # Check if type is valid: face-on or edge-on
        if config["galaxy"]["rotation"]["type"] not in valid_rotation_types:
            raise ValueError("Invalid type provided in rotation information")

        # if type is face on, alpha = beta = gamma = 0
        # if type is edge on, alpha = 90, beta = gamma = 0
        if config["galaxy"]["rotation"]["type"] == "face-on":
            logger.debug("Rotation Type found: Face-on")
            alpha = 0.0
            beta = 0.0
            gamma = 0.0

        else:
            # type is edge-on
            logger.debug("Rotation Type found: edge-on")
            alpha = 90.0
            beta = 0.0
            gamma = 0.0

    else:
        # If type is not provided, then alpha, beta, gamma should be set
        # Check if alpha, beta, gamma are provided
        for key in ["alpha", "beta", "gamma"]:
            if key not in config["galaxy"]["rotation"]:
                raise ValueError(f"{key} not provided in rotation information")

        # Get the rotation angles from the user config
        alpha = config["galaxy"]["rotation"]["alpha"]
        beta = config["galaxy"]["rotation"]["beta"]
        gamma = config["galaxy"]["rotation"]["gamma"]

    @jaxtyped(typechecker=typechecker)
    def rotate_galaxy(rubixdata: RubixData) -> RubixData:
        """Rotate the galaxy particle data based on the specified angles.

        Args:
            rubixdata (RubixData): The RubixData object containing
                particle data.

        Returns:
            RubixData: The rotated RubixData object.
        """
        logger.info(f"Rotating galaxy with alpha={alpha}, beta={beta}, gamma={gamma}.")
        logger.info(f"Rotating galaxy for simulation: {config['simulation']['name']}.")
        # Rotate gas
        if "gas" in config["data"]["args"]["particle_type"]:
            logger.info("Rotating gas")

            # Rotate the gas component
            new_coords_gas, new_velocities_gas = rotate_galaxy_core(
                positions=rubixdata.gas.coords,
                velocities=rubixdata.gas.velocity,
                positions_stars=rubixdata.stars.coords,
                masses_stars=rubixdata.stars.mass,
                halfmass_radius=rubixdata.galaxy.halfmassrad_stars,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                key=config["simulation"]["name"],
            )

            setattr(rubixdata.gas, "coords", new_coords_gas)
            setattr(rubixdata.gas, "velocity", new_velocities_gas)

            # Rotate the stellar component
            new_coords_stars, new_velocities_stars = rotate_galaxy_core(
                positions=rubixdata.stars.coords,
                velocities=rubixdata.stars.velocity,
                positions_stars=rubixdata.stars.coords,
                masses_stars=rubixdata.stars.mass,
                halfmass_radius=rubixdata.galaxy.halfmassrad_stars,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                key=config["simulation"]["name"],
            )

            setattr(rubixdata.stars, "coords", new_coords_stars)
            setattr(rubixdata.stars, "velocity", new_velocities_stars)

        else:
            logger.warning(
                "Gas not found in particle_type, only rotating stellar component."
            )
            # Rotate the stellar component
            new_coords_stars, new_velocities_stars = rotate_galaxy_core(
                positions=rubixdata.stars.coords,
                velocities=rubixdata.stars.velocity,
                positions_stars=rubixdata.stars.coords,
                masses_stars=rubixdata.stars.mass,
                halfmass_radius=rubixdata.galaxy.halfmassrad_stars,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                key=config["simulation"]["name"],
            )

            setattr(rubixdata.stars, "coords", new_coords_stars)
            setattr(rubixdata.stars, "velocity", new_velocities_stars)

        return rubixdata

    return rotate_galaxy
