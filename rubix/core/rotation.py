from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.galaxy.alignment import rotate_galaxy as rotate_galaxy_core
from rubix.logger import get_logger

from .data import RubixData


@jaxtyped(typechecker=typechecker)
def get_galaxy_rotation(config: dict):
    """
    Get the function to rotate the galaxy based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        The function to rotate the galaxy.

    Example
    --------
    >>> config = {
    ...     ...
    ...     "galaxy":
    ...         {"dist_z": 0.1,
    ...         "rotation": {"type": "edge-on"},
    ...         },
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
        # Check if type is valid: face-on or edge-on
        if config["galaxy"]["rotation"]["type"] not in ["face-on", "edge-on"]:
            raise ValueError("Invalid type provided in rotation information")

        # if type is face on, alpha = beta = gamma = 0
        # if type is edge on, alpha = 90, beta = gamma = 0
        if config["galaxy"]["rotation"]["type"] == "face-on":
            logger.debug("Roataion Type found: Face-on")
            alpha = 0.0
            beta = 0.0
            gamma = 0.0

        else:
            # type is edge-on
            logger.debug("Roataion Type found: edge-on")
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
        logger.info(f"Rotating galaxy with alpha={alpha}, beta={beta}, gamma={gamma}")

        """
        for particle_type in ["stars", "gas"]:
            if particle_type in config["data"]["args"]["particle_type"]:
                # Get the component (either stars or gas)
                logger.info(f"Rotating {particle_type}")
                component = getattr(rubixdata, particle_type)

                # Get the inputs
                coords = component.coords
                velocities = component.velocity
                masses = component.mass
                halfmass_radius = rubixdata.galaxy.halfmassrad_stars

                assert (
                    coords is not None
                ), f"Coordinates not found for {particle_type}. "
                assert (
                    velocities is not None
                ), f"Velocities not found for {particle_type}. "
                assert masses is not None, f"Masses not found for {particle_type}. "

                # Rotate the galaxy
                coords, velocities = rotate_galaxy_core(
                    positions=coords,
                    velocities=velocities,
                    positions_stars=rubixdata.stars.coords,
                    masses_stars=rubixdata.stars.mass,
                    halfmass_radius=halfmass_radius,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )

                # Update the inputs
                # rubixdata.stars.coords = coords
                # rubixdata.stars.velocity = velocities
                setattr(component, "coords", coords)
                setattr(component, "velocity", velocities)

        return rubixdata
        """
        logger.info("Rotating galaxy for simulation: " + config["simulation"]["name"])
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
