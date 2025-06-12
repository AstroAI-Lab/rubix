import jax.numpy as jnp
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
        if config["galaxy"]["rotation"]["type"] not in ["face-on", "edge-on", "matrix"]:
            raise ValueError("Invalid type provided in rotation information")

        # if type is face on, alpha = beta = gamma = 0
        # if type is edge on, alpha = 90, beta = gamma = 0
        if config["galaxy"]["rotation"]["type"] == "face-on":
            logger.debug("Roataion Type found: Face-on")
            alpha = 0.0
            beta = 0.0
            gamma = 0.0

        elif config["galaxy"]["rotation"]["type"] == "edge-on":
            # type is edge-on
            logger.debug("Roataion Type found: edge-on")
            alpha = 90.0
            beta = 0.0
            gamma = 0.0
    
        elif config["galaxy"]["rotation"]["type"] == "matrix":
            logger.debug("Roataion Type found: matrix")
            # If type is matrix, then rotation matrix should be provided
            alpha = 0.0
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

        for particle_type in ["stars", "gas"]:
            if particle_type in config["data"]["args"]["particle_type"]:
                # Get the component (either stars or gas)
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

                if config["galaxy"]["rotation"]["type"] == "matrix":
                    logger.debug(
                        "Rotation type is matrix, loading rotation matrix from file."
                    )
                    rot_np = jnp.load("./data/rotation_matrix.npy")
                    rot_jax = jnp.array(rot_np)
                    logger.info(f"Using rotation matrix from file: {rot_jax}.")
                    rotation_matrix = rot_jax
                else:
                    logger.debug(
                        "No rotation matrix provided, using Euler angles for rotation."
                    )
                    rotation_matrix = None

                # Rotate the galaxy
                coords, velocities = rotate_galaxy_core(
                    positions=coords,
                    velocities=velocities,
                    masses=masses,
                    halfmass_radius=halfmass_radius,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    R=rotation_matrix,
                )

                # Update the inputs
                # rubixdata.stars.coords = coords
                # rubixdata.stars.velocity = velocities
                setattr(component, "coords", coords)
                setattr(component, "velocity", velocities)

        return rubixdata

    return rotate_galaxy
