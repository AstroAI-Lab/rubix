import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from rubix.galaxy.alignment import rotate_galaxy as rotate_galaxy_core
from rubix.logger import get_logger

from .data import RubixData, update_gas_batch, update_stars_batch


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
            logger.debug("Rotation Type found: Face-on")
            alpha = 0.0
            beta = 0.0
            gamma = 0.0
        elif config["galaxy"]["rotation"]["type"] == "edge-on":
            logger.debug("Rotation Type found: edge-on")
            alpha = 90.0
            beta = 0.0
            gamma = 0.0
        else:  # matrix type
            logger.debug("Rotation Type found: matrix")
            alpha = beta = gamma = 0.0  # Will be overridden by matrix

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
        logger.info("Rotating galaxy for simulation: " + config["simulation"]["name"])

        # Handle matrix rotation if specified
        rotation_matrix = None
        if config["galaxy"]["rotation"].get("type") == "matrix":
            rot_np = jnp.load("./data/rotation_matrix.npy")
            rotation_matrix = jnp.array(rot_np)
            logger.info(f"Using rotation matrix from file: {rotation_matrix}.")

        # Always rotate stellar component (required)
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

        # Update stars using Equinox tree_at operations
        rubixdata = update_stars_batch(
            rubixdata, coords=new_coords_stars, velocity=new_velocities_stars
        )

        # Rotate gas component if present
        if (
            rubixdata.gas is not None
            and rubixdata.gas.coords is not None
            and "gas" in config["data"]["args"]["particle_type"]
        ):

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
                R=rotation_matrix,
            )

            # Update gas using Equinox tree_at operations
            rubixdata = update_gas_batch(
                rubixdata, coords=new_coords_gas, velocity=new_velocities_gas
            )

        else:
            logger.warning(
                "Gas not found in particle_type or gas data not present, "
                "only rotating stellar component."
            )

        return rubixdata

    return rotate_galaxy
