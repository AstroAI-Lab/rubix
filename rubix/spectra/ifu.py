from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped

from rubix import config

N_BINS_AXIS = "n_bins"
N_BINS_INITIAL_AXIS = "n_bins_initial"
N_BINS_TARGET_AXIS = "n_bins_target"
N_PARTICLE_AXIS = "n_particles"
VELOCITY_AXIS = "3"
PARTICLE_MATRIX_AXES = "n_particles 3"
ELLIPSIS_THREE_AXES = "... 3"
STAR_WAVE_AXES = "n_stars n_wave_bins"
SPAXEL_INDEX_AXIS = "n_stars"
SPAXEL_CUBE_AXES = "num_spaxels_x num_spaxels_y n_wave_bins"


@jaxtyped(typechecker=typechecker)
def convert_luminoisty_to_flux(
    luminosity: Float[Array, "..."],
    observation_lum_dist: Union[Float[Array, "..."], float],
    observation_z: float,
    pixel_size: float,
    CONSTANTS: dict = config["constants"],
) -> Float[Array, "..."]:
    """
    Convert luminosity to flux in units erg/s/cm^2/Angstrom as observed by
    the telescope.
    The luminosity is object specific, the flux depends on the distance to the
    object, the redshift, and the pixel size of the telescope.

    Args:
        luminosity (Float[Array, "..."]): Intrinsic luminosity per bin.
        observation_lum_dist (Union[Float[Array, "..."], float]): Luminosity
            distance in Mpc.
        observation_z (float): Object redshift.
        pixel_size (float): Telescope pixel size in cm.
        CONSTANTS (dict, optional): Conversion constants. Defaults to
            ``config["constants"]``.

    Returns:
        Float[Array, "..."]: Flux in erg/s/cm^2/Å.
    """
    CONST = float(CONSTANTS.get("LSOL_TO_ERG")) / (
        float(CONSTANTS.get("MPC_TO_CM")) ** 2
    )
    FACTOR = (
        CONST
        / (4 * jnp.pi * observation_lum_dist**2)
        / (1 + observation_z)
        / pixel_size
    )
    spectral_dist = luminosity * FACTOR
    return spectral_dist


@jaxtyped(typechecker=typechecker)
def convert_luminoisty_to_flux_factor(
    observation_lum_dist,
    observation_z,
    pixel_size,
    CONSTANTS=config["constants"],
):
    """Convert luminosity to flux in units erg/s/cm^2/Å."""
    CONST = np.float64(
        float(CONSTANTS.get("LSOL_TO_ERG")) / (float(CONSTANTS.get("MPC_TO_CM")) ** 2)
    )
    FACTOR = (
        CONST
        / (4 * np.pi * np.float64(observation_lum_dist) ** 2)
        / (1 + np.float64(observation_z))
        / np.float64(pixel_size)
    )
    FACTOR = jnp.float64(FACTOR)
    return FACTOR


def cosmological_doppler_shift(
    z: float,
    wavelength: Float[Array, N_BINS_AXIS],
) -> Float[Array, N_BINS_AXIS]:
    """Apply the cosmological Doppler shift to a wavelength grid.

    Args:
        z (float): Object redshift.
        wavelength (Float[Array, N_BINS_AXIS]): Wavelengths in Å.

    Returns:
        Float[Array, N_BINS_AXIS]: Doppler-shifted wavelengths in Å.
    """
    # Calculate the cosmological Doppler shift of a wavelength
    return (1 + z) * wavelength


@jaxtyped(typechecker=typechecker)
def calculate_diff(
    vec: Float[Array, "..."], pad_with_zero: bool = True
) -> Float[Array, "..."]:
    """Calculate consecutive differences along a vector.

    Args:
        vec (Float[Array, "..."]): Input grid.
        pad_with_zero (bool, optional): If ``True`` prepend the first element
            so the output matches the input length. Defaults to ``True``.

    Returns:
        Float[Array, "..."]: Finite differences of ``vec``.
    """

    if pad_with_zero:
        differences = jnp.diff(vec, prepend=vec[0])
    else:
        differences = jnp.diff(vec)
    return differences


def _get_velocity_component_single(
    vec: Float[Array, "..."],
    direction: str,
) -> Float:
    # Check that vec is of size 3
    if not vec.size == 3:
        raise ValueError(f"Expected vector of size 3, but got {vec.size}.")

    if direction == "x":
        return vec[0]
    elif direction == "y":
        return vec[1]
    elif direction == "z":
        return vec[2]

    else:
        raise ValueError(
            f"{direction} is not a valid direction. Supported directions are "
            f"'x', 'y', or 'z'."
        )


def _get_velocity_component_multiple(
    vecs: Float[Array, PARTICLE_MATRIX_AXES],
    direction: str,
) -> Float[Array, N_PARTICLE_AXIS]:
    # Check that vecs has shape (n_particles, 3)
    if vecs.shape[1] != 3:
        raise ValueError(
            f"Expected vectors of shape (n_particles, 3), but got " f"{vecs.shape}."
        )

    if direction == "x":
        return vecs[:, 0]
    elif direction == "y":
        return vecs[:, 1]
    elif direction == "z":
        return vecs[:, 2]
    else:
        raise ValueError(
            f"{direction} is not a valid direction. Supported directions are "
            f"'x', 'y', or 'z'."
        )


@jaxtyped(typechecker=typechecker)
def get_velocity_component(
    vec: Float[Array, "..."], direction: str
) -> Float[Array, "..."]:
    """
    This function returns the velocity component in a given direction.

    Args:
        vec (Float[Array, "..."]): The velocity vector.
        direction (str): The direction in which to get the velocity component.
            Supported directions are 'x', 'y', or 'z'.

    Returns:
        Float[Array, "..."]: Component extracted from ``vec``.

    Raises:
        ValueError: If ``vec`` does not have 1 or 2 dimensions or the
            direction is invalid.
    """
    if isinstance(vec, jax.Array) and vec.ndim == 2:
        return _get_velocity_component_multiple(vec, direction)
    elif isinstance(vec, jax.Array) and vec.ndim == 1:
        return _get_velocity_component_single(vec, direction)
    else:
        raise ValueError(
            f"Got wrong shapes. Expected vec.ndim =2 or vec.ndim=1, but got "
            f"vec.ndim = {vec.ndim}"
        )


def _velocity_doppler_shift_single(
    wavelength: Float[Array, N_BINS_AXIS],
    velocity: Float[Array, VELOCITY_AXIS],
    direction: str = "y",
    SPEED_OF_LIGHT: float = config["constants"]["SPEED_OF_LIGHT"],
) -> Float[Array, N_BINS_AXIS]:
    """Apply a velocity-induced Doppler shift for a single vector.

    Args:
        wavelength (Float[Array, N_BINS_AXIS]): Rest wavelengths in Å.
        velocity (Float[Array, VELOCITY_AXIS]): Velocity components in km/s.
        direction (str, optional): Component axis. Defaults to ``"y"``.
        SPEED_OF_LIGHT (float, optional): Speed of light in km/s. Defaults to
            ``config["constants"]["SPEED_OF_LIGHT"]``.

    Returns:
        Float[Array, N_BINS_AXIS]: Doppler shifted wavelengths in Å.
    """
    velocity = get_velocity_component(velocity, direction)
    # Calculate the Doppler shift of a wavelength due to a velocity
    # print(velocity/SPEED_OF_LIGHT)
    # classic dopplershift, which is approximated 1 + v/c
    return wavelength * jnp.exp(velocity / SPEED_OF_LIGHT)
    # relativistic dopplershift
    # return wavelength * jnp.sqrt(
    #     (1 + velocity / SPEED_OF_LIGHT)
    #     / (1 - velocity / SPEED_OF_LIGHT)
    # )
    # return wavelength


@jaxtyped(typechecker=typechecker)
def velocity_doppler_shift(
    wavelength: Float[Array, "..."],
    velocity: Float[Array, ELLIPSIS_THREE_AXES],
    direction: str = config["ifu"]["doppler"]["velocity_direction"],
    SPEED_OF_LIGHT: float = config["constants"]["SPEED_OF_LIGHT"],
) -> Float[Array, "..."]:
    """Vectorized Doppler shift over multiple velocity vectors.

    Args:
        wavelength (Float[Array, "..."]): Rest wavelengths in Å.
        velocity (Float[Array, ELLIPSIS_THREE_AXES]): Velocity components per
            sample.
        direction (str, optional): Axis to project onto. Defaults to
            ``config["ifu"]["doppler"]["velocity_direction"]``.
        SPEED_OF_LIGHT (float, optional): Speed of light in km/s. Defaults to
            ``config["constants"]["SPEED_OF_LIGHT"]``.

    Returns:
        Float[Array, "..."]: Doppler shifted wavelengths per velocity entry.
    """
    while velocity.shape[0] == 1:
        velocity = jnp.squeeze(velocity, axis=0)
    # if velocity.shape[0] == 1:
    #    velocity = jnp.squeeze(velocity, axis=0)
    # Vmap the function to handle multiple velocities with the same wavelength
    return jax.vmap(
        lambda v: _velocity_doppler_shift_single(
            wavelength, v, direction, SPEED_OF_LIGHT
        )
    )(velocity)


@jaxtyped(typechecker=typechecker)
def resample_spectrum(
    initial_spectrum: Float[Array, N_BINS_INITIAL_AXIS],
    initial_wavelength: Float[Array, N_BINS_INITIAL_AXIS],
    target_wavelength: Float[Array, N_BINS_TARGET_AXIS],
) -> Float[Array, N_BINS_TARGET_AXIS]:
    """Resample a spectrum onto a target wavelength grid.

    Args:
        initial_spectrum (Float[Array, N_BINS_INITIAL_AXIS]): Input spectrum.
        initial_wavelength (Float[Array, N_BINS_INITIAL_AXIS]): Input grid in
            Å.
        target_wavelength (Float[Array, N_BINS_TARGET_AXIS]): Target grid in Å.

    Returns:
        Float[Array, N_BINS_TARGET_AXIS]: Flux conserved on the new grid.
    """
    # Get wavelengths inside the telescope range
    in_range_mask = (initial_wavelength >= jnp.min(target_wavelength)) & (
        initial_wavelength <= jnp.max(target_wavelength)
    )
    intrinsic_wave_diff = calculate_diff(initial_wavelength) * in_range_mask

    # Get total luminsoity within the wavelength range
    total_lum = jnp.sum(initial_spectrum * intrinsic_wave_diff)

    # Interpolate the wavelegnth to the telescope grid
    particle_lum = jnp.interp(
        target_wavelength,
        initial_wavelength,
        initial_spectrum,
    )
    # New total luminosity
    new_total_lum = jnp.sum(particle_lum * calculate_diff(target_wavelength))

    # Factor to conserve flux in the new spectrum

    scale_factor = total_lum / new_total_lum
    scale_factor = jnp.nan_to_num(
        scale_factor, nan=0.0
    )  # Otherwise we get NaNs if new_total_lum is zero
    lum = particle_lum * scale_factor
    # jax.debug.print("total_lum: {}", total_lum)
    # jax.debug.print("new_total_lum: {}", new_total_lum)
    # jax.debug.print("scale_factor: {}", scale_factor)
    # jax.debug.print("resampled spectrum: {}", lum)
    # jax.debug.print("intrinsic_wave_diff: {}", intrinsic_wave_diff)
    return lum


@jaxtyped(typechecker=typechecker)
def calculate_cube(
    spectra: Float[Array, STAR_WAVE_AXES],
    spaxel_index: Int[Array, SPAXEL_INDEX_AXIS],
    num_spaxels: int,
) -> Float[Array, SPAXEL_CUBE_AXES]:
    """Aggregate stellar spectra into a spatial data cube.

    Args:
        spectra (Float[Array, STAR_WAVE_AXES]): Individual spectra.
        spaxel_index (Int[Array, SPAXEL_INDEX_AXIS]): Flat spaxel indices per
            star.
        num_spaxels (int): Number of spaxels per axis.

    Returns:
        Float[Array, SPAXEL_CUBE_AXES]: Summed cube.
    """
    datacube = jax.ops.segment_sum(
        spectra,
        spaxel_index,
        num_segments=num_spaxels**2,
    )
    datacube = datacube.reshape(num_spaxels, num_spaxels, spectra.shape[-1])
    return datacube
