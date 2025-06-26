from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped

from rubix import config


@jaxtyped(typechecker=typechecker)
def convert_luminoisty_to_flux(
    luminosity: Float[Array, "..."],
    observation_lum_dist: Union[Float[Array, "..."], float],
    observation_z: float,
    pixel_size: float,
    CONSTANTS=config["constants"],
) -> Float[Array, "..."]:
    """
    Convert luminosity to flux in units erg/s/cm^2/Angstrom as observed by the telescope.
    The luminosity is object specific, the flux depends on the distance to the object, the redshift, and the pixel size of the telescope.

    Args:
        luminosity (array-like): The luminosity of the object.
        observation_lum_dist (float): The luminosity distance to the object in Mpc.
        observation_z (float): The redshift of the object.
        pixel_size (float): The pixel size of the telescope in cm.
        CONSTANTS (dict, optional): A dictionary containing the constants used in the calculation. Defaults to config["constants"].

    Returns:
        The flux of the object in units erg/s/cm^2/Angstrom as observed by the telescope (array-like).
    """
    CONST = float(CONSTANTS.get("LSOL_TO_ERG")) / float(CONSTANTS.get("MPC_TO_CM")) ** 2
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
    """Convert luminosity to flux in units erg/s/cm^2/Angstrom as observed by the telescope"""
    CONST = np.float64(
        float(CONSTANTS.get("LSOL_TO_ERG")) / float(CONSTANTS.get("MPC_TO_CM")) ** 2
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
    z: float, wavelength: Float[Array, " n_bins"]
) -> Float[Array, " n_bins"]:
    """
    Calculate the cosmological Doppler shift of a wavelength.

    Args:
        z (float): The redshift.
        wavelength (array-like): The wavelength in Angstrom.

    Returns:
        The Doppler shifted wavelength in Angstrom.
    """
    # Calculate the cosmological Doppler shift of a wavelength
    return (1 + z) * wavelength


@jaxtyped(typechecker=typechecker)
def calculate_diff(
    vec: Float[Array, "..."], pad_with_zero: bool = True
) -> Float[Array, "..."]:
    """
    Calculate the difference between each element in a vector.

    Args:
        vec (array-like): The input vector.
        pad_with_zero (bool, optional): Whether to prepend the first element of the vector to the differences. Default is True.

    Returns:
        The differences between each element in the vector (array-like).
    """

    if pad_with_zero:
        differences = jnp.diff(vec, prepend=vec[0])
    else:
        differences = jnp.diff(vec)
    return differences


def _get_velocity_component_single(vec: Float[Array, "..."], direction: str) -> Float:
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
            f"{direction} is not a valid direction. Supported directions are 'x', 'y', or 'z'."
        )


def _get_velocity_component_multiple(
    vecs: Float[Array, "n_particles 3"], direction: str
) -> Float[Array, "n_particles 1"]:
    # Check that vecs has shape (n_particles, 3)
    if vecs.shape[1] != 3:
        raise ValueError(
            f"Expected vectors of shape (n_particles, 3), but got {vecs.shape}."
        )

    if direction == "x":
        return vecs[:, 0]
    elif direction == "y":
        return vecs[:, 1]
    elif direction == "z":
        return vecs[:, 2]
    else:
        raise ValueError(
            f"{direction} is not a valid direction. Supported directions are 'x', 'y', or 'z'."
        )


@jaxtyped(typechecker=typechecker)
def get_velocity_component(
    vec: Float[Array, "..."], direction: str
) -> Float[Array, "..."]:
    """
    This function returns the velocity component in a given direction.

    Args:
        vec (array-like): The velocity vector.
        direction (str): The direction in which to get the velocity component. Supported directions are 'x', 'y', or 'z'.

    Returns:
        The velocity component in the given direction (array-like).
    """
    if isinstance(vec, jax.Array) and vec.ndim == 2:
        return _get_velocity_component_multiple(vec, direction)
    elif isinstance(vec, jax.Array) and vec.ndim == 1:
        return _get_velocity_component_single(vec, direction)
    else:
        raise ValueError(
            f"Got wrong shapes. Expected vec.ndim =2 or vec.ndim=1, but got vec.ndim = {vec.ndim}"
        )


def _velocity_doppler_shift_single(
    wavelength: Float[Array, " n_bins"],
    velocity: Float[Array, "3"],
    direction="y",
    SPEED_OF_LIGHT=config["constants"]["SPEED_OF_LIGHT"],
) -> Float[Array, " n_bins"]:
    """Calculate the Doppler shift of a wavelength due to a velocity.

    Args:
        wavelengt (array-like): The wavelength in Angstrom.
        velocity (array-like): The velocity in km/s.
        direction (str, optional): The direction in which the velocity acts. Default is "y".
        SPEED_OF_LIGHT (float, optional): The speed of light in km/s. Default is config["constants"]["SPEED_OF_LIGHT"].

    Returns:
        The Doppler shifted wavelength in Angstrom (float).

    """
    velocity = get_velocity_component(velocity, direction)
    # Calculate the Doppler shift of a wavelength due to a velocity
    # print(velocity/SPEED_OF_LIGHT)
    # classic dopplershift, which is approximated 1 + v/c
    return wavelength * jnp.exp(velocity / SPEED_OF_LIGHT)
    # relativistic dopplershift
    # return wavelength * jnp.sqrt((1 + velocity / SPEED_OF_LIGHT) / (1 - velocity / SPEED_OF_LIGHT))
    # return wavelength


@jaxtyped(typechecker=typechecker)
def velocity_doppler_shift(
    wavelength: Float[Array, "..."],
    velocity: Float[Array, " * 3"],
    direction: str = config["ifu"]["doppler"]["velocity_direction"],
    SPEED_OF_LIGHT: float = config["constants"]["SPEED_OF_LIGHT"],
) -> Float[Array, "..."]:
    """
    Calculate the Doppler shift of a wavelength due to a velocity.

    Args:
        wavelength (array-like): The wavelength in Angstrom.
        velocity (array-like): The velocity in km/s.
        direction (str, optional): The direction in which the velocity acts. Default is "y".
        SPEED_OF_LIGHT (float, optional): The speed of light in km/s. Default is config["constants"]["SPEED_OF_LIGHT"].

    Returns:
        The Doppler shifted wavelength in Angstrom (array-like).
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
    initial_spectrum: Float[Array, " n_bins_initial"],
    initial_wavelength: Float[Array, " n_bins_initial"],
    target_wavelength: Float[Array, " n_bins_target"],
) -> Float[Array, " n_bins_target"]:
    """
    Resample a spectrum to the wavelength grid of a telescope.

    Args:
        initial_spectrum (array-like): The initial spectrum.
        initial_wavelength (array-like): The initial wavelength grid.
        target_wavelength (array-like): The target wavelength grid.

    Returns:
        The resampled spectrum (array-like).
    """
    # Get wavelengths inside the telescope range
    in_range_mask = (initial_wavelength >= jnp.min(target_wavelength)) & (
        initial_wavelength <= jnp.max(target_wavelength)
    )
    intrinsic_wave_diff = calculate_diff(initial_wavelength) * in_range_mask

    # Get total luminsoity within the wavelength range
    total_lum = jnp.sum(initial_spectrum * intrinsic_wave_diff)

    # Interpolate the wavelegnth to the telescope grid
    particle_lum = jnp.interp(target_wavelength, initial_wavelength, initial_spectrum)
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
    spectra: Float[Array, "n_stars n_wave_bins"],
    spaxel_index: Int[Array, " n_stars"],
    num_spaxels: int,
) -> Float[Array, "num_spaxels num_spaxels n_wave_bins"]:
    """
    Calculate the spectral data cube, which implies to sum up the spectra of all stars in each spaxel to get the spectral data cube.

    Args:
        spectra (array-like): The spectra of all stars.
        spaxel_index (array-like): The spaxel index of each star. This defines into which telescope pixel the star falls.
        num_spaxels (int): The number of spaxels in one direction of the telescope aperture. The resulting number of telescope bins is `num_spaxels^2`. Assumes that the maximum value in `spaxel_index` does not exceed this value.

    Returns:
        The spectral data cube in an array-like format with shape `(num_spaxels, num_spaxels, n_wave_bins)`.
    """
    datacube = jax.ops.segment_sum(spectra, spaxel_index, num_segments=num_spaxels**2)
    datacube = datacube.reshape(num_spaxels, num_spaxels, spectra.shape[-1])
    return datacube
