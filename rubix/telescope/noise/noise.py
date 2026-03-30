from typing import Optional
import jax

import jax.numpy as jnp
from jax import random as jrandom
from jaxtyping import Array, Float
from rubix.spectra.ifu import convert_luminoisty_to_flux
from rubix.cosmology import PLANCK15

SUPPORTED_NOISE_DISTRIBUTIONS = ["normal", "uniform"]


def sample_noise(
    shape: tuple, type: str = "normal", key: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Sample noise from a normal or uniform distribution.

    Args:
        shape (tuple): The shape of the noise array.
        type (str, optional): The type of distribution to sample from. Can be either "normal" or "uniform". Defaults to "normal".
        key (Optional[jnp.ndarray], optional): The random key to use for
            sampling. Defaults to ``None``.

    Returns:
        jnp.ndarray: The sampled noise.

    Raises:
        ValueError: If an unsupported noise distribution is requested.
    """
    if key is None:
        key = jrandom.PRNGKey(0)
    if type == "normal":
        return jrandom.normal(key, shape)
    elif type == "uniform":
        return jrandom.uniform(key, shape)
    else:
        raise ValueError(
            f"Invalid noise type: {type}. Supported types: {SUPPORTED_NOISE_DISTRIBUTIONS}"
        )

def calculate_target_min_luminosity(magnitude):
    luminosity = 10 ** (-0.4 * (magnitude - 4.83)) #in solar luminosity

    return luminosity


def calculate_S2N(
    datacube: Float[Array, "n_x n_y n_wave_bins"], observation_signal_to_noise: float, s2n_magnitude: float
) -> Float[Array, "n_y n_y"]:
    """Calculate the signal-to-noise ratio array from a data cube.

    Adapted from: https://github.com/kateharborne/SimSpin/blob/4e8f0af0ebc0e43cc31729978deb3a554e039f6b/R/build_datacube.R#L386
    which implements equation 4 from Nanni et al. 2022.

    Args:
        datacube (Float[Array, "n_x n_y n_wave_bins"]): The data cube with dimensions (n_x, n_y, n_wave_bins).
        observation_signal_to_noise (float): The signal-to-noise ratio of the observation.

    Returns:
        Float[Array, "n_x n_y"]: The signal-to-noise ratio array.
    """
    # Sum up the spectra along the wavelength bins to get the flux image
    flux_image = jnp.sum(datacube, axis=-1)
    #jax.debug.print("Calculated flux image with shape: {x} and values: {y}", x=flux_image.shape, y=flux_image)

    # Mask out regions where the flux is zero
    nonzero_mask = flux_image > 0

    # Calculate the median flux value where the flux is non-zero
    #median_flux = jnp.median(jnp.where(nonzero_mask, flux_image, jnp.nan))
    median_flux = jnp.nanmedian(jnp.where(nonzero_mask, flux_image, jnp.nan))
    min_luminosity = calculate_target_min_luminosity(s2n_magnitude)
    #jax.debug.print("Calculated median flux: {x}", x=median_flux)
    #jax.debug.print("sqrt of median flux: {x}", x=jnp.sqrt(median_flux))
    #median_flux = jnp.nan_to_num(median_flux, nan=0.0)
    #jax.debug.print("s2n config: {x}", x=observation_signal_to_noise)
    # Calculate the noise factor
    noise_factor = jnp.sqrt(min_luminosity) / observation_signal_to_noise
    #jax.debug.print("Calculated noise factor: {x}", x=noise_factor)

    # Calculate the signal-to-noise ratio for each pixel
    S2N = noise_factor / jnp.sqrt(flux_image)

    # Apply the mask to set S2N to zero where the flux is zero
    S2N = jnp.where(nonzero_mask, S2N, 0)
    jax.debug.print("Calculated S/N array with shape: {x} and values: {y}", x=S2N.shape, y=S2N)

    return S2N


def calculate_noise_cube(
    cube: Float[Array, "n_x n_y n_wave_bins"],
    signal_to_noise: float,
    s2n_magnitude: float,
    noise_distribution: str = "normal",
) -> Float[Array, "n_x n_y n_wave_bins"]:
    """Calculate the noise cube given the cube and the signal-to-noise ratio.

    Adapted from: https://github.com/kateharborne/SimSpin/blob/4e8f0af0ebc0e43cc31729978deb3a554e039f6b/R/utilities.R#L587

    Args:
        cube (Float[Array, "n_x n_y n_wave_bins"]): The data cube.
        signal_to_noise (float): The signal-to-noise ratio of the observation.
        noise_distribution (str, optional): The type of distribution to sample from. Can be either "normal" or "uniform". Defaults to "normal".

    Returns:
        Float[Array, "n_x n_y n_wave_bins"]: The noise cube.
    """
    key = jrandom.PRNGKey(0)
    # S2N = jnp.where(
    #     jnp.isinf(S2N), 0, S2N
    # )  # removing infinite noise where particles per pixel = 0
    S2N = calculate_S2N(cube, signal_to_noise, s2n_magnitude)

    # Generate noise for each element in the cube based on the S/N
    noise = sample_noise(cube.shape, type=noise_distribution, key=key) * S2N[:, :, None]

    # Scale the noise by the cube to get S/N
    noise = cube * noise

    return noise
