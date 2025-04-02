from typing import Callable, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from rubix import config as rubix_config
from rubix.core.data import GasData, StarsData
from rubix.logger import get_logger
from rubix.spectra.ifu import (
    calculate_cube,
    cosmological_doppler_shift,
    resample_spectrum,
    velocity_doppler_shift,
)

from .data import RubixData
from .ssp import (
    get_lookup_interpolation,
    get_lookup_interpolation_pmap,
    get_lookup_interpolation_vmap,
    get_ssp,
)
from .telescope import get_telescope


@jaxtyped(typechecker=typechecker)
def get_calculate_spectra(config: dict) -> Callable:
    """
    The function gets the lookup function that performs the lookup to the SSP model,
    and parallelizes the funciton across all GPUs.

    Args:
        config (dict): The configuration dictionary

    Returns:
        The function that calculates the spectra of the stars.

    Example
    -------
    >>> config = {
    ...     "ssp": {
    ...         "template": {
    ...             "name": "BruzualCharlot2003"
    ...             },
    ...          },
    ...     }

    >>> from rubix.core.ifu import get_calculate_spectra
    >>> calcultae_spectra = get_calculate_spectra(config)

    >>> rubixdata = calcultae_spectra(rubixdata)
    >>> # Access the spectra of the stars
    >>> rubixdata.stars.spectra
    """
    logger = get_logger(config.get("logger", None))
    lookup_interpolation_pmap = get_lookup_interpolation_pmap(config)
    # lookup_interpolation_vmap = get_lookup_interpolation_vmap(config)
    lookup_interpolation = get_lookup_interpolation(config)

    @jaxtyped(typechecker=typechecker)
    def calculate_spectra(rubixdata: RubixData) -> RubixData:
        logger.info("Calculating IFU cube...")
        logger.debug(
            f"Input shapes: Metallicity: {len(rubixdata.stars.metallicity)}, Age: {len(rubixdata.stars.age)}"
        )
        # Ensure metallicity and age are arrays and reshape them to be at least 1-dimensional
        # age_data = jax.device_get(rubixdata.stars.age)
        age_data = rubixdata.stars.age
        # metallicity_data = jax.device_get(rubixdata.stars.metallicity)
        metallicity_data = rubixdata.stars.metallicity
        # Ensure they are not scalars or empty; convert to 1D arrays if necessary
        age = jnp.atleast_1d(age_data)
        metallicity = jnp.atleast_1d(metallicity_data)

        """
        spectra1 = lookup_interpolation(
            # rubixdata.stars.metallicity, rubixdata.stars.age
            metallicity[0][:250000],
            age[0][:250000],
        )  # * inputs["mass"]
        spectra2 = lookup_interpolation(
            # rubixdata.stars.metallicity, rubixdata.stars.age
            metallicity[0][250000:500000],
            age[0][250000:500000],
        )
        spectra3 = lookup_interpolation(
            # rubixdata.stars.metallicity, rubixdata.stars.age
            metallicity[0][500000:750000],
            age[0][500000:750000],
        )
        spectra = jnp.concatenate([spectra1, spectra2, spectra3], axis=0)
        """
        # Define the chunk size (number of particles per chunk)
        chunk_size = 100000
        total_length = metallicity[0].shape[
            0
        ]  # assuming metallicity[0] is your 1D array of particles

        # List to hold the spectra chunks
        spectra_chunks = []

        # Loop over the data in chunks
        for start in range(0, total_length, chunk_size):
            end = min(start + chunk_size, total_length)
            current_chunk = lookup_interpolation(
                metallicity[0][start:end],
                age[0][start:end],
            )
            spectra_chunks.append(current_chunk)

        # Concatenate all the chunks along axis 0
        spectra = jnp.concatenate(spectra_chunks, axis=0)
        logger.debug(f"Calculation Finished! Spectra shape: {spectra.shape}")
        spectra_jax = jnp.array(spectra)
        spectra_jax = jnp.expand_dims(spectra_jax, axis=0)
        rubixdata.stars.spectra = spectra_jax
        # setattr(rubixdata.gas, "spectra", spectra)
        # jax.debug.print("Calculate Spectra: Spectra {}", spectra)
        return rubixdata

    return calculate_spectra


@jaxtyped(typechecker=typechecker)
def get_scale_spectrum_by_mass(config: dict) -> Callable:
    """
    The spectra of the stellar particles are scaled by the mass of the stars.

    Args:
        config (dict): The configuration dictionary
    Returns:
        The function that scales the spectra by the mass of the stars.

    Example
    -------
    >>> from rubix.core.ifu import get_scale_spectrum_by_mass
    >>> scale_spectrum_by_mass = get_scale_spectrum_by_mass(config)

    >>> rubixdata = scale_spectrum_by_mass(rubixdata)
    >>> # Access the spectra of the stars, which is now scaled by the stellar mass
    >>> rubixdata.stars.spectra
    """

    logger = get_logger(config.get("logger", None))

    @jaxtyped(typechecker=typechecker)
    def scale_spectrum_by_mass(rubixdata: RubixData) -> RubixData:

        logger.info("Scaling Spectra by Mass...")
        mass = jnp.expand_dims(rubixdata.stars.mass, axis=-1)
        # rubixdata.stars.spectra = rubixdata.stars.spectra * mass
        spectra_mass = rubixdata.stars.spectra * mass
        setattr(rubixdata.stars, "spectra", spectra_mass)
        # jax.debug.print("mass mult: Spectra {}", inputs["spectra"])
        return rubixdata

    return scale_spectrum_by_mass


# Vectorize the resample_spectrum function
@jaxtyped(typechecker=typechecker)
def get_resample_spectrum_vmap(target_wavelength) -> Callable:
    """
    The spectra of the stars are resampled to the telescope wavelength grid.

    Args:
        target_wavelength (jax.Array): The telescope wavelength grid

    Returns:
        The function that resamples the spectra to the telescope wavelength grid.
    """

    @jaxtyped(typechecker=typechecker)
    def resample_spectrum_vmap(initial_spectrum, initial_wavelength):
        return resample_spectrum(
            initial_spectrum=initial_spectrum,
            initial_wavelength=initial_wavelength,
            target_wavelength=target_wavelength,
        )

    return jax.vmap(resample_spectrum_vmap, in_axes=(0, 0))


# Parallelize the vectorized function across devices
@jaxtyped(typechecker=typechecker)
def get_resample_spectrum_pmap(target_wavelength) -> Callable:
    """
    Pmap the function that resamples the spectra of the stars to the telescope wavelength grid.

    Args:
        target_wavelength (jax.Array): The telescope wavelength grid

    Returns:
        The function that resamples the spectra to the telescope wavelength grid.
    """
    vmapped_resample_spectrum = get_resample_spectrum_vmap(target_wavelength)
    return jax.pmap(vmapped_resample_spectrum)


@jaxtyped(typechecker=typechecker)
def get_velocities_doppler_shift_vmap(
    ssp_wave: Float[Array, "..."], velocity_direction: str
) -> Callable:
    """
    The function doppler shifts the wavelength based on the velocity of the stars.

    Args:
        ssp_wave (jax.Array): The wavelength of the SSP grid
        velocity_direction (str): The velocity component of the stars that is used to doppler shift the wavelength

    Returns:
        The function that doppler shifts the wavelength based on the velocity of the stars.
    """

    def func(velocity):
        return velocity_doppler_shift(
            wavelength=ssp_wave, velocity=velocity, direction=velocity_direction
        )

    return jax.vmap(func, in_axes=0)


@jaxtyped(typechecker=typechecker)
def get_doppler_shift_and_resampling(config: dict) -> Callable:
    """
    The function doppler shifts the wavelength based on the velocity of the stars and resamples the spectra to the telescope wavelength grid.

    Args:
        config (dict): The configuration dictionary

    Returns:
        The function that doppler shifts the wavelength based on the velocity of the stars and resamples the spectra to the telescope wavelength grid.

    Example
    -------
    >>> from rubix.core.ifu import get_doppler_shift_and_resampling
    >>> doppler_shift_and_resampling = get_doppler_shift_and_resampling(config)

    >>> rubixdata = doppler_shift_and_resampling(rubixdata)
    >>> # Access the spectra of the stars, which is now doppler shifted and resampled to the telescope wavelength grid
    >>> rubixdata.stars.spectra
    """
    logger = get_logger(config.get("logger", None))

    # The velocity component of the stars that is used to doppler shift the wavelength
    velocity_direction = rubix_config["ifu"]["doppler"]["velocity_direction"]

    # The redshift at which the user wants to observe the galaxy
    galaxy_redshift = config["galaxy"]["dist_z"]

    # Get the telescope wavelength bins
    telescope = get_telescope(config)
    telescope_wavelength = telescope.wave_seq

    # Get the SSP grid to doppler shift the wavelengths
    ssp = get_ssp(config)

    # Doppler shift the SSP wavelenght based on the cosmological distance of the observed galaxy
    ssp_wave = cosmological_doppler_shift(z=galaxy_redshift, wavelength=ssp.wavelength)
    logger.debug(f"SSP Wave: {ssp_wave.shape}")

    # Function to Doppler shift the wavelength based on the velocity of the stars particles
    # This binds the velocity direction, such that later we only need the velocity during the pipeline
    doppler_shift = get_velocities_doppler_shift_vmap(ssp_wave, velocity_direction)

    @jaxtyped(typechecker=typechecker)
    def doppler_shift_and_resampling(rubixdata: object) -> object:
        if rubixdata.stars.spectra is not None:
            # Doppler shift the SSP Wavelengths based on the velocity of the stars
            doppler_shifted_ssp_wave = doppler_shift(rubixdata.stars.velocity)
            logger.info("Doppler shifting and resampling stellar spectra...")
            logger.debug(f"Doppler Shifted SSP Wave: {doppler_shifted_ssp_wave.shape}")
            logger.debug(f"Telescope Wave Seq: {telescope.wave_seq.shape}")
            # Function to resample the spectrum to the telescope wavelength grid
            resample_spectrum_pmap = get_resample_spectrum_pmap(telescope_wavelength)
            # jax.debug.print("doppler shifted ssp wave {}", doppler_shifted_ssp_wave)
            # jax.debug.print("Spectra before resampling {}", inputs["spectra"])
            # if (
            #    rubixdata.stars.spectra.shape[0] == 1
            #    and rubixdata.stars.spectra.shape[1] == 1
            # ):
            #    rubixdata.stars.spectra = jnp.squeeze(rubixdata.stars.spectra, axis=0)
            # if (
            #     rubixdata.stars.spectra.shape[0] == 1
            #    and rubixdata.stars.spectra.shape[1] == 1
            # ):
            #    rubixdata.stars.spectra = jnp.squeeze(rubixdata.stars.spectra, axis=0)
            while (
                rubixdata.stars.spectra.shape[0] == 1
                and rubixdata.stars.spectra.shape[1] == 1
            ):
                rubixdata.stars.spectra = jnp.squeeze(rubixdata.stars.spectra, axis=0)
            # if doppler_shifted_ssp_wave.shape[0] == 1:
            #    doppler_shifted_ssp_wave = jnp.squeeze(doppler_shifted_ssp_wave, axis=0)
            # print(f"Shape of spectra...............: {rubixdata.stars.spectra.shape}")
            # print(
            #    f"Shape of doppler_shifted_ssp_wave..................: {doppler_shifted_ssp_wave.shape}"
            # )

            spectrum_resampled = resample_spectrum_pmap(
                rubixdata.stars.spectra, doppler_shifted_ssp_wave
            )
            # rubixdata.stars.spectra = spectrum_resampled
            setattr(rubixdata.stars, "spectra", spectrum_resampled)
            # jax.debug.print("doppler shift and resampl: Spectra {}", inputs["spectra"])

        if rubixdata.gas.spectra is not None:
            # Doppler shift the SSP Wavelengths based on the velocity of the gas particles
            doppler_shifted_ssp_wave = doppler_shift(rubixdata.gas.velocity)
            logger.info("Doppler shifting and resampling gas spectra...")
            logger.debug(f"Doppler Shifted SSP Wave: {doppler_shifted_ssp_wave.shape}")
            logger.debug(f"Telescope Wave Seq: {telescope.wave_seq.shape}")
            # Function to resample the spectrum to the telescope wavelength grid
            resample_spectrum_pmap = get_resample_spectrum_pmap(telescope_wavelenght)
            spectrum_resampled = resample_spectrum_pmap(
                rubixdata.gas.spectra, doppler_shifted_ssp_wave
            )

        return rubixdata

    return doppler_shift_and_resampling


@jaxtyped(typechecker=typechecker)
def get_calculate_datacube(config: dict) -> Callable:
    """
    The function returns the function that calculates the datacube of the stars.

    Args:
        config (dict): The configuration dictionary

    Returns:
        The function that calculates the datacube of the stars.

    Example
    -------
    >>> from rubix.core.ifu import get_calculate_datacube
    >>> calculate_datacube = get_calculate_datacube(config)

    >>> rubixdata = calculate_datacube(rubixdata)
    >>> # Access the datacube of the stars
    >>> rubixdata.stars.datacube
    """
    logger = get_logger(config.get("logger", None))
    telescope = get_telescope(config)
    num_spaxels = int(telescope.sbin)

    # Bind the num_spaxels to the function
    calculate_cube_fn = jax.tree_util.Partial(calculate_cube, num_spaxels=num_spaxels)
    calculate_cube_pmap = jax.pmap(calculate_cube_fn)

    @jaxtyped(typechecker=typechecker)
    def calculate_datacube(rubixdata: RubixData) -> RubixData:
        logger.info("Calculating Data Cube...")
        # logger.debug(f"pixel assignment: {rubixdata.stars.pixel_assignment.shape}")
        ifu_cubes = calculate_cube_pmap(
            spectra=rubixdata.stars.spectra,
            spaxel_index=rubixdata.stars.pixel_assignment,
        )
        datacube = jnp.sum(ifu_cubes, axis=0)
        logger.debug(f"Datacube Shape: {datacube.shape}")
        # logger.debug(f"This is the datacube: {datacube}")
        datacube_jax = jnp.array(datacube)
        setattr(rubixdata.stars, "datacube", datacube_jax)
        # rubixdata.stars.datacube = datacube
        return rubixdata

    return calculate_datacube
