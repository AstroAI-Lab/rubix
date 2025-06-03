from typing import Callable, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax
from jaxtyping import Array, Float, jaxtyped

from rubix import config as rubix_config
from rubix.core.data import GasData, StarsData
from rubix.logger import get_logger
from rubix.spectra.ifu import (
    _velocity_doppler_shift_single,
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
    # lookup_interpolation_pmap = get_lookup_interpolation_pmap(config)
    # lookup_interpolation_vmap = get_lookup_interpolation_vmap(config)
    lookup_interpolation = get_lookup_interpolation(config)

    def lookup_interpolation_laxmap(age_metallicity):
        age, metallicity = age_metallicity
        return lookup_interpolation(metallicity, age)

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

        # Define the chunk size (number of particles per chunk)
        # chunk_size = 250000
        # total_length = metallicity.shape[
        #    0
        # ]  # assuming metallicity[0] is your 1D array of particles

        # List to hold the spectra chunks
        # spectra_chunks = []

        # Loop over the data in chunks
        # for start in range(0, total_length, chunk_size):
        #    end = min(start + chunk_size, total_length)
        #    current_chunk = lookup_interpolation(
        #        metallicity[start:end],
        #        age[start:end],
        #    )
        #    spectra_chunks.append(current_chunk)

        # Concatenate all the chunks along axis 0
        # spectra = jnp.concatenate(spectra_chunks, axis=0)
        # Single, batched lookup over all stars:
        spectra = lookup_interpolation(
            metallicity,
            age,
        )
        # spectra = jax.lax.map(
        #    lookup_interpolation_laxmap,
        #    (metallicity, age),
        #    batch_size=2,
        # )
        logger.debug(f"Calculation Finished! Spectra shape: {spectra.shape}")
        spectra_jax = jnp.array(spectra)
        # spectra_jax = jnp.expand_dims(spectra_jax, axis=0)
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
# @jaxtyped(typechecker=typechecker)
# def get_resample_spectrum_pmap(target_wavelength) -> Callable:
#    """
#    Pmap the function that resamples the spectra of the stars to the telescope wavelength grid.

#    Args:
#        target_wavelength (jax.Array): The telescope wavelength grid

#    Returns:
#        The function that resamples the spectra to the telescope wavelength grid.
#    """
#    vmapped_resample_spectrum = get_resample_spectrum_vmap(target_wavelength)
#    return jax.pmap(vmapped_resample_spectrum)


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

    # def func(velocity):
    #    return velocity_doppler_shift(
    #        wavelength=ssp_wave, velocity=velocity, direction=velocity_direction
    #    )

    # return jax.vmap(func, in_axes=0)
    def doppler_fn(velocities):
        return velocity_doppler_shift(
            wavelength=ssp_wave,
            velocity=velocities,
            direction=velocity_direction,
        )

    return doppler_fn


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
    def process_particle(
        particle: Union[StarsData, GasData],
    ) -> Union[Float[Array, "..."], None]:
        if particle.spectra is not None:
            # Doppler shift based on the velocity of the particle
            doppler_shifted_ssp_wave = doppler_shift(particle.velocity)
            logger.info(f"Doppler shifting and resampling spectra...")
            logger.debug(f"Doppler Shifted SSP Wave: {doppler_shifted_ssp_wave.shape}")
            logger.debug(f"Telescope Wave Seq: {telescope_wavelength.shape}")

            # Function to resample the spectrum to the telescope wavelength grid
            # resample_spectrum_pmap = get_resample_spectrum_pmap(telescope_wavelength)
            # spectrum_resampled = resample_spectrum_pmap(
            #    particle.spectra, doppler_shifted_ssp_wave
            # )
            resample_fn = get_resample_spectrum_vmap(telescope_wavelength)
            spectrum_resampled = resample_fn(particle.spectra, doppler_shifted_ssp_wave)
            return spectrum_resampled
        return particle.spectra

    @jaxtyped(typechecker=typechecker)
    def doppler_shift_and_resampling(rubixdata: RubixData) -> RubixData:
        for particle_name in ["stars", "gas"]:
            particle = getattr(rubixdata, particle_name)
            particle.spectra = process_particle(particle)

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
    # calculate_cube_fn = jax.tree_util.Partial(calculate_cube, num_spaxels=num_spaxels)
    # calculate_cube_pmap = jax.pmap(calculate_cube_fn)

    @jaxtyped(typechecker=typechecker)
    def calculate_datacube(rubixdata: RubixData) -> RubixData:
        logger.info("Calculating Data Cube...")
        # ifu_cubes = calculate_cube_fn(
        #    spectra=rubixdata.stars.spectra,
        #    spaxel_index=rubixdata.stars.pixel_assignment,
        # )
        datacube = calculate_cube(
            rubixdata.stars.spectra, rubixdata.stars.pixel_assignment, num_spaxels
        )
        # datacube = jnp.sum(ifu_cubes, axis=0)
        logger.debug(f"Datacube Shape: {datacube.shape}")
        # logger.debug(f"This is the datacube: {datacube}")
        datacube_jax = jnp.array(datacube)
        setattr(rubixdata.stars, "datacube", datacube_jax)
        # rubixdata.stars.datacube = datacube
        return rubixdata

    return calculate_datacube


@jaxtyped(typechecker=typechecker)
def get_particle_spectrum(config: dict) -> Callable:
    """
    Returns a function which, for a *single* star with inputs
      (age, metallicity, mass, velocity)
    will do:
      1) SSP lookup
      2) scale by mass
      3) Doppler‐shift the SSP wavelengths
      4) resample onto the telescope grid
    and return the final 1D spectrum.
    """
    # 1) the SSP lookup (metallicity, age) -> spectrum_on_ssp_grid
    lookup_ssp = get_lookup_interpolation(config)

    # 2) prepare Doppler + resampling
    velocity_direction = rubix_config["ifu"]["doppler"]["velocity_direction"]
    z_obs = config["galaxy"]["dist_z"]

    # get telescope grid
    telescope = get_telescope(config)
    target_wavelength = telescope.wave_seq  # shape (n_wave_tel,)

    # get the SSP wavelengths for cosmological redshift
    ssp_model = get_ssp(config)
    ssp_wave0 = cosmological_doppler_shift(
        z=z_obs, wavelength=ssp_model.wavelength
    )  # shape (n_wave_ssp,)

    @jaxtyped(typechecker=typechecker)
    def particle_spectrum(
        age: Float[Array, ""],
        metallicity: Float[Array, ""],
        mass: Float[Array, ""],
        velocity: Float[Array, ""],
    ) -> Float[Array, "n_wave_tel"]:
        # --- 1) SSP lookup
        spec_ssp = lookup_ssp(metallicity, age)  # (n_wave_ssp,)

        # --- 2) mass scale
        spec_mass = spec_ssp * mass  # (n_wave_ssp,)

        # --- 3) Doppler‐shift the SSP wavelengths
        shifted_wave = velocity_doppler_shift(
            wavelength=ssp_wave0,
            velocity=velocity,
            direction=velocity_direction,
        )  # (n_wave_ssp,)

        # --- 4) resample onto telescope grid
        spec_tel = resample_spectrum(
            initial_spectrum=spec_mass,
            initial_wavelength=shifted_wave,
            target_wavelength=target_wavelength,
        )  # (n_wave_tel,)

        return spec_tel

    return particle_spectrum


@jaxtyped(typechecker=typechecker)
def get_calculate_datacube_laxscan(config: dict) -> Callable:
    """
    The function returns the function that calculates the datacube of the stars.
    It takes RubixData as input. It calculates the spectrum for one stellar particle,
    weights it by mass, doppler shifts it, resamples it to the telescope wavelength grid,
    and finally adds the spectrum at the right position in the datacube.

    This is done for every stellar particle in the RubixData object.
    This is done by using a JAX lax.scan, which is a more efficient way to do this than a for loop.

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
    num_segments = num_spaxels**2
    wave_grid = telescope.wave_seq

    # Bind the num_spaxels to the function
    # calculate_cube_fn = jax.tree_util.Partial(calculate_cube, num_spaxels=num_spaxels)
    # calculate_cube_pmap = jax.pmap(calculate_cube_fn)

    @jaxtyped(typechecker=typechecker)
    def calculate_datacube(rubixdata: RubixData) -> RubixData:
        logger.info("Calculating Data Cube...")

        # 1. extract arrays
        specs = rubixdata.stars.spectra  # (n_stars, n_wave)
        pix = rubixdata.stars.pixel_assignment  # (n_stars,)
        nstar = specs.shape[0]

        # initial empty cube: (num_segments, n_wave)
        init_cube = jnp.zeros((num_segments, wave_grid.shape[-1]))

        def scan_body(cube, i):
            # process the single spectrum
            spec_i = specs[i]  # shape (n_wave,)
            pix_i = pix[i]  # scalar in [0..nseg)
            # accumulate
            cube = cube.at[pix_i].add(spec_i)
            return cube, None

        # scan over all particle indices 0..n_particles-1
        cube_flat, _ = lax.scan(
            scan_body, init_cube, jnp.arange(nstar, dtype=jnp.int32)
        )

        # reshape to (n_spaxels, n_spaxels, n_wave)
        cube_3d = cube_flat.reshape(num_spaxels, num_spaxels, -1)

        setattr(rubixdata.stars, "datacube", cube_3d)
        logger.debug(f"Datacube shape: {cube_3d.shape}")
        return rubixdata

    return calculate_datacube


@jaxtyped(typechecker=typechecker)
def get_calculate_datacube_particlewise(config: dict) -> Callable:
    """
    Returns a function that builds the IFU cube by, for each star:
      1) looking up SSP
      2) scaling by mass
      3) Doppler‐shifting
      4) resampling
      5) accumulating into the shared datacube
    """
    logger = get_logger(config.get("logger", None))
    telescope = get_telescope(config)
    ns = int(telescope.sbin)
    nseg = ns * ns
    target_wave = telescope.wave_seq  # (n_wave_tel,)

    # prepare SSP lookup
    lookup_ssp = get_lookup_interpolation(config)

    # prepare Doppler machinery
    velocity_direction = rubix_config["ifu"]["doppler"]["velocity_direction"]
    z_obs = config["galaxy"]["dist_z"]
    ssp_model = get_ssp(config)
    ssp_wave0 = cosmological_doppler_shift(
        z=z_obs, wavelength=ssp_model.wavelength
    )  # (n_wave_ssp,)

    @jaxtyped(typechecker=typechecker)
    def calculate_datacube_particlewise(rubixdata: RubixData) -> RubixData:
        logger.info("Calculating Data Cube (combined per‐particle)…")

        stars = rubixdata.stars
        ages = stars.age  # (n_stars,)
        metallicity = stars.metallicity  # (n_stars,)
        masses = stars.mass  # (n_stars,)
        velocities = stars.velocity  # (n_stars,)
        pix_idx = stars.pixel_assignment  # (n_stars,)
        nstar = ages.shape[0]

        # init flat cube: (nseg, n_wave_tel)
        init_cube = jnp.zeros((nseg, target_wave.shape[-1]))

        def body(cube, i):
            age_i = ages[i]  # scalar
            Z_i = metallicity[i]  # scalar
            m_i = masses[i]  # scalar
            v_i = velocities[i]  # scalar or vector
            pix_i = pix_idx[i].astype(jnp.int32)

            # 1) SSP lookup
            spec_ssp = lookup_ssp(Z_i, age_i)  # (n_wave_ssp,)
            # 2) scale by mass
            spec_mass = spec_ssp * m_i  # (n_wave_ssp,)
            # 3) Doppler‐shift wavelengths
            shifted_wave = _velocity_doppler_shift_single(
                wavelength=ssp_wave0,
                velocity=v_i,
                direction=velocity_direction,
            )  # (n_wave_ssp,)
            # 4) resample onto telescope grid
            spec_tel = resample_spectrum(
                initial_spectrum=spec_mass,
                initial_wavelength=shifted_wave,
                target_wavelength=target_wave,
            )  # (n_wave_tel,)

            # 5) accumulate
            cube = cube.at[pix_i].add(spec_tel)
            return cube, None

        cube_flat, _ = lax.scan(body, init_cube, jnp.arange(nstar, dtype=jnp.int32))

        cube_3d = cube_flat.reshape(ns, ns, -1)
        setattr(rubixdata.stars, "datacube", cube_3d)
        logger.debug(f"Datacube shape: {cube_3d.shape}")
        return rubixdata

    # return jax.jit(calculate_datacube_particlewise)
    return calculate_datacube_particlewise
