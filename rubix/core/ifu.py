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
    cosmological_doppler_shift,
    resample_spectrum,
)

from .data import RubixData
from .ssp import get_lookup_interpolation, get_ssp, get_vectorized_ssp_lookup
from .telescope import get_telescope


@jaxtyped(typechecker=typechecker)
def get_calculate_datacube_vectorized(config: dict) -> Callable:
    """
    Returns a vectorized function that builds the IFU cube by processing all stars at once:
      1) vectorized SSP lookup for all stars
      2) scaling by mass (vectorized)
      3) Doppler-shifting (vectorized)
      4) resampling (vectorized)
      5) accumulating into the shared datacube using segment_sum

    This should be much faster than the particle-wise approach.
    """
    logger = get_logger(config.get("logger", None))
    telescope = get_telescope(config)
    ns = int(telescope.sbin)
    nseg = ns * ns
    target_wave = telescope.wave_seq  # (n_wave_tel,)

    # Prepare vectorized SSP lookup
    lookup_ssp_vectorized = get_vectorized_ssp_lookup(config)

    # Prepare Doppler machinery
    velocity_direction = rubix_config["ifu"]["doppler"]["velocity_direction"]
    z_obs = config["galaxy"]["dist_z"]
    ssp_model = get_ssp(config)
    ssp_wave0 = cosmological_doppler_shift(
        z=z_obs, wavelength=ssp_model.wavelength
    )  # (n_wave_ssp,)

    # Vectorized Doppler shift function
    doppler_shift_vmap = jax.vmap(
        _velocity_doppler_shift_single,
        in_axes=(None, 0, None),  # wavelength is broadcasted, velocity is vectorized
    )

    # Vectorized resampling function
    resample_vmap = jax.vmap(
        resample_spectrum,
        in_axes=(
            0,
            0,
            None,
        ),  # initial_spectrum and wavelength vectorized, target fixed
    )

    @jaxtyped(typechecker=typechecker)
    def calculate_datacube_vectorized(rubixdata: RubixData) -> RubixData:
        logger.info("Calculating Data Cube (vectorized)...")

        stars = rubixdata.stars
        ages = stars.age  # (n_stars,)
        metallicity = stars.metallicity  # (n_stars,)
        masses = stars.mass  # (n_stars,)
        velocities = stars.velocity  # (n_stars,) or (n_stars, 3)
        pix_idx = stars.pixel_assignment.astype(jnp.int32)  # (n_stars,)

        # 1) Vectorized SSP lookup for all stars at once
        spectra_ssp = lookup_ssp_vectorized(metallicity, ages)  # (n_stars, n_wave_ssp)

        # 2) Scale by mass (broadcasting)
        spectra_mass = spectra_ssp * masses[:, None]  # (n_stars, n_wave_ssp)

        # 3) Vectorized Doppler shifting
        # Handle velocity direction
        if velocity_direction in [0, 1, 2]:  # specific component
            v_radial = (
                velocities[:, velocity_direction] if velocities.ndim > 1 else velocities
            )
        else:  # assume line-of-sight is provided
            v_radial = velocities

        shifted_waves = doppler_shift_vmap(
            ssp_wave0, v_radial, velocity_direction
        )  # (n_stars, n_wave_ssp)

        # 4) Vectorized resampling
        spectra_tel = resample_vmap(
            spectra_mass, shifted_waves, target_wave
        )  # (n_stars, n_wave_tel)

        # 5) Accumulate into datacube using segment_sum for efficiency
        # This is much faster than using lax.scan with .at[].add()
        cube_flat = jax.ops.segment_sum(
            spectra_tel,  # data to sum: (n_stars, n_wave_tel)
            pix_idx,  # segment ids: (n_stars,)
            num_segments=nseg,
            indices_are_sorted=False,
        )  # Result: (nseg, n_wave_tel)

        cube_3d = cube_flat.reshape(ns, ns, -1)
        setattr(rubixdata.stars, "datacube", cube_3d)
        logger.debug(f"Datacube shape: {cube_3d.shape}")
        return rubixdata

    return calculate_datacube_vectorized


# Keep the original function for backward compatibility
@jaxtyped(typechecker=typechecker)
def get_calculate_datacube_particlewise(config: dict) -> Callable:
    """
    Returns a function that builds the IFU cube by, for each star:
      1) looking up SSP
      2) scaling by mass
      3) Doppler‐shifting
      4) resampling
      5) accumulating into the shared datacube

    Args
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


@jaxtyped(typechecker=typechecker)
def get_calculate_datacube_optimized(config: dict) -> Callable:
    """
    Returns a fully optimized function that builds the IFU cube using vectorized operations
    with optional chunking for memory management.
    """
    logger = get_logger(config.get("logger", None))
    telescope = get_telescope(config)
    ns = int(telescope.sbin)
    nseg = ns * ns
    target_wave = telescope.wave_seq  # (n_wave_tel,)

    # Prepare vectorized SSP lookup
    lookup_ssp = get_lookup_interpolation(config)
    lookup_ssp_vectorized = jax.vmap(lookup_ssp, in_axes=(0, 0))

    # Prepare Doppler machinery
    velocity_direction = rubix_config["ifu"]["doppler"]["velocity_direction"]
    z_obs = config["galaxy"]["dist_z"]
    ssp_model = get_ssp(config)
    ssp_wave0 = cosmological_doppler_shift(
        z=z_obs, wavelength=ssp_model.wavelength
    )  # (n_wave_ssp,)

    # Vectorized functions
    doppler_shift_vmap = jax.vmap(
        _velocity_doppler_shift_single, in_axes=(None, 0, None)
    )

    resample_vmap = jax.vmap(resample_spectrum, in_axes=(0, 0, None))

    def process_chunk(
        ages_chunk, metallicity_chunk, masses_chunk, velocities_chunk, pix_idx_chunk
    ):
        """Process a chunk of particles"""
        # 1) Vectorized SSP lookup
        spectra_ssp = lookup_ssp_vectorized(metallicity_chunk, ages_chunk)

        # 2) Scale by mass
        spectra_mass = spectra_ssp * masses_chunk[:, None]

        # 3) Vectorized Doppler shifting
        if velocity_direction in [0, 1, 2]:
            v_radial = (
                velocities_chunk[:, velocity_direction]
                if velocities_chunk.ndim > 1
                else velocities_chunk
            )
        else:
            v_radial = velocities_chunk

        shifted_waves = doppler_shift_vmap(ssp_wave0, v_radial, velocity_direction)

        # 4) Vectorized resampling
        spectra_tel = resample_vmap(spectra_mass, shifted_waves, target_wave)

        return spectra_tel, pix_idx_chunk

    @jaxtyped(typechecker=typechecker)
    def calculate_datacube_optimized(rubixdata: RubixData) -> RubixData:
        logger.info("Calculating Data Cube (optimized)...")

        stars = rubixdata.stars
        ages = stars.age
        metallicity = stars.metallicity
        masses = stars.mass
        velocities = stars.velocity
        pix_idx = stars.pixel_assignment.astype(jnp.int32)
        nstar = ages.shape[0]

        # Determine chunk size based on memory constraints
        chunk_size = min(10000, nstar)  # Adjust based on your GPU memory

        if nstar <= chunk_size:
            # Process all at once if small enough
            spectra_tel, pixel_indices = process_chunk(
                ages, metallicity, masses, velocities, pix_idx
            )
        else:
            # Process in chunks for large datasets
            spectra_list = []
            pixel_list = []

            for start_idx in range(0, nstar, chunk_size):
                end_idx = min(start_idx + chunk_size, nstar)

                chunk_spectra, chunk_pixels = process_chunk(
                    ages[start_idx:end_idx],
                    metallicity[start_idx:end_idx],
                    masses[start_idx:end_idx],
                    velocities[start_idx:end_idx],
                    pix_idx[start_idx:end_idx],
                )

                spectra_list.append(chunk_spectra)
                pixel_list.append(chunk_pixels)

            spectra_tel = jnp.concatenate(spectra_list, axis=0)
            pixel_indices = jnp.concatenate(pixel_list, axis=0)

        # 5) Accumulate using segment_sum
        cube_flat = jax.ops.segment_sum(
            spectra_tel,
            pixel_indices,
            num_segments=nseg,
            indices_are_sorted=False,
        )

        cube_3d = cube_flat.reshape(ns, ns, -1)
        setattr(rubixdata.stars, "datacube", cube_3d)
        logger.debug(f"Datacube shape: {cube_3d.shape}")
        return rubixdata

    return calculate_datacube_optimized
