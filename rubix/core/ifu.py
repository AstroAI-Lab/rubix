import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Callable
from jax import lax
from jaxtyping import Array, Float, jaxtyped

from rubix import config as rubix_config
from rubix.core.data import GasData, StarsData
from rubix.logger import get_logger
from rubix.spectra.dust.extinction_models import RV_MODELS, Rv_model_dict
from rubix.spectra.ifu import (
    _velocity_doppler_shift_single,
    cosmological_doppler_shift,
    resample_spectrum,
)

from .data import RubixData
from .ssp import get_lookup_interpolation, get_ssp
from .telescope import get_telescope


@jaxtyped(typechecker=typechecker)
def _get_performance_options(config: dict) -> tuple[int, bool]:
    """Read optional particlewise performance settings from the config.

    Args:
        config (dict): Runtime configuration dictionary.

    Returns:
        tuple[int, bool]:
            ``(chunk_size, use_remat)`` where ``chunk_size == 0`` means
            unchunked execution.
    """
    perf_config = config.get("performance", {})
    chunk_size = perf_config.get("particle_chunk_size", 0)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        chunk_size = 0

    use_remat = bool(perf_config.get("remat_particlewise", False))
    return chunk_size, use_remat


@jaxtyped(typechecker=typechecker)
def _scan_particles(
    init_cube: Float[Array, "n_spaxels n_wave_bins"],
    nstar: int,
    step_fn: Callable,
    chunk_size: int,
) -> Float[Array, "n_spaxels n_wave_bins"]:
    """Accumulate per-particle contributions with optional chunking.

    Args:
        init_cube (Float[Array, "n_spaxels n_wave_bins"]): Initial cube.
        nstar (int): Number of particles.
        step_fn (Callable): Particle step function of signature
            ``(cube, index) -> (cube, aux)``.
        chunk_size (int): Chunk size; ``0`` disables chunking.

    Returns:
        Float[Array, "n_spaxels n_wave_bins"]: Accumulated flat cube.
    """
    if nstar == 0:
        return init_cube

    if chunk_size <= 0:
        cube_flat, _ = lax.scan(step_fn, init_cube, jnp.arange(nstar, dtype=jnp.int32))
        return cube_flat

    n_chunks = (nstar + chunk_size - 1) // chunk_size
    max_index = nstar - 1

    def chunk_body(cube, chunk_idx):
        start = chunk_idx * chunk_size
        local = jnp.arange(chunk_size, dtype=jnp.int32)
        idx = start + local
        idx_safe = jnp.minimum(idx, max_index)
        valid = idx < nstar

        def inner_body(i, cube_inner):
            def do_step(current_cube):
                cube_new, _ = step_fn(current_cube, idx_safe[i])
                return cube_new

            return lax.cond(
                valid[i],
                do_step,
                lambda current_cube: current_cube,
                cube_inner,
            )

        cube = lax.fori_loop(0, chunk_size, inner_body, cube)
        return cube, None

    cube_flat, _ = lax.scan(
        chunk_body, init_cube, jnp.arange(n_chunks, dtype=jnp.int32)
    )
    return cube_flat


@jaxtyped(typechecker=typechecker)
def get_calculate_datacube_particlewise(config: dict) -> Callable:
    """Prepare a per-particle datacube builder for the star component.

    The returned callable performs an SSP lookup, scales by mass, applies the
    Doppler shift, resamples onto the telescope wavelength grid, and
    aggregates the flux into spatial pixels.

    First, it looks up the SSP spectrum for each star based on its age and metallicity,
    scales it by the star's mass, applies a Doppler shift based on the star's velocity,
    resamples the spectrum onto the telescope's wavelength grid, and finally accumulates
    the resulting spectra into the appropriate pixels of the datacube.

    Args:
        config (dict): Configuration dictionary containing telescope and galaxy
            parameters.

    Returns:
        Callable[[RubixData], RubixData]:
            Function that computes ``stars.datacube``.
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
    chunk_size, use_remat = _get_performance_options(config)

    @jaxtyped(typechecker=typechecker)
    def calculate_datacube_particlewise(rubixdata: RubixData) -> RubixData:
        """Compute the star datacube for a single RubixData batch.

        Args:
            rubixdata (RubixData): Particle data with star attributes populated.

        Returns:
            RubixData: Same RubixData with ``stars.datacube`` populated.
        """
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

        particle_step = body
        if use_remat:
            particle_step = jax.checkpoint(body)

        cube_flat = _scan_particles(
            init_cube=init_cube,
            nstar=nstar,
            step_fn=particle_step,
            chunk_size=chunk_size,
        )

        cube_3d = cube_flat.reshape(ns, ns, -1)
        setattr(rubixdata.stars, "datacube", cube_3d)
        logger.debug(f"Datacube shape: {cube_3d.shape}")
        return rubixdata

    return calculate_datacube_particlewise


@jaxtyped(typechecker=typechecker)
def get_calculate_dusty_datacube_particlewise(config: dict) -> Callable:
    """Prepare a dusty per-particle datacube builder for the star component.

    The returned callable is similar to
    :func:`get_calculate_datacube_particlewise` but applies
    wavelength-dependent extinction using the configured dust model.

    First, it looks up the SSP spectrum for each star based on its age and metallicity,
    scales it by the star's mass, applies a Doppler shift based on the star's velocity,
    resamples the spectrum onto the telescope's wavelength grid, and finally accumulates
    the resulting spectra into the appropriate pixels of the datacube.

    Args:
        config (dict): Configuration dictionary containing telescope and galaxy
            parameters as well as ``ssp.dust`` settings.

    Returns:
        Callable[[RubixData], RubixData]:
            Function that computes ``stars.datacube`` with extinction.
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
    chunk_size, use_remat = _get_performance_options(config)

    @jaxtyped(typechecker=typechecker)
    def calculate_dusty_datacube_particlewise(
        rubixdata: RubixData,
    ) -> RubixData:
        """Apply SSP spectra, Doppler shifts, and extinction per particle.

        Args:
            rubixdata (RubixData): Particle data with dust extinction arrays.

        Returns:
            RubixData: Input data updated with ``stars.datacube``.

        Raises:
            ValueError: If the configured extinction model is unavailable.
        """
        logger.info("Calculating Data Cube (combined per‐particle)…")

        stars = rubixdata.stars
        ages = stars.age  # (n_stars,)
        metallicity = stars.metallicity  # (n_stars,)
        masses = stars.mass  # (n_stars,)
        velocities = stars.velocity  # (n_stars,)
        pix_idx = stars.pixel_assignment  # (n_stars,)
        Av_array = stars.extinction  # (n_stars, n_wave_ssp)
        nstar = ages.shape[0]

        # dust model
        ext_model = config["ssp"]["dust"]["extinction_model"]
        Rv = config["ssp"]["dust"]["Rv"]
        # Dynamically choose the extinction model based on the string name
        if ext_model not in RV_MODELS:  # pragma: no cover
            raise ValueError(
                "Extinction model '{ext_model}' is not available. "
                f"Choose from {RV_MODELS}."
            )

        ext_model_class = Rv_model_dict[ext_model]
        ext = ext_model_class(Rv=Rv)

        # init flat cube: (nseg, n_wave_tel)
        init_cube = jnp.zeros((nseg, target_wave.shape[-1]))

        def body(cube, i):
            age_i = ages[i]  # scalar
            Z_i = metallicity[i]  # scalar
            m_i = masses[i]  # scalar
            v_i = velocities[i]  # scalar or vector
            pix_i = pix_idx[i].astype(jnp.int32)
            av_i = Av_array[i]  # (n_wave_ssp,)

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

            # apply extinction
            extinction = ext.extinguish(target_wave / 1e4, av_i)

            spec_extincted = spec_tel * extinction  # (n_wave_tel,)

            # 5) accumulate
            cube = cube.at[pix_i].add(spec_extincted)
            return cube, None

        particle_step = body
        if use_remat:
            particle_step = jax.checkpoint(body)

        cube_flat = _scan_particles(
            init_cube=init_cube,
            nstar=nstar,
            step_fn=particle_step,
            chunk_size=chunk_size,
        )

        cube_3d = cube_flat.reshape(ns, ns, -1)
        setattr(rubixdata.stars, "datacube", cube_3d)
        logger.debug(f"Datacube shape: {cube_3d.shape}")
        return rubixdata

    return calculate_dusty_datacube_particlewise
