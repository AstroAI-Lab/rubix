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

        cube_flat, _ = lax.scan(
            body,
            init_cube,
            jnp.arange(nstar, dtype=jnp.int32),
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

        cube_flat, _ = lax.scan(
            body,
            init_cube,
            jnp.arange(nstar, dtype=jnp.int32),
        )

        cube_3d = cube_flat.reshape(ns, ns, -1)
        setattr(rubixdata.stars, "datacube", cube_3d)
        logger.debug(f"Datacube shape: {cube_3d.shape}")
        return rubixdata

    return calculate_dusty_datacube_particlewise
