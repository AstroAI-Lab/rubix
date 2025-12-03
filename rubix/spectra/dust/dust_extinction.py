from typing import Final

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from rubix import config as rubix_config
from rubix.core.data import RubixData
from rubix.logger import get_logger

from .extinction_models import RV_MODELS, Rv_model_dict

N_GAS_AXIS: Final[str] = "n_gas"
N_STAR_AXIS: Final[str] = "n_star"
N_WAVE_AXIS: Final[str] = "n_wave"


@jaxtyped(typechecker=typechecker)
def calculate_dust_to_gas_ratio(
    gas_metallicity: Float[Array, N_GAS_AXIS], model: str, Xco: str
) -> Float[Array, N_GAS_AXIS]:
    r"""
    Calculate the dust-to-gas ratio using the empirical relations from
    Remy-Ruyer et al. (2014). We use the fitting formula from table 1.

    Args:
        gas_metallicity (Float[Array, N_GAS_AXIS]):
            The gas metallicity for each cell, expressed as 12 + log(O/H).
        model (str):
            The fitting relation to use as specified in Table 1 of Remy-Ruyer et al. (2014). Allowed values are ``power law slope
            fixed``, ``power law slope free``, and ``broken power law fit``.
        Xco (str):
            The CO-to-H$_2$ conversion factor. ``MW`` and ``Z`` are available.

    Returns:
        Float[Array, "n_gas"]: The dust-to-gas ratio per gas cell.

    Raises:
        NotImplementedError: If the requested model/Xco combination is missing.
    """

    x_sol = 8.69  # solar oxygen abundance from Asplund et al. 2009

    if Xco == "MW":
        if model == "power law slope fixed":
            raise NotImplementedError(
                "power law slope fixed not implemented yet."
            )  # pragma no cover
        elif model == "power law slope free":
            # power law slope fixed
            # log(D/G) = a + b * log(O/H)
            alpha = 1.62
            a = 2.21
            dust_to_gas_ratio = 1 / 10 ** (a + alpha * (x_sol - gas_metallicity))
        elif model == "broken power law fit":
            # broken power law fit
            # log(D/G) = a + b * log(O/H) for log(O/H) < 8.4
            # log(D/G) = c + d * log(O/H) for log(O/H) >= 8.4
            a = 2.21
            alpha_h = 1.00
            b = 0.68
            alpha_l = 3.08
            x_transition = 7.96
            dust_to_gas_ratio = 1 / jnp.where(
                gas_metallicity > x_transition,
                10 ** (a + alpha_h * (x_sol - gas_metallicity)),
                10 ** (b + alpha_l * (x_sol - gas_metallicity)),
            )
    elif Xco == "Z":
        if model == "power law slope fixed":
            raise NotImplementedError(
                "power law slope fixed not implemented yet."
            )  # pragma no cover
        elif model == "power law slope free":
            # power law slope fixed
            # log(D/G) = a + b * log(O/H)
            alpha = 2.02
            a = 2.21
            dust_to_gas_ratio = 1 / 10 ** (a + alpha * (x_sol - gas_metallicity))
        elif model == "broken power law fit":
            # broken power law fit
            # log(D/G) = a + b * log(O/H) for log(O/H) < 8.4
            # log(D/G) = c + d * log(O/H) for log(O/H) >= 8.4
            a = 2.21
            alpha_h = 1.00
            b = 0.96
            alpha_l = 3.10
            x_transition = 8.10
            dust_to_gas_ratio = 1 / jnp.where(
                gas_metallicity > x_transition,
                10 ** (a + alpha_h * (x_sol - gas_metallicity)),
                10 ** (b + alpha_l * (x_sol - gas_metallicity)),
            )

    return dust_to_gas_ratio


@jaxtyped(typechecker=typechecker)
def calculate_extinction(
    dust_column_density: Float[Array, N_GAS_AXIS],
    dust_grain_density: float,
    effective_wavelength: float = 5448,  # Johnson V band effective wavelength in Angstrom
) -> Float[Array, N_GAS_AXIS]:
    r"""
    Calculate the extinction of gas cells due to dust.

    The extinction is derived from the dust column density and the dust-to-gas
    ratio following Ibarra-Medel et al. (2018, Appendix A, formula A5 and A6).

    Args:
        dust_column_density (Float[Array, N_GAS_AXIS]):
            The gas column density of each cell.
        dust_grain_density (float):
            Dust grain density in g/cm^3.
        effective_wavelength (float, optional):
            Effective wavelength (Angstrom) used for the extinction estimate.
            Defaults to 5448 A (Johnson V band as taken from https://www.aavso.org/filters).

    Returns:
        Float[Array, N_GAS_AXIS]: Extinction for each gas cell.

    Notes:
    Extinction is calculated as:
        .. math::
            A_{\lambda}(z) =
                \frac{3 m_H \pi \Sigma(z)}{0.4 \log(10) \lambda_V \rho_D}
                \cdot (D/G)

    where \(m_H\) is the proton mass and
    \(\Sigma(z)\) is the gas column density,
    \(\lambda_V\) is the effective wavelength, \(\rho_D\) is the dust grain
    density, and \(D/G\) is the dust-to-gas ratio.
    """

    # Constants
    m_H = rubix_config["constants"][
        "MASS_OF_PROTON"
    ]  # mass of a hydrogen atom in grams

    # dust_grain_density is in g/cm^3
    # dust_column_density is internally in Msun per kpc^2.
    # It should be converted to g/cm^2.
    # coordinates internally are in kpc
    # effective_wavelength is in Angstrom = 10^-10 m
    # dust_to_gas_ratio is dimensionless
    # m_H is in grams
    # Note: we adopt a different equation than Ibarra-Medel et al. 2018.
    # Our dust_column_density is in Msun per kpc^2 rather than particle number.
    # Ibarra-Medel give:
    #     dust_extinction = 3 * m_H * jnp.pi * gas_column_density
    #     / (
    #         0.4 * jnp.log(10)
    #         * effective_wavelength
    #         * 1e-8
    #         * dust_grain_density
    #     )
    #     * dust_to_gas_ratio

    # convert the surface density to grams per cm^2
    CONVERT_MASS_PER_AREA = (
        float(rubix_config["constants"]["MSUN_TO_GRAMS"])
        / float(rubix_config["constants"]["KPC_TO_CM"]) ** 2
    )
    effective_wavelength = effective_wavelength * 1e-8  # convert to cm
    dust_extinction = (
        3
        * jnp.pi
        * m_H
        * dust_column_density
        * CONVERT_MASS_PER_AREA
        / (0.4 * jnp.log(10) * effective_wavelength * dust_grain_density)
    )

    return dust_extinction


@jaxtyped(typechecker=typechecker)
def apply_spaxel_extinction(
    config: dict,
    rubixdata: RubixData,
    wavelength: Float[Array, N_WAVE_AXIS],
    n_spaxels: int,
    spaxel_area: Float[Array, "..."],
) -> Float[Array, N_STAR_AXIS]:
    r"""
    Calculate the extinction for each star in a spaxel and cache the Av values.

    The dust column density is calculated by effectively integrating the dust mass along the z-axis and dividing by pixel area.
    This is done by first sorting the RubixData by spaxel index and within each spaxel segment the gas cells are sorted by their z position.
    Then we calculate the column density of the dust as a function of distance.

    The dust column density is then interpolated to the z positions of the stars.
    The extinction is calculated using the dust column density and the dust-to-gas ratio.
    The extinction is then applied to the SSP fluxes using an Av/Rv dependent extinction model. Default is chosen as Cardelli89.

    Args:
        config (dict): Configuration dictionary.
        rubixdata (RubixData): RubixData object holding spaxel data.
        wavelength (Float[Array, N_WAVE_AXIS]):
            Wavelengths of the SSP template fluxes.
        n_spaxels (int): Number of spaxels.
        spaxel_area (Float[Array, "..."]): Area of one spaxel.

    Returns:
        Float[Array, N_STAR_AXIS]: Visual extinction for each star in the cube.

    Raises:
        ValueError: If the requested extinction model is not available.

    Notes:
        .. math::
            \Sigma(z) = \sum_{i=0}^{n} \rho_i \Delta z_i

        where \Sigma(z)\ is the gas column density at position z, \(\rho_i\) is the gas density of the i-th gas cell and
        \(\Delta z_i\) the difference between the z positions of the i-th and (i-1)-th gas cells.

        The implementation assumes gas cells are much smaller than a spaxel.
        Rasterizing the gas cells to a regular grid may improve accuracy.
    """

    logger = get_logger(config.get("logger", None))
    logger.info("Applying dust extinction to the spaxel data using vmap...")

    ext_model = config["ssp"]["dust"]["extinction_model"]
    Rv = config["ssp"]["dust"]["Rv"]

    # Dynamically choose the extinction model based on the string name
    if ext_model not in RV_MODELS:
        raise ValueError(
            f"Extinction model '{ext_model}' is not available. "
            f"Choose from {RV_MODELS}."
        )

    ext_model_class = Rv_model_dict[ext_model]
    ext = ext_model_class(Rv=Rv)

    # sort the arrays by pixel assignment and z position
    gas_sorted_idx = jnp.lexsort(
        (rubixdata.gas.coords[:, 2], rubixdata.gas.pixel_assignment)
    )
    stars_sorted_idx = jnp.lexsort(
        (rubixdata.stars.coords[:, 2], rubixdata.stars.pixel_assignment)
    )

    # determine the segment boundaries
    spaxel_IDs = jnp.arange(n_spaxels)
    # Searchsorted identifies boundaries for the gas and star arrays.
    # Concatenating the sorted lengths produces the final boundary.
    gas_segment_boundaries = jnp.concatenate(
        [
            jnp.searchsorted(
                rubixdata.gas.pixel_assignment[gas_sorted_idx],
                spaxel_IDs,
                side="left",
            ),
            jnp.array([len(gas_sorted_idx)]),
        ]
    )
    stars_segment_boundaries = jnp.concatenate(
        [
            jnp.searchsorted(
                rubixdata.stars.pixel_assignment[stars_sorted_idx],
                spaxel_IDs,
                side="left",
            ),
            jnp.array([len(stars_sorted_idx)]),
        ]
    )
    # Notes for performance for searchsorted:
    # The method argument controls the algorithm used for insertion indices.
    #
    # 'scan' is more performant on CPU, particularly when a is very large.
    # 'scan_unrolled' can be faster on GPU but increases compile time.
    # 'sort' often wins on accelerator backends when v is large.
    # 'compare_all' tends to be the most performant when a is very small.

    # Calculate the oxygen abundance (number fraction of oxygen and hydrogen).
    # This provides the dust-to-gas ratio.
    # with this we can calculate the dust mass
    # we need to correct by factor of 16 for the difference in atomic mass
    log_OH = 12 + jnp.log10(
        rubixdata.gas.metals[:, 4] / (16 * rubixdata.gas.metals[:, 0])
    )
    dust_to_gas_ratio = calculate_dust_to_gas_ratio(
        log_OH,
        rubix_config["ssp"]["dust"]["dust_to_gas_model"],
        rubix_config["ssp"]["dust"]["Xco"],
    )
    dust_mass = rubixdata.gas.mass * dust_to_gas_ratio

    dust_grain_density = config["ssp"]["dust"]["dust_grain_density"]
    extinction = (
        calculate_extinction(dust_mass[gas_sorted_idx], dust_grain_density)
        / spaxel_area
    )

    # Preallocate arrays
    Av_array = jnp.zeros_like(rubixdata.stars.mass)

    def body_fn(carry, idx):
        Av_array = carry
        gas_start, gas_end = (
            gas_segment_boundaries[idx],
            gas_segment_boundaries[idx + 1],
        )
        star_start, star_end = (
            stars_segment_boundaries[idx],
            stars_segment_boundaries[idx + 1],
        )

        # Create masks for the current segment
        gas_mask = (jnp.arange(gas_sorted_idx.shape[0]) >= gas_start) & (
            jnp.arange(gas_sorted_idx.shape[0]) < gas_end
        )
        star_mask = (jnp.arange(stars_sorted_idx.shape[0]) >= star_start) & (
            jnp.arange(stars_sorted_idx.shape[0]) < star_end
        )
        # Create a mask for the gas positions.
        # Non-segment positions are moved to effectively infinity.
        gas_mask2 = jnp.where(gas_mask, 1, 1e30)

        cumulative_dust_mass = jnp.cumsum(extinction * gas_mask) * gas_mask

        # Sort the arrays because jnp.interp requires sorted inputs.
        # Our mask-based approach is not compatible with that requirement.
        xp_arr = rubixdata.gas.coords[:, 2][gas_sorted_idx] * gas_mask2
        fp_arr = cumulative_dust_mass

        xp_arr, fp_arr = jax.lax.sort_key_val(xp_arr, fp_arr)

        interpolated_column_density = (
            jnp.interp(
                rubixdata.stars.coords[:, 2][stars_sorted_idx],
                xp_arr,
                fp_arr,
                left="extrapolate",
            )
            * star_mask
        )

        # calculate the extinction for each star
        Av_array += interpolated_column_density

        return Av_array, None

    Av_array, _ = jax.lax.scan(body_fn, Av_array, spaxel_IDs)

    # get the extinguished SSP flux for different amounts of dust
    # Vectorize the extinction calculation using vmap
    extinguish_vmap = jax.vmap(ext.extinguish, in_axes=(None, 0))
    # note, we need to pass wavelength in microns here to the extinction model.
    # Rubix stores wavelengths in Angstroms; we divide by 1e4 to get microns.
    extinction = extinguish_vmap(wavelength / 1e4, Av_array)

    # undo the sorting of the stars
    undo_sort = jnp.argsort(stars_sorted_idx)
    Av_array = Av_array[undo_sort]
    # extinction = extinction[undo_sort]

    # Apply the extinction to the SSP fluxes
    # extincted_ssp_template_fluxes = rubixdata.stars.spectra * extinction

    return Av_array
