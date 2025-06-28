import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from rubix import config as rubix_config
from rubix.core.data import RubixData
from rubix.logger import get_logger

from .extinction_models import *


@jaxtyped(typechecker=typechecker)
def calculate_dust_to_gas_ratio(
    gas_metallicity: Float[Array, "n_gas"], model: str, Xco: str
) -> Float[Array, "n_gas"]:
    """
    Calculate the dust_to_gas ratio following the empirical relations from Remy-Ruyer et al. 2014.
    We use the fitting formula from table 1.

    Parameters
    ----------
    gas_metallicity : Float[Array, "n_gas"]
        The metallicity of the gas cells. Remy-Ruyer et al. 2014 use 12 + log(O/H) as a proxy for metallicity.

    model : str
        The model to use for the gas-to-dust ratio as specified in Table 1 of Remy-Ruyer et al. 2014. Options are:
        - power law fixed slope
        - power law free slope
        - broken power law fit
    Returns
    -------
    Float[Array, "n_gas"]
        The gas-to-dust ratio for each gas cell.
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
    dust_column_density: Float[Array, "n_gas"],
    dust_grain_density: float,
    effective_wavelength: float = 5448,  # Johnson V band effective wavelength in Angstrom
) -> Float[Array, "n_gas"]:
    r"""
    Calculate the extinction of the gas cells due to dust.

        The extinction is calculated using the dust column density and the dust-to-gas ratio.
        The dust column density is calculated by multiplying the gas column density with the dust-to-gas ratio.
        See e.g. formula A5 and A6 in the appendix of the paper by Ibarra-Medel et al. 2018.

        Parameters
        ----------
        dust_column_density : Float[Array, "n_gas"]
            The gas column density of each gas cell.
        dust_grain_density : Float
            The dust grain density.
        effective_wavelength : Float
            The effective wavelength in Angstroem of the light at which we calculate the extinction.
            Default value is 5448 Angstrom for Johnson V as taken from here: https://www.aavso.org/filters

        Returns
        -------
        Float[Array, "n_gas"]
            The extinction of the gas cells due to dust.

        Notes
        -----
        Extinction is calculated as:

        .. :math: `A_{\lambda}(z)=\frac{3m_H\pi \Sigma(z)}{0.4 \log(10)\lambda_V \rho_{D}}\times (D/G)`

        where:
        - :math:`A_{\lambda}` is the extinction at the effective wavelength
        - :math:`m_H` is the mass of a hydrogen atom
        - :math:`\Sigma(z)` is the column density of the gas cells
        - :math:`\lambda_V` is the effective wavelength`
        - :math:`\rho_{D}` is the dust grain density
        - :math:`D/G` is the dust-to-gas ratio
    """

    # Constants
    m_H = rubix_config["constants"][
        "MASS_OF_PROTON"
    ]  # mass of a hydrogen atom in grams

    # dust_grain_density is in g/cm^3
    # dust_column_density is internally in Msun per kpc^2, but should be in g/cm^2
    # coordinates internally are in kpc
    # effective_wavelength is in Angstrom = 10^-10 m
    # dust_to_gas_ratio is dimensionless
    # m_H is in grams

    # Note: we adopt a different equation than in Ibarra-Medel et al. 2018 as our dust_column_density is in Msun per kpc^2 and not in particle number per cm^2.
    # Ibarra-Medel give: dust_extinction = 3 * m_H * jnp.pi * gas_column_density / (0.4 * jnp.log(10) * effective_wavelength * 1e-8 * dust_grain_density) * dust_to_gas_ratio

    # convert the surface density to grams per cm^2
    CONVERT_MASS_PER_AREA = (
        float(rubix_config["constants"]["MSUN_TO_GRAMS"])
        / float(rubix_config["constants"]["KPC_TO_CM"]) ** 2
    )
    effective_wavelength = effective_wavelength * 1e-8  # convert to cm
    dust_extinction = (
        3
        * jnp.pi
        * dust_column_density
        * CONVERT_MASS_PER_AREA
        / (0.4 * jnp.log(10) * effective_wavelength * dust_grain_density)
    )

    return dust_extinction


@jaxtyped(typechecker=typechecker)
def apply_spaxel_extinction(
    config: dict,
    rubixdata: RubixData,
    wavelength: Float[Array, "n_wave"],
    n_spaxels: int,
    spaxel_area: Float[Array, "..."],
) -> Float[Array, "1 n_star n_wave"]:
    r"""
    Calculate the extinction for each star in the spaxel and apply dust extinction to it's associated SSP.

    The dust column density is calculated by effectively integrating the dust mass along the z-axis and dividing by pixel area.
    This is done by first sorting the RubixData by spaxel index and within each spaxel segment the gas cells are sorted by their z position.
    Then we calculate the column density of the dust as a function of distance.

    The dust column density is then interpolated to the z positions of the stars.
    The extinction is calculated using the dust column density and the dust-to-gas ratio.
    The extinction is then applied to the SSP fluxes using an Av/Rv dependent extinction model. Default is chosen as Cardelli89.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    rubixdata : RubixData
        The RubixData object containing the spaxel data.
    wavelength : Float[Array, "n_wave"]
        The wavelength of the SSP template fluxes.
    n_spaxels : int
        The number of spaxels.
    spaxel_area : Float[Array, "..."]
        The area of a spaxel.

    Returns
    -------

    Float[Array, "n_star, n_wave"]
        The SSP template fluxes after applying the dust extinction.

    Notes
    -----
        .. math::
        \Sigma(z) = \sum_{i=0}^{n} \rho_i \Delta z_i

        where:
        - :math:`\Sigma(z)` is the column density at position :math:`z`
        - :math:`\rho_i` is the gas density of the i-th cell
        - :math:`\Delta z_i` is the difference between consecutive z positions

        This function makes some approximations that should be valid for densly populated gas cells, i.e.
        the gas cells are much smaller than a spaxel size. The behaviour of this function might be improved by
        rasterizing the gas cells first onto a regular grid.

    """

    logger = get_logger(config.get("logger", None))
    logger.info("Applying dust extinction to the spaxel data using vmap...")

    ext_model = config["ssp"]["dust"]["extinction_model"]
    Rv = config["ssp"]["dust"]["Rv"]

    # Dynamically choose the extinction model based on the string name
    if ext_model not in RV_MODELS:
        raise ValueError(
            f"Extinction model '{ext_model}' is not available. Choose from {RV_MODELS}."
        )

    ext_model_class = Rv_model_dict[ext_model]
    ext = ext_model_class(Rv=Rv)

    # sort the arrays by pixel assignment and z position
    gas_sorted_idx = jnp.lexsort(
        (rubixdata.gas.coords[0, :, 2], rubixdata.gas.pixel_assignment[0])
    )
    stars_sorted_idx = jnp.lexsort(
        (rubixdata.stars.coords[0, :, 2], rubixdata.stars.pixel_assignment[0])
    )

    # determine the segment boundaries
    spaxel_IDs = jnp.arange(n_spaxels)
    # we use searchsorted to get the segment boundaries for the gas and stars arrays and we concatenate the length of the sorted arrays to get the last segment boundary.
    gas_segment_boundaries = jnp.concatenate(
        [
            jnp.searchsorted(
                rubixdata.gas.pixel_assignment[0][gas_sorted_idx],
                spaxel_IDs,
                side="left",
            ),
            jnp.array([len(gas_sorted_idx)]),
        ]
    )
    stars_segment_boundaries = jnp.concatenate(
        [
            jnp.searchsorted(
                rubixdata.stars.pixel_assignment[0][stars_sorted_idx],
                spaxel_IDs,
                side="left",
            ),
            jnp.array([len(stars_sorted_idx)]),
        ]
    )
    # Notes for performance for searchsorted:
    # The method argument controls the algorithm used to compute the insertion indices.
    #
    # 'scan' (the default) tends to be more performant on CPU, particularly when a is very large.
    # 'scan_unrolled' is more performant on GPU at the expense of additional compile time.
    # 'sort' is often more performant on accelerator backends like GPU and TPU, particularly when v is very large.
    # 'compare_all' tends to be the most performant when a is very small.

    # calculate the oxygen abundance, i.e. number fraction of oxygen and hydrogen and with that the dust-to-gas ratio
    # with this we can calculate the dust mass
    # we need to correct by factor of 16 for the difference in atomic mass
    log_OH = 12 + jnp.log10(
        rubixdata.gas.metals[0, :, 4] / (16 * rubixdata.gas.metals[0, :, 0])
    )
    dust_to_gas_ratio = calculate_dust_to_gas_ratio(
        log_OH,
        rubix_config["ssp"]["dust"]["dust_to_gas_model"],
        rubix_config["ssp"]["dust"]["Xco"],
    )
    dust_mass = rubixdata.gas.mass[0] * dust_to_gas_ratio

    dust_grain_density = config["ssp"]["dust"]["dust_grain_density"]
    extinction = (
        calculate_extinction(dust_mass[gas_sorted_idx], dust_grain_density)
        / spaxel_area
    )

    # Preallocate arrays
    Av_array = jnp.zeros_like(rubixdata.stars.mass[0])

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
        # create one mask for the gas positions to move non-segment positions to effectively infinity.
        gas_mask2 = jnp.where(gas_mask, 1, 1e30)

        cumulative_dust_mass = jnp.cumsum(extinction * gas_mask) * gas_mask

        # resort the arrays as jnp.interp requires sorted arrays and our approach of using masks to select the segment is not compatible with this requirement.
        xp_arr = rubixdata.gas.coords[0, :, 2][gas_sorted_idx] * gas_mask2
        fp_arr = cumulative_dust_mass

        xp_arr, fp_arr = jax.lax.sort_key_val(xp_arr, fp_arr)

        interpolated_column_density = (
            jnp.interp(
                rubixdata.stars.coords[0, :, 2][stars_sorted_idx],
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
    # in Rubix the wavelength is in Angstroms, so we divide by 1e4 to get microns.
    extinction = extinguish_vmap(wavelength / 1e4, Av_array)

    # undo the sorting of the stars
    undo_sort = jnp.argsort(stars_sorted_idx)
    extinction = extinction[undo_sort]

    # Apply the extinction to the SSP fluxes
    extincted_ssp_template_fluxes = rubixdata.stars.spectra * extinction

    return extincted_ssp_template_fluxes


def apply_spaxel_extinction_factor(
    config: dict,
    rubixdata: RubixData,
    wavelength: Float[Array, "n_wave"],
    n_spaxels: int,
    spaxel_area: Float[Array, "..."],
) -> Float[Array, "1 n_star n_wave"]:
    r"""
    Calculate the extinction for each star in the spaxel and apply dust extinction to it's associated SSP.

    The dust column density is calculated by effectively integrating the dust mass along the z-axis and dividing by pixel area.
    This is done by first sorting the RubixData by spaxel index and within each spaxel segment the gas cells are sorted by their z position.
    Then we calculate the column density of the dust as a function of distance.

    The dust column density is then interpolated to the z positions of the stars.
    The extinction is calculated using the dust column density and the dust-to-gas ratio.
    The extinction is then applied to the SSP fluxes using an Av/Rv dependent extinction model. Default is chosen as Cardelli89.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    rubixdata : RubixData
        The RubixData object containing the spaxel data.
    wavelength : Float[Array, "n_wave"]
        The wavelength of the SSP template fluxes.
    n_spaxels : int
        The number of spaxels.
    spaxel_area : Float[Array, "..."]
        The area of a spaxel.

    Returns
    -------

    Float[Array, "n_star, n_wave"]
        The SSP template fluxes after applying the dust extinction.

    Notes
    -----
        .. math::
        \Sigma(z) = \sum_{i=0}^{n} \rho_i \Delta z_i

        where:
        - :math:`\Sigma(z)` is the column density at position :math:`z`
        - :math:`\rho_i` is the gas density of the i-th cell
        - :math:`\Delta z_i` is the difference between consecutive z positions

        This function makes some approximations that should be valid for densly populated gas cells, i.e.
        the gas cells are much smaller than a spaxel size. The behaviour of this function might be improved by
        rasterizing the gas cells first onto a regular grid.

    """

    logger = get_logger(config.get("logger", None))
    logger.info("Applying dust extinction to the spaxel data using vmap...")

    ext_model = config["ssp"]["dust"]["extinction_model"]
    Rv = config["ssp"]["dust"]["Rv"]

    # Dynamically choose the extinction model based on the string name
    if ext_model not in RV_MODELS:
        raise ValueError(
            f"Extinction model '{ext_model}' is not available. Choose from {RV_MODELS}."
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
    # we use searchsorted to get the segment boundaries for the gas and stars arrays and we concatenate the length of the sorted arrays to get the last segment boundary.
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
    # The method argument controls the algorithm used to compute the insertion indices.
    #
    # 'scan' (the default) tends to be more performant on CPU, particularly when a is very large.
    # 'scan_unrolled' is more performant on GPU at the expense of additional compile time.
    # 'sort' is often more performant on accelerator backends like GPU and TPU, particularly when v is very large.
    # 'compare_all' tends to be the most performant when a is very small.

    # calculate the oxygen abundance, i.e. number fraction of oxygen and hydrogen and with that the dust-to-gas ratio
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
        # create one mask for the gas positions to move non-segment positions to effectively infinity.
        gas_mask2 = jnp.where(gas_mask, 1, 1e30)

        cumulative_dust_mass = jnp.cumsum(extinction * gas_mask) * gas_mask

        # resort the arrays as jnp.interp requires sorted arrays and our approach of using masks to select the segment is not compatible with this requirement.
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
    # in Rubix the wavelength is in Angstroms, so we divide by 1e4 to get microns.
    extinction = extinguish_vmap(wavelength / 1e4, Av_array)

    # undo the sorting of the stars
    undo_sort = jnp.argsort(stars_sorted_idx)
    extinction = extinction[undo_sort]

    # Apply the extinction to the SSP fluxes
    #extincted_ssp_template_fluxes = rubixdata.stars.spectra * extinction

    return extinction