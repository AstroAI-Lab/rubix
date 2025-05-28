import equinox
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from .dust_baseclasses import BaseExtRvModel
from .generic_models import FM90, Drude1d, Polynomial1d, PowerLaw1d, _modified_drude
from .helpers import _smoothstep

RV_MODELS = [
    "Cardelli89",
    "Gordon23",
]  # "O94", "F99", "F04", "VCG04", "GCC09", "M14", "G16", "F19", "D22", "G23"]

wave_range_CCM89 = [0.3, 10.0]
Rv_range_CCM89 = [2.0, 6.0]

wave_range_G23 = [0.0912, 32.0]
Rv_range_G23 = [2.3, 5.6]


@equinox.filter_jit
@jaxtyped(typechecker=typechecker)
class Cardelli89(BaseExtRvModel):
    r"""
    Calculate the extinction curve of the Milky Way according to the
    Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model.

    Parameters
    ----------
    Rv : Float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Returns
    -------
    Float[Array, "n_wave"]
        A(x)/A(V) extinction curve [mag]

    Raises
    ------
    InputParameterError
        Input Rv values outside of defined range

    Notes
    -----
    From Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245)

    Example showing CCM89 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from rubix.spectra.dust.extinction_models import Cardelli89

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.5,10.0,0.1) # units of micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
            ext_model = Cardelli89(Rv=cur_Rv)
            ax.plot(x,ext_model,label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    # wave: Float[Array, "n_wave"]

    # wave_range: Float[Array, "2"] = equinox.field(converter=jnp.asarray, static=True, default_factory=lambda: jnp.array(wave_range_CCM89))
    wave_range_l: float = equinox.field(
        converter=float, static=True, default=wave_range_CCM89[0]
    )
    wave_range_h: float = equinox.field(
        converter=float, static=True, default=wave_range_CCM89[1]
    )

    Rv: float = equinox.field(converter=float, static=True, default=3.1)
    # Rv_range: Float[Array, "2"] = equinox.field(converter=jnp.asarray, static=True, default_factory=lambda: jnp.array(Rv_range_CCM89))
    Rv_range_l: float = equinox.field(
        converter=float, static=True, default=Rv_range_CCM89[0]
    )
    Rv_range_h: float = equinox.field(
        converter=float, static=True, default=Rv_range_CCM89[1]
    )

    def evaluate(self, wave: Float[Array, "n_wave"]) -> Float[Array, "n_wave"]:
        """
        Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245) function

        Parameters
        ----------
        wave: float
            expects wave as wavelengths in microns.

        Returns
        -------
        axav: jax numpy array (float)
            A(wave)/A(V) extinction curve [mag]
        """

        # setup the a & b coefficient vectors
        a = jnp.zeros(wave.shape)
        b = jnp.zeros(wave.shape)

        # define the ranges
        ir_mask = jnp.logical_and(0.3 <= wave, wave < 1.1)
        opt_mask = jnp.logical_and(1.1 <= wave, wave < 3.3)
        nuv_mask = jnp.logical_and(3.3 <= wave, wave <= 8.0)
        fnuv_mask = jnp.logical_and(5.9 <= wave, wave <= 8)
        fuv_mask = jnp.logical_and(8 < wave, wave <= 10)

        # Infrared
        a = jnp.where(ir_mask, 0.574 * wave**1.61, a)
        b = jnp.where(ir_mask, -0.527 * wave**1.61, b)

        # NIR/optical
        y = wave - 1.82
        a = jnp.where(
            opt_mask,
            1
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7,
            a,
        )
        b = jnp.where(
            opt_mask,
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7,
            b,
        )

        a = jnp.where(
            nuv_mask, 1.752 - 0.316 * wave - 0.104 / ((wave - 4.67) ** 2 + 0.341), a
        )
        b = jnp.where(
            nuv_mask, -3.09 + 1.825 * wave + 1.206 / ((wave - 4.62) ** 2 + 0.263), b
        )

        # far-NUV
        y = wave - 5.9
        a = jnp.where(fnuv_mask, a + (-0.04473 * (y**2) - 0.009779 * (y**3)), a)
        b = jnp.where(fnuv_mask, b + (0.2130 * (y**2) + 0.1207 * (y**3)), b)

        # FUV
        y = wave - 8.0
        a = jnp.where(fuv_mask, -1.073 - 0.628 * y + 0.137 * y**2 - 0.070 * y**3, a)
        b = jnp.where(fuv_mask, 13.670 + 4.257 * y - 0.420 * y**2 + 0.374 * y**3, b)

        # return A(x)/A(V)
        return a + b / self.Rv


@jaxtyped(typechecker=typechecker)
class Gordon23(BaseExtRvModel):
    r"""
    Gordon et al. (2023) Milky Way R(V) dependent model

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Returns
    -------
    Float[Array, "n_wave"]
        A(x)/A(V) extinction curve [mag]

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    From Gordon et al. (2023, ApJ, in press)

    Example showing G23 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import G23

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(0.0912), np.log10(30.0), num=1000) * u.micron

        Rvs = [2.5, 3.1, 4.0, 4.75, 5.5]
        for cur_Rv in Rvs:
           ext_model = G23(Rv=cur_Rv)
           ax.plot(lam,ext_model(lam),label='R(V) = ' + str(cur_Rv))

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    # wave_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array(wave_range_G23))
    # Rv_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array(Rv_range_G23))

    wave_range_l: float = equinox.field(
        converter=float, static=True, default=wave_range_G23[0]
    )
    wave_range_h: float = equinox.field(
        converter=float, static=True, default=wave_range_G23[1]
    )

    Rv: float = equinox.field(converter=float, static=True, default=3.1)
    # Rv_range: Float[Array, "2"] = equinox.field(converter=jnp.asarray, static=True, default_factory=lambda: jnp.array(Rv_range_CCM89))
    Rv_range_l: float = equinox.field(
        converter=float, static=True, default=Rv_range_G23[0]
    )
    Rv_range_h: float = equinox.field(
        converter=float, static=True, default=Rv_range_G23[1]
    )

    def evaluate(self, wave: Float[Array, "n_wave"]) -> Float[Array, "n_wave"]:
        """
        Gordon 2023 function (The Astrophysical Journal, Volume 950, Issue 2, id.86, 13 pp.)

        Parameters
        ----------
        wave: float
           expects wave as wavelengths in micron

        Returns
        -------
        axav: np array (float)
            A(wave)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input wave values outside of defined range
        """
        # setup the a & b coefficient vectors
        a = jnp.zeros(wave.shape)
        b = jnp.zeros(wave.shape)

        # define the ranges
        ir_mask = jnp.logical_and(1.0 <= wave, wave < 35.0)
        opt_mask = jnp.logical_and(0.3 <= wave, wave < 1.1)
        uv_mask = jnp.logical_and(0.09 <= wave, wave <= 0.3)

        # overlap ranges
        optir_waves = [0.9, 1.1]
        optir_overlap = jnp.logical_and(wave >= optir_waves[0], wave <= optir_waves[1])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = jnp.logical_and(wave >= uvopt_waves[0], wave <= uvopt_waves[1])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.38526, 1.68467, 0.78791, 4.30578, 4.78338,
                0.06652, 9.8434, 2.21205, -0.24703,
                0.0267 , 19.58294, 17., -0.27]
        # fmt: on

        a = jnp.where(ir_mask, self.nirmir_intercept(wave, ir_a), a)
        b = jnp.where(
            ir_mask, PowerLaw1d(x=wave, amplitude=-1.01251, x_0=1.0, alpha=-1.06099), b
        )

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.35848, 0.7122 , 0.08746, -0.05403, 0.00674,
                 0.03893, 2.288, 0.243,
                 0.02965, 2.054, 0.179,
                 0.01747, 1.587, 0.243]
        opt_b = [0.12354, -2.68335, 2.01901, -0.39299, 0.03355,
                 0.18453, 2.288, 0.243,
                 0.19728, 2.054, 0.179,
                 0.1713 , 1.587, 0.243]
        # fmt: on

        def compound_polynomial_drude_model(
            x: Float[Array, "n_wave"], params: Float[Array, "m"]
        ) -> Float[Array, "n_wave"]:
            """
            Compound polynomial and Drude model

            Parameters
            ----------
            x : ndarray
                input wavelengths in wavenumbers [1/micron]
            params : ndarray
                model parameters

            Returns
            -------
            y : ndarray
                output profile
            """
            # Extract polynomial coefficients and Drude model parameters from opt_a
            poly_coeffs = params[:5]  # First 5 elements for polynomial coefficients
            drude_params = params[5:]  # Remaining elements for Drude model parameters

            # Evaluate the polynomial model
            poly_result = Polynomial1d(x, poly_coeffs)

            # Evaluate the Drude models
            drude_result_1 = Drude1d(
                x, amplitude=drude_params[0], x_0=drude_params[1], fwhm=drude_params[2]
            )
            drude_result_2 = Drude1d(
                x, amplitude=drude_params[3], x_0=drude_params[4], fwhm=drude_params[5]
            )
            drude_result_3 = Drude1d(
                x, amplitude=drude_params[6], x_0=drude_params[7], fwhm=drude_params[8]
            )

            # Combine the results
            return poly_result + drude_result_1 + drude_result_2 + drude_result_3

        a = jnp.where(opt_mask, compound_polynomial_drude_model(1 / wave, opt_a), a)
        b = jnp.where(opt_mask, compound_polynomial_drude_model(1 / wave, opt_b), b)

        # overlap between optical/ir
        weights = _smoothstep(wave, x_min=optir_waves[0], x_max=optir_waves[1], N=1)
        a = jnp.where(
            optir_overlap,
            (1.0 - weights) * compound_polynomial_drude_model(1 / wave, opt_a)
            + weights * self.nirmir_intercept(wave, ir_a),
            a,
        )
        b = jnp.where(
            optir_overlap,
            (1.0 - weights) * compound_polynomial_drude_model(1 / wave, opt_b)
            + weights * PowerLaw1d(x=wave, amplitude=-1.01251, x_0=1.0, alpha=-1.06099),
            b,
        )

        # Ultraviolet
        a = jnp.where(
            uv_mask, FM90(1 / wave, 0.81297, 0.2775, 1.06295, 0.11303, 4.60, 0.99), a
        )
        b = jnp.where(
            uv_mask, FM90(1 / wave, -2.97868, 1.89808, 3.10334, 0.65484, 4.60, 0.99), b
        )

        # overlap between uv/optical
        weights = _smoothstep(wave, x_min=uvopt_waves[0], x_max=uvopt_waves[1], N=1)
        a = jnp.where(
            uvopt_overlap,
            (1.0 - weights)
            * FM90(1 / wave, 0.81297, 0.2775, 1.06295, 0.11303, 4.60, 0.99)
            + weights * compound_polynomial_drude_model(1 / wave, opt_a),
            a,
        )
        b = jnp.where(
            uvopt_overlap,
            (1.0 - weights)
            * FM90(1 / wave, -2.97868, 1.89808, 3.10334, 0.65484, 4.60, 0.99)
            + weights * compound_polynomial_drude_model(1 / wave, opt_b),
            b,
        )

        # return A(x)/A(V)
        return a + b * (1 / self.Rv - 1 / 3.1)

    @staticmethod
    def nirmir_intercept(wave, params):
        """
        Functional form for the NIR/MIR intercept term.
        Based on modifying the G21 shape model to have two power laws instead
        of one with a break wavelength.

        Parameters
        ----------
        wave: float
           expects x as wavelength in micron.
        params: floats
           paramters of function

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]
        """

        # fmt: off
        (scale, alpha, alpha2, swave, swidth,
            sil1_amp, sil1_center, sil1_fwhm, sil1_asym,
            sil2_amp, sil2_center, sil2_fwhm, sil2_asym) = params
        # fmt: on

        # broken powerlaw with a smooth transition
        axav_pow1 = scale * (wave ** (-1.0 * alpha))

        norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
        axav_pow2 = scale * norm_ratio * (wave ** (-1.0 * alpha2))

        # use smoothstep to smoothly transition between the two powerlaws
        weights = _smoothstep(
            wave, x_min=swave - swidth / 2, x_max=swave + swidth / 2, N=1
        )
        axav = axav_pow1 * (1.0 - weights) + axav_pow2 * weights

        # silicate feature drudes
        axav += _modified_drude(wave, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
        axav += _modified_drude(wave, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)

        return axav


# TODO: Implement more jax versions of extinction models from astropy, see https://dust-extinction.readthedocs.io/en/latest/index.html

# Create a dictionary to map model names to classes
Rv_model_dict = {
    "Cardelli89": Cardelli89,
    "Gordon23": Gordon23,
}
