from typing import Tuple

import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from .helpers import poly_map_domain

# TODO: add runtime type checking for valid x ranges
# can be achieved by using chekify...
# from .dust_baseclasses import test_valid_x_range


# TODO: Implement functions as classes?


@jaxtyped(typechecker=typechecker)
def PowerLaw1d(
    x: Float[Array, "n_wave"], amplitude: float, x_0: float, alpha: float
) -> Float[Array, "n_wave"]:
    """
    Calculate a power law function.
    Function inspired by astropy.modeling.functional_models.PowerLaw1D.

    Parameters
    ----------
    x : Float[Array, "n_wave"]
        Input array.
    amplitude : float
        Amplitude of the power law.
    x_0 : float
        Reference x value.
    alpha : float
        Power law index.

    Returns
    -------
    Float[Array, "n_wave"]
        Output array after applying the power law.

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude`` and :math:`\\alpha` for ``alpha``):

        .. math:: f(x) = A (x / x_0) ^ {-\\alpha}
    """
    xx = x / x_0
    return amplitude * xx ** (-alpha)


def Polynomial1d(
    x: Float[Array, "n"],
    coeffs: Float[Array, "m"],
    domain: Tuple[float, float] = (-1.0, 1.0),
    window: Tuple[float, float] = (-1.0, 1.0),
) -> Float[Array, "n"]:
    r"""
    Evaluate a 1D polynomial model defined as

    .. math::

        P = \sum_{i=0}^{i=n}C_{i} * x^{i}

    This function inspired by astropy.modelling.polynomial.Polynomial1D.

    Parameters
    ----------
    x : ndarray
        Input values.
    coeffs : ndarray
        Coefficients of the polynomial, ordered from the constant term to the highest degree term.
    domain : tuple, optional
        Domain of the input values. Default is (-1, 1).
    window : tuple, optional
        Window to which the domain is mapped. Default is (-1, 1).

    Returns
    -------
    result : ndarray
        Evaluated polynomial values.
    """

    def horner(x: Float[Array, "n"], coeffs: Float[Array, "m"]) -> Float[Array, "n"]:
        """
        Evaluate polynomial using Horner's method.
        """
        if len(coeffs) == 1:
            return coeffs[-1] * jnp.ones_like(x)
        c0 = coeffs[-1]
        for i in range(2, len(coeffs) + 1):
            c0 = coeffs[-i] + c0 * x
        return c0

    if domain is not None:
        x = poly_map_domain(x, domain, window)
    return horner(x, coeffs)


@jaxtyped(typechecker=typechecker)
def Drude1d(
    x: Float[Array, "n"], amplitude: float = 1.0, x_0: float = 1.0, fwhm: float = 1.0
):
    r"""
    Evaluate the Drude model function.
    This function is inspired by astropy.modeling.functional_models.Drude1D.

    Model formula:

        .. math:: f(x) = A \\frac{(fwhm/x_0)^2}{((x/x_0 - x_0/x)^2 + (fwhm/x_0)^2}

    Parameters
    ----------
    x : ndarray
        Input values.
    amplitude : float, optional
        Peak value. Default is 1.0.
    x_0 : float, optional
        Position of the peak. Default is 1.0.
    fwhm : float, optional
        Full width at half maximum. Default is 1.0.

    Returns
    -------
    result : ndarray
        Evaluated Drude model values.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from rubix.spectra.dust.extinction_models import Drude1D

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(7.5 , 12.5 , 0.1)

        dmodel = Drude1D(amplitude=1.0, fwhm=1.0, x_0=10.0)
        ax.plot(x, dmodel(x))

        ax.set_xlabel('x')
        ax.set_ylabel('F(x)')

        plt.show()
    """
    if x_0 == 0:
        raise ValueError("0 is not an allowed value for x_0")
    return (
        amplitude * ((fwhm / x_0) ** 2) / ((x / x_0 - x_0 / x) ** 2 + (fwhm / x_0) ** 2)
    )


@jaxtyped(typechecker=typechecker)
def _modified_drude(
    x: Float[Array, "n"], scale: float, x_o: float, gamma_o: float, asym: float
) -> Float[Array, "n"]:
    """
    Modified Drude function to have a variable asymmetry.  Drude profiles
    are intrinsically asymmetric with the asymmetry fixed by specific central
    wavelength and width.  This modified Drude introduces an asymmetry
    parameter that allows for variable asymmetry at fixed central wavelength
    and width.

    Parameters
    ----------
    x : ndarray
        input wavelengths

    scale : float
        central amplitude

    x_o : float
        central wavelength

    gamma_o : float
        full-width-half-maximum of profile

    asym : float
        asymmetry where a value of 0 results in a standard Drude profile

    Returns
    -------
    y : ndarray
        output profile
    """
    gamma = 2.0 * gamma_o / (1.0 + jnp.exp(asym * (x - x_o)))
    y = scale * ((gamma / x_o) ** 2) / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)

    return y


@jaxtyped(typechecker=typechecker)
def FM90(
    x: Float[Array, "n"],
    C1: float = 0.10,
    C2: float = 0.70,
    C3: float = 3.23,
    C4: float = 0.41,
    xo: float = 4.59,
    gamma: float = 0.95,
) -> Float[Array, "n"]:
    r"""
    Fitzpatrick & Massa (1990) 6 parameter ultraviolet shape model

    Parameters
    ----------
    x: float
        wavenumber x in units of [1/micron]

    C1: float
       y-intercept of linear term

    C2: float
       slope of liner term

    C3: float
       strength of "2175 A" bump (true amplitude is C3/gamma^2)

    C4: float
       amplitude of FUV rise

    xo: float
       centroid of "2175 A" bump

    gamma: float
       width of "2175 A" bump

    Returns
        -------
        exvebv: np array (float)
            E(x-V)/E(B-V) extinction curve [mag]

    Raises
    ------
        ValueError
           Input x values outside of defined range

    Notes
    -----
    From Fitzpatrick & Massa (1990, ApJS, 72, 163)

    Only applicable at UV wavelengths

    Example showing a FM90 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from rubix.spectra.dust.extinction_models import FM90

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(3.8,8.6,0.1)/u.micron

        ext_model = FM90(x=x)
        ax.plot(x,ext_model,label='total')

        ext_model = FM90(x=x, C3=0.0, C4=0.0)
        ax.plot(x,ext_model,label='linear term')

        ext_model = FM90(x=x, C1=0.0, C2=0.0, C4=0.0)
        ax.plot(x,ext_model,label='bump term')

        ext_model = FM90(x=x, C1=0.0, C2=0.0, C3=0.0)
        ax.plot(x,ext_model,label='FUV rise term')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$E(\lambda - V)/E(B - V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.12, 0.15, 0.2, 0.3])
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

    # Define bounds based on Gordon et al. (2024) results
    bounds = {
        "C1": (-10.0, 5.0),
        "C2": (-0.1, 5.0),
        "C3": (-1.0, 6.0),
        "C4": (-0.5, 1.5),
        "xo": (4.5, 4.9),
        "gamma": (0.6, 1.7),
    }

    # Check if parameters are within bounds
    if not (bounds["C1"][0] <= C1 <= bounds["C1"][1]):
        raise ValueError(f"C1 is out of bounds: {C1}")
    if not (bounds["C2"][0] <= C2 <= bounds["C2"][1]):
        raise ValueError(f"C2 is out of bounds: {C2}")
    if not (bounds["C3"][0] <= C3 <= bounds["C3"][1]):
        raise ValueError(f"C3 is out of bounds: {C3}")
    if not (bounds["C4"][0] <= C4 <= bounds["C4"][1]):
        raise ValueError(f"C4 is out of bounds: {C4}")
    if not (bounds["xo"][0] <= xo <= bounds["xo"][1]):
        raise ValueError(f"xo is out of bounds: {xo}")
    if not (bounds["gamma"][0] <= gamma <= bounds["gamma"][1]):
        raise ValueError(f"gamma is out of bounds: {gamma}")

    x_range = [1 / 0.35, 1 / 0.09]
    # test_valid_x_range(x, x_range, "FM90")

    # linear term
    exvebv = C1 + C2 * x

    # bump term
    x2 = x**2
    exvebv += C3 * (x2 / ((x2 - xo**2) ** 2 + x2 * (gamma**2)))

    # FUV rise term
    fnuv_mask = x >= 5.9
    y = jnp.where(fnuv_mask, x - 5.9, 0.0)
    exvebv = jnp.where(
        fnuv_mask, exvebv + C4 * (0.5392 * (y**2) + 0.05644 * (y**3)), exvebv
    )

    # return E(x-V)/E(B-V)
    return exvebv
