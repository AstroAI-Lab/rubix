from abc import abstractmethod

import equinox
import jax.numpy as jnp
from beartype import beartype as typechecker

# TODO: add runtime type checking for valid x ranges
# can be achieved by using chekify...
# from .helpers import test_valid_x_range
from jaxtyping import Array, Float, jaxtyped

__all__ = [
    "BaseExtModel",
    "BaseExtRvModel",
]  # , "BaseExtRvAfAModel", "BaseExtGrainModel"]


@jaxtyped(typechecker=typechecker)
class BaseExtModel(equinox.Module):
    """
    Base class for dust extinction models.
    """

    wave_range_l: equinox.AbstractVar[float]
    wave_range_h: equinox.AbstractVar[float]

    def __call__(self, wave: Float[Array, "n_wave"]) -> Float[Array, "n_wave"]:
        """
        Evaluate the dust extinction model at the input wavelength for the given model parameters.
        """

        # test_valid_x_range(wave, [self.wave_range_l,self.wave_range_h], self.__class__.__name__)

        return self.evaluate(wave)

    @abstractmethod
    def evaluate(self, wave: Float[Array, "n_wave"]) -> Float[Array, "n_wave"]:
        """
        Abstract function to evaluate the dust extinction model at the input wavelength for the given model parameters.

        Parameters
        ----------
        wave : Float[Array, "n_wave"]
            The wavelength to calculate the dust extinction for.
            The wavelength has to be passed as wavenumber in units of [1/microns].

        Returns
        -------
        Float[Array, "n_wave"]
            The dust extinction as a function of wavenumber.
        """

    @abstractmethod
    def extinguish(self) -> Float[Array, "n_wave"]:
        """
        Abstract function to calculate the dust extinction for a given wavelength as a fraction.

        Parameters
        ----------
        wave : Float[Array, "n_wave"]
            The wavelength to calculate the dust extinction for.
            The wavelength has to be passed as wavenumber in units of [1/microns].

        Returns
        -------
        Float[Array, "n_wave"]
            The fractional extinction as a function of wavenumber.
        """


@jaxtyped(typechecker=typechecker)
class BaseExtRvModel(BaseExtModel):
    """
    Base class for dust extinction models with Rv parameter.
    """

    Rv: equinox.AbstractVar[float]
    Rv_range_l: equinox.AbstractVar[float]  # [Array, "2"]]
    Rv_range_h: equinox.AbstractVar[float]

    """
    The Rv parameter (R(V) = A(V)/E(B-V) total-to-selective extinction) of the dust extinction model and its valid range.
    """

    # def __check_init__(self) -> None:
    #    """
    #    Check if the Rv parameter of the dust extinction model is within Rv_range.

    #    Parameters
    #    ----------
    #    Rv : Float
    #        The Rv parameter of the dust extinction model.

    #    Raises
    #    ------
    #    ValueError
    #        If the Rv parameter is outsied of defined range.
    #    """
    #    #if jnp.logical_or(self.Rv < self.Rv_range[0], self.Rv > self.Rv_range[1]): #not (self.Rv_range[0] <= self.Rv <= self.Rv_range[1]):
    #    #    raise ValueError(
    #    #        "parameter Rv must be between "
    #    #        + str(self.Rv_range[0])
    #    #        + " and "
    #    #        + str(self.Rv_range[1])
    #    #    )
    #    #else:
    #    #    pass

    #    def true_fn(_):
    #        raise ValueError(f"Rv value {self.Rv} is out of range [{self.Rv_range_l},{self.Rv_range_h}]")

    #    def false_fn(_):
    #        return None

    #    condition = jnp.logical_or(self.Rv < self.Rv_range_l, self.Rv > self.Rv_range_h)
    #    jax.debug.print("Condition: {}", condition)

    #    jax.lax.cond(
    #        jnp.logical_or(self.Rv < self.Rv_range_l, self.Rv > self.Rv_range_h),
    #        true_fn,
    #        false_fn,
    #        operand=None
    #    )

    def extinguish(
        self, wave: Float[Array, "n_wave"], Av: Float = None, Ebv: Float = None
    ) -> Float[Array, "n_wave"]:
        """
        Calculate the dust extinction for a given wavelength as a fraction.

        Parameters
        ----------
        wave : Float[Array, "n_wave"]
            The wavelength to calculate the dust extinction for.
            The wavelength has to be passed as wavenumber in units of [1/microns].

        Av : Float
            The visual extinction.
            A(V) value of dust column.
            Note: Av or Ebv must be set.

        Ebv : Float
            The color excess.
            E(B-V) value of dust column.
            Note: Av or Ebv must be set.

        Returns
        -------
        Float[Array, "n_wave"]
            The fractional extinction as a function of wavenumber.
        """
        # get the extinction curve
        axav = self(wave)

        # check that av or ebv is set
        if (Av is None) and (Ebv is None):
            raise ValueError("neither Av or Ebv passed, one of them is required!")

        # if Av is not set and Ebv set, convert to Av
        if Av is None:
            Av = self.Rv * Ebv

        # return fractional extinction
        return jnp.power(10.0, -0.4 * axav * Av)
