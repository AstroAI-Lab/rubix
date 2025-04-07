from typing import Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# from jax.scipy.special import comb
from scipy.special import (  # whenever there is a jax version of comb, replace this!!!
    comb,
)

# Might come soon according to this github PR: https://github.com/jax-ml/jax/pull/18389



def test_valid_x_range(
    wave: Float[Array, "n"], wave_range: Float[Array, "2"], outname: str
) -> None:  # pragma no cover
    """
    Test if the input wavelength is within the valid range of the model.

    Parameters
    ----------
    wave : Float[Array, "n"]
        The input wavelength to test.

    wave_range : Float[Array, "2"]
        The valid range of the model.

    outname : str
        The name of the model for error message.

    Returns
    -------
    None
    """

    deltacheck = 1e-6  # delta to allow for small numerical issues

    # if jnp.logical_or(
    #    jnp.any(wave <= (wave_range[0] - deltacheck)), jnp.any(wave >= (wave_range[1] + deltacheck))
    # ):
    #    raise ValueError(
    #        "Input wave outside of range defined for "
    #        + outname
    #        + " ["
    #        + str(wave_range[0])
    #        + " <= wave <= "
    #        + str(wave_range[1])
    #        + ", wave has units 1/micron]"
    #    )
    def true_fn(_):
        raise ValueError(
            f"Input wave (min: {jnp.min(wave)}, max: {jnp.max(wave)}) outside of range defined for {outname} [{wave_range[0]} <= wave <= {wave_range[1]}, wave has units 1/micron]."
        )

    def false_fn(_):
        return None

    condition = jnp.logical_or(
        jnp.any(wave <= (wave_range[0] - deltacheck)),
        jnp.any(wave >= (wave_range[1] + deltacheck)),
    )
    jax.lax.cond(condition, true_fn, false_fn, operand=None)


@jaxtyped(typechecker=typechecker)
def _smoothstep(
    x: Float[Array, "n_wave"], x_min: float = 0, x_max: float = 1, N: int = 1
) -> Float[Array, "n_wave"]:
    """
    Smoothstep function. This function is a polynomial approximation to the smoothstep function.
    The smoothstep function is a function commonly used in computer graphics to interpolate smoothly between two values.
    """
    x = jnp.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


@jaxtyped(typechecker=typechecker)
def poly_map_domain(
    oldx: Float[Array, "n"], domain: Tuple[float, float], window: Tuple[float, float]
) -> Float[Array, "n"]:
    """
    Map domain into window by shifting and scaling.

    Parameters
    ----------
    oldx : array
          original coordinates
    domain : tuple of length 2
          function domain
    window : tuple of length 2
          range into which to map the domain
    """
    domain = jnp.array(domain)
    window = jnp.array(window)

    scl = (window[1] - window[0]) / (domain[1] - domain[0])
    off = (window[0] * domain[1] - window[1] * domain[0]) / (domain[1] - domain[0])
    return off + scl * oldx
