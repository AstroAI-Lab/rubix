import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Final, Tuple
from jaxtyping import Array, Float, jaxtyped

# from jax.scipy.special import comb
# whenever there is a jax version of comb, replace this!!!
from scipy.special import comb

# Might come soon according to this github PR:
# https://github.com/jax-ml/jax/pull/18389

N_AXIS: Final[str] = "n"
N_WAVE_AXIS: Final[str] = "n_wave"
N_PARAMS_AXIS: Final[str] = "m"


def test_valid_x_range(
    wave: Float[Array, N_AXIS],
    wave_range: Float[Array, "2"],
    outname: str,
) -> None:  # pragma no cover
    """
    Ensure the input wavelength stays inside the configured range.

    Args:
        wave (Float[Array, N_AXIS]): The input wavelengths to test.
        wave_range (Float[Array, "2"]): The valid bounds for the model.
        outname (str): The model name used in the error message.

    This helper raises a ``ValueError`` if the wavelengths fall outside
    the requested range.
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
            "Input wave (min: "
            f"{jnp.min(wave)}, max: {jnp.max(wave)}) outside of "
            f"range defined for {outname} [{wave_range[0]} <= wave <= "
            f"{wave_range[1]}, wave has units 1/micron]."
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
    x: Float[Array, N_WAVE_AXIS],
    x_min: float = 0,
    x_max: float = 1,
    N: int = 1,
) -> Float[Array, N_WAVE_AXIS]:
    """
    Smoothstep interpolation defined by a polynomial between 0 and 1.
    The smoothstep function is a function commonly used in computer graphics to interpolate smoothly between two values.

    Args:
        x (Float[Array, N_WAVE_AXIS]): Input values in the unit interval.
        x_min (float): Lower bound of the input domain.
        x_max (float): Upper bound of the input domain.
        N (int): Number of times to apply the base smoothstep polynomial.

    Returns:
        Float[Array, N_WAVE_AXIS]: Smoothly interpolated values.
    """
    x = jnp.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


@jaxtyped(typechecker=typechecker)
def poly_map_domain(
    oldx: Float[Array, N_AXIS],
    domain: Tuple[float, float],
    window: Tuple[float, float],
) -> Float[Array, N_AXIS]:
    """
    Map domain coordinates into a target window via an affine transform.

    Args:
        oldx (Float[Array, N_AXIS]): Original coordinates.
        domain (Tuple[float, float]): Domain of the input values.
        window (Tuple[float, float]): Window into which to map the domain.

    Returns:
        Float[Array, N_AXIS]: Transformed coordinates.
    """
    domain = jnp.array(domain)
    window = jnp.array(window)

    scl = (window[1] - window[0]) / (domain[1] - domain[0])
    off = (window[0] * domain[1] - window[1] * domain[0]) / (domain[1] - domain[0])
    return off + scl * oldx
