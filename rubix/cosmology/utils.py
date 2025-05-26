from typing import Union

import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import jit
from jax.lax import scan
from jaxtyping import Array, Float, jaxtyped


# Source: https://github.com/ArgonneCPAC/dsps/blob/b81bac59e545e2d68ccf698faba078d87cfa2dd8/dsps/utils.py#L247C1-L256C1
@jaxtyped(typechecker=typechecker)
@jit
def _cumtrapz_scan_func(carryover, el):
    """
    Integral helper function, which uses the formula for trapezoidal integration.

    Args:
        carryover (tuple): Tuple of (a, fa, cumtrapz)
        a: current value of x-coordinate
        fa: current value of function at a
        cumtrapz: cumulative sum of trapezoidal integration so far
        el (tuple): Tuple of (b, fb)
        b: next value of x-coordinate
        fb: next value of function at b

    Returns:
        The carryover tuple, which contain (b, fb, cumtrapz)

        The accumulated integral value
    """
    b, fb = el
    a, fa, cumtrapz = carryover
    cumtrapz = cumtrapz + (b - a) * (fb + fa) / 2.0
    carryover = b, fb, cumtrapz
    accumulated = cumtrapz
    return carryover, accumulated


# Source: https://github.com/ArgonneCPAC/dsps/blob/b81bac59e545e2d68ccf698faba078d87cfa2dd8/dsps/utils.py#L278C1-L298C1
@jaxtyped(typechecker=typechecker)
@jit
def trapz(
    xarr: Union[jnp.ndarray, Float[Array, "n"]],
    yarr: Union[jnp.ndarray, Float[Array, "n"]],
) -> jnp.ndarray:
    """
    The function performs the trapezoidal integration using the ``_cumtrapz_scan_func`` helper function.

    Args:
        xarr (ndarray): The x-coordinates of the data points in shape (n, ).
        yarr (ndarray): The y-coordinates of the data points in shape (n, ).

    Returns:
        The result of the trapezoidal integration.

    Example
    -------
    >>> from rubix.cosmology.utils import trapz
    >>> import jax.numpy as jnp

    >>> x = jnp.array([0, 1, 2, 3])
    >>> y = jnp.array([0, 1, 4, 9])
    >>> print(trapz(x, y))
    """
    res_init = xarr[0], yarr[0], 0.0
    scan_data = xarr, yarr
    cumtrapz = scan(_cumtrapz_scan_func, res_init, scan_data)[1]
    return cumtrapz[-1]
