from typing import Union

import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Tuple
from jax import jit
from jax.lax import scan
from jaxtyping import Array, Float, jaxtyped

CarryoverState = Tuple[
    Float[Array, "..."],
    Float[Array, "..."],
    Float[Array, "..."],
]
ElementPair = Tuple[Float[Array, "..."], Float[Array, "..."]]


# Source:
#   https://github.com/ArgonneCPAC/dsps/blob/b81bac59e545e2d68ccf698faba078d87cfa2dd8/dsps/utils.py#L247C1-L256C1
@jit
@jaxtyped(typechecker=typechecker)
def _cumtrapz_scan_func(
    carryover: CarryoverState,
    el: ElementPair,
) -> Tuple[
    CarryoverState,
    Float[Array, "..."],
]:
    """
    Integral helper that implements the trapezoidal rule step.

    Args:
        carryover (CarryoverState): Current ``(a, fa, cumtrapz)`` state values.
        el (ElementPair): Next ``(b, fb)`` pair.

    Note:
        a: current value of x-coordinate.
        fa: current value of function at a.
        cumtrapz: cumulative sum of trapezoidal integration so far.
        b: next value of x-coordinate.
        fb: next value of function at b.

    Returns:
        Tuple[CarryoverState, Float[Array, "..."]]:
            Updated carryover and accumulated integral.
    """
    b, fb = el
    a, fa, cumtrapz = carryover
    cumtrapz = cumtrapz + (b - a) * (fb + fa) / 2.0
    carryover = b, fb, cumtrapz
    accumulated = cumtrapz
    return carryover, accumulated


# Source:
#   https://github.com/ArgonneCPAC/dsps/blob/b81bac59e545e2d68ccf698faba078d87cfa2dd8/dsps/utils.py#L278C1-L298C1
@jit
@jaxtyped(typechecker=typechecker)
def trapz(
    xarr: Union[jnp.ndarray, Float[Array, "..."]],
    yarr: Union[jnp.ndarray, Float[Array, "..."]],
) -> jnp.ndarray:
    """
    Perform trapezoidal integration using ``_cumtrapz_scan_func``.

    Args:
        xarr (Union[jnp.ndarray, Float[Array, "..."]]): The x-coordinates.
        yarr (Union[jnp.ndarray, Float[Array, "..."]]): The y-values.

    Returns:
        jnp.ndarray: Scalar results collected from the scan.

    Example:

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
