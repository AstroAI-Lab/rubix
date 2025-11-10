import jax.numpy as jnp
import numpy as np
import pytest  # type: ignore # noqa

from rubix.telescope.apertures import (
    CIRCULAR_APERTURE,
    HEXAGONAL_APERTURE,
    SQUARE_APERTURE,
)


def test_square_aperture():
    sbin = np.int64(10)
    expected = jnp.ones((sbin, sbin)).flatten()
    result = SQUARE_APERTURE(sbin)
    assert jnp.all(result == expected), "Square aperture mask should be all ones"


def test_circular_aperture():
    sbin = np.int64(10)
    result = CIRCULAR_APERTURE(sbin).reshape(sbin, sbin)
    xcentre, ycentre = sbin / 2 + 0.5, sbin / 2 + 0.5
    x = jnp.tile(jnp.arange(1, sbin + 1), (sbin, 1))
    y = jnp.tile(jnp.arange(sbin, 0, -1), (sbin, 1)).T
    xx, yy = x - xcentre, y - ycentre
    rr = jnp.sqrt(xx**2 + yy**2)
    expected = rr <= sbin / 2
    assert jnp.all(
        result.flatten() == expected.flatten()
    ), "Circular aperture mask is incorrect"


def test_hexagonal_aperture():
    sbin = np.int64(10)
    result = HEXAGONAL_APERTURE(sbin).reshape(sbin, sbin)
    expected = jnp.zeros((sbin, sbin))
    xcentre, ycentre = sbin / 2 + 0.5, sbin / 2 + 0.5
    for x in range(1, sbin + 1):
        for y in range(1, sbin + 1):
            xx = x - xcentre
            yy = y - ycentre
            rr = (
                (2 * (sbin / 4) * (sbin * jnp.sqrt(3) / 4))
                - ((sbin / 4) * jnp.abs(yy))
                - ((sbin * jnp.sqrt(3) / 4) * jnp.abs(xx))
            )
            if (
                (rr >= 0)
                and (jnp.abs(xx) < sbin / 2)
                and (jnp.abs(yy) < (sbin * jnp.sqrt(3) / 4))
            ):
                expected = expected.at[x - 1, y - 1].set(1)
    assert jnp.all(
        result.flatten() == expected.flatten()
    ), "Hexagonal aperture mask is incorrect"
