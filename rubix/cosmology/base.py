from typing import Union

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import jit, lax, vmap
from jaxtyping import Array, Float, jaxtyped

from .utils import trapz

# TODO: maybe change this to load from the config file?
C_SPEED = 2.99792458e8  # m/s
RHO_CRIT0_KPC3_UNITY_H = 277.536627  # multiply by h**2 in cosmology conversion
MPC = 3.08567758149e24  # Mpc in cm
YEAR = 31556925.2  # year in seconds


class BaseCosmology(eqx.Module):
    """
    Handle cosmological calculations using JAX-backed implementations.

    The implementations follow the DSPS ``flat_wcdm`` module
    (https://github.com/ArgonneCPAC/dsps/blob/main/dsps/cosmology/flat_wcdm.py.)
    but are wrapped in a dataclass-style container so that the parameters are stored as ``jax``
    arrays and can be reused safely in other JAX workflows.

    Once initialized with the cosmological parameters, the class can be used to calculate
    various cosmological quantities.

    Args:
        Om0 (float): The present day matter density.
        w0 (float): The present day dark energy equation of state.
        wa (float): The dark energy equation of state parameter.
        h (float): The dimensionless Hubble constant.

    Attributes:
        Om0 (jnp.float32): Stored matter density parameter.
        w0 (jnp.float32): Dark energy equation of state today.
        wa (jnp.float32): Evolution of the dark energy equation of state.
        h (jnp.float32): Dimensionless Hubble constant.

    Example:

            >>> # Create Planck15 cosmology
            >>> from rubix.cosmology import COSMOLOGY
            >>> cosmo = COSMOLOGY(0.3089, -1.0, 0.0, 0.6774)
    """

    Om0: jnp.float32
    w0: jnp.float32
    wa: jnp.float32
    h: jnp.float32

    @jaxtyped(typechecker=typechecker)
    def __init__(self, Om0: float, w0: float, wa: float, h: float):
        self.Om0 = jnp.float32(Om0)
        self.w0 = jnp.float32(w0)
        self.wa = jnp.float32(wa)
        self.h = jnp.float32(h)

    @jit
    @jaxtyped(typechecker=typechecker)
    def scale_factor_to_redshift(
        self, a: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Convert the scale factor to redshift.

        Args:
            a (Union[Float[Array, "..."], float]): Scale factor.

        Returns:
            Float[Array, "..."]: Redshift ``1/a - 1``.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Convert scale factor 0.5 to redshift
                >>> cosmo.scale_factor_to_redshift(jnp.array(0.5))
        """
        z = 1.0 / a - 1.0
        return z

    @jit
    @jaxtyped(typechecker=typechecker)
    def _rho_de_z(self, z: Union[Float[Array, "..."], float]) -> Float[Array, "..."]:
        a = 1.0 / (1.0 + z)
        de_z = a ** (-3.0 * (1.0 + self.w0 + self.wa)) * lax.exp(
            -3.0 * self.wa * (1.0 - a)
        )
        return de_z

    @jit
    @jaxtyped(typechecker=typechecker)
    def _Ez(self, z: Union[Float[Array, "..."], float]) -> Float[Array, "..."]:
        zp1 = 1.0 + z
        Ode0 = 1.0 - self.Om0
        t = self.Om0 * zp1**3 + Ode0 * self._rho_de_z(z)
        E = jnp.sqrt(t)
        return E

    @jit
    @jaxtyped(typechecker=typechecker)
    def _integrand_oneOverEz(
        self, z: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        return 1 / self._Ez(z)

    @jit
    @jaxtyped(typechecker=typechecker)
    def comoving_distance_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Calculate the comoving distance to the input redshift.

        Args:
            redshift (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Comoving distance in Mpc.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Calculate comoving distance to redshift 0.5
                >>> cosmo.comoving_distance_to_z(0.5)
        """
        z_table = jnp.linspace(0, redshift, 256)
        integrand = self._integrand_oneOverEz(z_table)
        return trapz(z_table, integrand) * C_SPEED * 1e-5 / self.h

    @jit
    @jaxtyped(typechecker=typechecker)
    def luminosity_distance_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Compute the luminosity distance at the requested redshift.

        Args:
            redshift (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Luminosity distance in Mpc.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Compute the luminosity distance to redshift 0.5
                >>> cosmo.luminosity_distance_to_z(0.5)
        """
        return self.comoving_distance_to_z(redshift) * (1 + redshift)

    @jit
    @jaxtyped(typechecker=typechecker)
    def angular_diameter_distance_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Compute the angular diameter distance for the given redshift.

        Args:
            redshift (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Angular diameter distance in Mpc.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Compute the angular diameter distance to redshift 0.5
                >>> cosmo.angular_diameter_distance_to_z(0.5)
        """
        return self.comoving_distance_to_z(redshift) / (1 + redshift)

    @jit
    @jaxtyped(typechecker=typechecker)
    def distance_modulus_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Compute the distance modulus for the requested redshift.

        Args:
            redshift (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Distance modulus.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Compute the distance modulus to redshift 0.5
                >>> cosmo.distance_modulus_to_z(0.5)
        """
        d_lum = self.luminosity_distance_to_z(redshift)
        mu = 5.0 * jnp.log10(d_lum * 1e5)
        return mu

    @jit
    @jaxtyped(typechecker=typechecker)
    def _hubble_time(self, z: Union[Float[Array, "..."], float]) -> Float[Array, "..."]:
        """
        Calculate the Hubble time at the given redshift.

        Args:
            z (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Hubble time in seconds.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Calculate the Hubble time at redshift 0.5
                >>> cosmo._hubble_time(0.5)
        """
        E0 = self._Ez(z)
        htime = 1e-16 * MPC / YEAR / self.h / E0
        return htime

    @jit
    @jaxtyped(typechecker=typechecker)
    def lookback_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Calculate the lookback time to the requested redshift.

        Args:
            redshift (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Lookback time in seconds.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Calculate the lookback time to redshift 0.5
                >>> cosmo.lookback_to_z(0.5)
        """
        z_table = jnp.linspace(0, redshift, 512)
        integrand = 1 / self._Ez(z_table) / (1 + z_table)
        res = trapz(z_table, integrand)
        th = self._hubble_time(0.0)
        return th * res

    @jit
    @jaxtyped(typechecker=typechecker)
    def age_at_z0(self) -> Float[Array, "..."]:
        """
        The function calculates the age of the universe at redshift 0.

        Returns:
            The age of the universe at redshift 0 (float).

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Calculate the age of the universe at redshift 0
                >>> cosmo.age_at_z0()
        """
        z_table = jnp.logspace(0, 3, 512) - 1.0
        integrand = 1 / self._Ez(z_table) / (1 + z_table)
        res = trapz(z_table, integrand)
        th = self._hubble_time(0.0)
        return th * res

    @jit
    @jaxtyped(typechecker=typechecker)
    def _age_at_z_kern(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        t0 = self.age_at_z0()
        tlook = self.lookback_to_z(redshift)
        return t0 - tlook

    def _age_at_z_vmap(self):
        return jit(vmap(self._age_at_z_kern))

    @jit
    @jaxtyped(typechecker=typechecker)
    def age_at_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Return the age of the universe at the provided redshift.

        Args:
            redshift (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Age in seconds.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Calculate the age of the universe at redshift 0.5
                >>> cosmo.age_at_z(0.5)
        """
        fun = self._age_at_z_vmap()
        return fun(jnp.atleast_1d(redshift))

    @jit
    @jaxtyped(typechecker=typechecker)
    def angular_scale(
        self, z: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Angular scale in kpc/arcsec at redshift ``z``.

        Args:
            z (Union[Float[Array, "..."], float]): Redshift value(s).

        Returns:
            Float[Array, "..."]: Angular scale in kpc/arcsec.

        Example:

                >>> from rubix.cosmology import PLANCK15 as cosmo
                >>> # Calculate the angular scale at redshift 0.5
                >>> cosmo.angular_scale(0.5)
        """
        # Angular scale in kpc/arcsec at redshift z.
        D_A = self.angular_diameter_distance_to_z(z)  # in Mpc
        scale = D_A * (jnp.pi / (180 * 60 * 60)) * 1e3  # in kpc/arcsec
        return scale
