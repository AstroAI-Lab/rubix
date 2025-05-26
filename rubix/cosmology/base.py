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
    """Class to handle cosmological calculations.

    The methods in this class are mainly taken from https://github.com/ArgonneCPAC/dsps/blob/main/dsps/cosmology/flat_wcdm.py.
    Here they are wrapped in a class to be used in JAX.

    Once initialized with the cosmological parameters, the class can be used to calculate various cosmological quantities.

    Parameters
    ----------
    Om0 : float
        The present day matter density.
    w0 : float
        The present day dark energy equation of state.
    wa : float
        The dark energy equation of state parameter.
    h : float
        The Hubble constant.

    Returns
    -------
        A Cosmology instance.

    Example
    --------
    >>> # Create Planck15 cosmology
    >>> cosmo = Cosmology(0.3089, -1.0, 0.0, 0.6774)
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

    @jaxtyped(typechecker=typechecker)
    @jit
    def scale_factor_to_redshift(
        self, a: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        The function converts the scale factor to redshift.

        Args:
            a (float): The scale factor.

        Returns:
            The redshift (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Convert scale factor 0.5 to redshift
        >>> cosmo.scale_factor_to_redshift(jnp.array(0.5))
        """
        z = 1.0 / a - 1.0
        return z

    @jaxtyped(typechecker=typechecker)
    @jit
    def _rho_de_z(self, z: Union[Float[Array, "..."], float]) -> Float[Array, "..."]:
        a = 1.0 / (1.0 + z)
        de_z = a ** (-3.0 * (1.0 + self.w0 + self.wa)) * lax.exp(
            -3.0 * self.wa * (1.0 - a)
        )
        return de_z

    @jaxtyped(typechecker=typechecker)
    @jit
    def _Ez(self, z: Union[Float[Array, "..."], float]) -> Float[Array, "..."]:
        zp1 = 1.0 + z
        Ode0 = 1.0 - self.Om0
        t = self.Om0 * zp1**3 + Ode0 * self._rho_de_z(z)
        E = jnp.sqrt(t)
        return E

    @jaxtyped(typechecker=typechecker)
    @jit
    def _integrand_oneOverEz(
        self, z: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        return 1 / self._Ez(z)

    @jaxtyped(typechecker=typechecker)
    @jit
    def comoving_distance_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        The function calculates the comoving distance to a given redshift.

        Args:
            redshift (float): The redshift.

        Returns:
            The comoving distance to a given redshift (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the comoving distance to redshift 0.5
        >>> cosmo.comoving_distance_to_z(0.5)
        """
        z_table = jnp.linspace(0, redshift, 256)
        integrand = self._integrand_oneOverEz(z_table)
        return trapz(z_table, integrand) * C_SPEED * 1e-5 / self.h

    @jaxtyped(typechecker=typechecker)
    @jit
    def luminosity_distance_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        The function calculates the luminosity distance to a given redshift.

        Args:
            redshift (float): The redshift.

        Returns:
            The luminosity distance to the redshift (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the luminosity distance to redshift 0.5
        >>> cosmo.luminosity_distance_to_z(0.5)
        """
        return self.comoving_distance_to_z(redshift) * (1 + redshift)

    @jaxtyped(typechecker=typechecker)
    @jit
    def angular_diameter_distance_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        The function calculates the angular diameter distance to a given redshift.

        Args:
            redshift (float): The redshift.

        Returns:
            The angular diameter distance to the redshift (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the angular diameter distance to redshift 0.5
        >>> cosmo.angular_diameter_distance_to_z(0.5)
        """
        return self.comoving_distance_to_z(redshift) / (1 + redshift)

    @jaxtyped(typechecker=typechecker)
    @jit
    def distance_modulus_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        The function calculates the distance modulus to a given redshift.

        Args:
            redshift (float): The redshift.

        Returns:
            The distance modulus to the redshift (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the distance modulus to redshift 0.5
        >>> cosmo.distance_modulus_to_z(0.5)
        """
        d_lum = self.luminosity_distance_to_z(redshift)
        mu = 5.0 * jnp.log10(d_lum * 1e5)
        return mu

    @jaxtyped(typechecker=typechecker)
    @jit
    def _hubble_time(self, z: Union[Float[Array, "..."], float]) -> Float[Array, "..."]:
        E0 = self._Ez(z)
        htime = 1e-16 * MPC / YEAR / self.h / E0
        return htime

    @jaxtyped(typechecker=typechecker)
    @jit
    def lookback_to_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        The function calculates the lookback time to a given redshift.

        Args:
            redshift (float): The redshift.

        Returns:
            The lookback time to the redshift (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the lookback time to redshift 0.5
        >>> cosmo.lookback_to_z(0.5)
        """
        z_table = jnp.linspace(0, redshift, 512)
        integrand = 1 / self._Ez(z_table) / (1 + z_table)
        res = trapz(z_table, integrand)
        th = self._hubble_time(0.0)
        return th * res

    @jaxtyped(typechecker=typechecker)
    @jit
    def age_at_z0(self) -> Float[Array, "..."]:
        """
        The function calculates the age of the universe at redshift 0.

        Returns:
            The age of the universe at redshift 0 (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the age of the universe at redshift 0
        >>> cosmo.age_at_z0()
        """
        z_table = jnp.logspace(0, 3, 512) - 1.0
        integrand = 1 / self._Ez(z_table) / (1 + z_table)
        res = trapz(z_table, integrand)
        th = self._hubble_time(0.0)
        return th * res

    @jaxtyped(typechecker=typechecker)
    @jit
    def _age_at_z_kern(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        t0 = self.age_at_z0()
        tlook = self.lookback_to_z(redshift)
        return t0 - tlook

    @jaxtyped(typechecker=typechecker)
    @jit
    def age_at_z(
        self, redshift: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        The function calculates the age of the universe at a given redshift.

        Args:
            redshift (float): The redshift.

        Returns:
            The age of the universe at the redshift (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the age of the universe at redshift 0.5
        >>> cosmo.age_at_z(0.5)
        """
        fun = self._age_at_z_vmap()
        return fun(jnp.atleast_1d(redshift))

    def _age_at_z_vmap(self):
        return jit(vmap(self._age_at_z_kern))

    @jaxtyped(typechecker=typechecker)
    @jit
    def angular_scale(
        self, z: Union[Float[Array, "..."], float]
    ) -> Float[Array, "..."]:
        """
        Angular scale in kpc/arcsec at redshift z.

        Args:
            z (float): Redshift

        Returns:
            Angular scale in kpc/arcsec at redshift z (float).

        Example
        --------
        >>> from rubix.cosmology import PLANCK15 as cosmo
        >>> # Calculate the angular scale at redshift 0.5
        >>> cosmo.angular_scale(0.5)
        """
        # Angular scale in kpc/arcsec at redshift z.
        D_A = self.angular_diameter_distance_to_z(z)  # in Mpc
        scale = D_A * (jnp.pi / (180 * 60 * 60)) * 1e3  # in kpc/arcsec
        return scale

    """
    I dont think we need this currently, but keeping it here for reference
    @jit
    def rho_crit(self, redshift):
        rho_crit0 = RHO_CRIT0_KPC3_UNITY_H * self.h * self.h
        rho_crit = rho_crit0 * self._Ez(redshift) ** 2
        return rho_crit

    @jit
    def _integrand_oneOverEz1pz(self, z):
        return 1.0 / self._Ez(z) / (1.0 + z)

    @jit
    def _Om_at_z(self, z):
        E = self._Ez(z)
        return self.Om0 * (1.0 + z) ** 3 / E / E




    @jit
    def _delta_vir(self, z):
        x = self._Om(z) - 1.0
        Delta = 18 * jnp.pi**2 + 82.0 * x - 39.0 * x**2
        return Delta

    @jit
    def virial_dynamical_time(self, redshift):
        delta = self._delta_vir(redshift)
        t_cross = 2**1.5 * self._hubble_time(redshift) * delta**-0.5
        return t_cross

"""
