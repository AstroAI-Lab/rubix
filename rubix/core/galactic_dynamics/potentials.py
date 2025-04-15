"""
This module defines the basic gravitational potential functions for the galactic disk, bulge, 
and halo, as well as functions to derive important frequencies (the circular velocity v_c, 
the epicyclic frequency kappa, and the vertical oscillation frequency nu) from the total potential.
"""

import jax
import jax.numpy as jnp
from jax import grad

#Gravitational constant
G = 4.30091e-6

#Potentials
def phi_disk(R, z, M_disk=1e11, a=6.5, b=0.26):
    """
    Computes the gravitational potential of the galactic disk
    
    Parameters:
      - R, z: Radial and vertical distances (in kpc)
      - M_disk: Mass of the disk (in solar masses)
      - a, b: Scale parameters
      
    Returns:
      The potential value at (R, z)
    """
    return -G * M_disk / jnp.sqrt(R**2 + (a + jnp.sqrt(z**2 + b**2))**2)

def phi_bulge(R, z, M_bulge=1e10, c=0.7):
    """
    Computes the gravitational potential of the galactic bulge.
    
    Parameters:
      - R, z: Radial and vertical distances (in kpc)
      - M_bulge: Mass of the bulge (in solar masses)
      - c: Scale parameter
      
    Returns:
      The potential value at (R, z)
    """
    r = jnp.sqrt(R**2 + z**2)
    return -G * M_bulge / (r + c)

def phi_halo_NFW(R, z, M_halo=1e12, r_s=20.0):
    """
    Computes the NFW potential for the halo
    
    Parameters:
      - R, z: Radial and vertical distances (in kpc)
      - M_halo: Halo mass (in solar masses)
      - r_s: Scale radius of the halo (in kpc)
      
    Returns:
      The potential value at (R, z)
    """
    r = jnp.sqrt(R**2 + z**2)
    #Avoid division by zero
    return -G * M_halo / jnp.maximum(r, 1e-3) * jnp.log(1 + r / r_s)

def phi_total(R, z, **kwargs):
    """
    Computes the total gravitational potential as the sum of the disk, bulge, and halo potentials
    
    Additional parameters (such as masses and scale parameters) can be passed via kwargs.
    
    Parameters:
      - R, z: Radial and vertical distances (in kpc)
      
    Returns:
      The total potential at (R, z)
    """
    return phi_disk(R, z, **kwargs) + phi_bulge(R, z, **kwargs) + phi_halo_NFW(R, z, **kwargs)

#Frequencies
def v_c(R):
    """
    Computes the circular velocity v_c at radius R

    Parameters:
      - R: Radius (in kpc)
      
    Returns:
      Circular velocity at R (in km/s).
    """
    phi_R = grad(lambda R: phi_total(R, 0.0))
    expr = R * phi_R(R)
    expr = jnp.maximum(expr, 1e-6) #avoid division by zero
    return jnp.sqrt(expr)

def kappa(R):
    """
    Computes the epicyclic frequency κ
    
    Parameters:
      - R: Radius (in kpc)
      
    Returns:
      The epicyclic frequency at R.
    """
    Omega = v_c(R) / R
    dOmega_dR = grad(lambda R: v_c(R)/R)(R)
    expr = R * 2 * Omega * dOmega_dR + 4 * Omega**2
    expr = jnp.maximum(expr, 1e-6)
    return jnp.sqrt(expr)

def nu(R):
    """
    Computes the vertical oscillation frequency ν
    
    Parameters:
      - R: Radius (in kpc)
      
    Returns:
      The vertical oscillation frequency at R.
    """
    phi_z = lambda z: phi_total(R, z)
    phi_zz = grad(lambda z: grad(phi_z)(z))
    val = jnp.abs(phi_zz(0.0))
    val = jnp.maximum(val, 1e-6)
    return jnp.sqrt(val)
