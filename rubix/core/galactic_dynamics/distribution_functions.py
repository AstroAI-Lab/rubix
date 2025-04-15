"""
This module implements distribution functions for a galactic disk. It includes:
  - A DF for the thin disk that accounts for different age cohorts (with varying dispersions)
  - A DF for the thick disk with fixed dispersions
  - A combined total DF that weights the thin and thick disk components
The functions make use of gravitational frequencies derived from the potentials.
"""

import jax
import jax.numpy as jnp
from jax import jit, random
from .potentials import kappa, nu

#Helper functions used by the DFs
def Rc_of_Lz(Lz, v0):
    """
    Computes the guiding center radius given angular momentum Lz and circular velocity v0.
    
    Parameters:
      - Lz: Angular momentum about the z-axis.
      - v0: Characteristic circular velocity.
    
    Returns:
      The guiding center radius Rc
    """
    return Lz / v0

def surface_density_factor(Rc, R0, Rd, Sigma0=1.0):
    """
    Computes the surface density factor for a disk.
    
    Parameters:
      - Rc: Guiding center radius
      - R0: Reference radius
      - Rd: Scale length of the disk
      - Sigma0: Normalization constant (default = 1.0)
      
    Returns:
      The surface density factor
    """
    return Sigma0 * jnp.exp(-(Rc - R0) / Rd)

def df_thin_age_potential(Jr, Jz, Lz, params):
    """
    Calculates the DF for the thin disk with age cohorts.
    
    For each age cohort (discretized into n_age_bins), different dispersions (sigma_r and sigma_z)
    are used. An exponentially declining star formation rate
    is assumed to compute weights for the age cohorts.
    
    Parameters:
      - Jr: Radial action.
      - Jz: Vertical action.
      - Lz: Angular momentum about the z-axis.
      - params: A dictionary of parameters containing:
          * R0, Rd, v0, L0: Characteristic scales
          * tau_m: Maximum age
          * tau1: Age offset
          * beta: Exponent for age-dependent dispersion
          * t0: Timescale for SFR decline
          * n_age_bins: Number of age cohorts
          * sigma_r0: Base radial dispersion for the thin disk (km/s)
          * sigma_z0: Base vertical dispersion for the thin disk (km/s)
    
    Returns:
      The DF value for the thin disk at (Jr, Jz, Lz)
    """
    #Unpack parameters
    R0   = params["R0"]
    Rd   = params["Rd"]
    v0   = params["v0"]
    L0   = params["L0"]
    
    tau_m = params["tau_m"]
    tau1  = params["tau1"]
    beta  = params["beta"]
    t0    = params["t0"]
    n_age_bins = params.get("n_age_bins", 10)
    
    #Create an array of age values and corresponding weights from an exponential SFR
    tau_values = jnp.linspace(0, tau_m, n_age_bins)
    weights = jnp.exp(-tau_values / t0)
    weights = weights / jnp.sum(weights)
    
    #Compute age-dependent dispersions
    sigma_r_ages = params["sigma_r0"] * ((tau_values + tau1) / (tau_m + tau1)) ** beta
    sigma_z_ages = params["sigma_z0"] * ((tau_values + tau1) / (tau_m + tau1)) ** beta
    
    #Compute guiding center radius from Lz and v0
    Rc_val = Rc_of_Lz(Lz, v0)
    
    #Compute frequencies
    kap = kappa(Rc_val)
    nu_val = nu(Rc_val)
    
    Sigma = surface_density_factor(Rc_val, R0, Rd)
    rot_factor = 1.0 + jnp.tanh(Lz / L0)
    
    f_sum = 0.0
    for i in range(n_age_bins):
        sig_r = sigma_r_ages[i]
        sig_z = sigma_z_ages[i]
        f_r = jnp.exp(-kap * Jr / (sig_r**2))
        f_z = jnp.exp(-nu_val * Jz / (sig_z**2))
        f_sum += weights[i] * f_r * f_z
    return Sigma * rot_factor * f_sum

def df_thick_potential(Jr, Jz, Lz, params):
    """
    Calculates the DF for the thick disk.
    
    Uses fixed dispersions for the thick disk.
    
    Parameters:
      - Jr: Radial action.
      - Jz: Vertical action.
      - Lz: Angular momentum about the z-axis.
      - params: A dictionary containing:
            * R0: Reference radius.
            * Rd_thick: Scale length of the thick disk.
            * v0: Characteristic circular velocity.
            * L0_thick: tanh parameter for the thick disk rotation factor.
            * sigma_r0_thick: Base radial dispersion for the thick disk.
            * sigma_z0_thick: Base vertical dispersion for the thick disk.
    
    Returns:
      The DF value for the thick disk at (Jr, Jz, Lz).
    """
    R0 = params["R0"]
    Rd_thick = params["Rd_thick"]
    v0 = params["v0"]
    L0_thick = params["L0_thick"]
    sigma_r0_thick = params["sigma_r0_thick"]
    sigma_z0_thick = params["sigma_z0_thick"]
    
    Rc_val = Rc_of_Lz(Lz, v0)
    
    kap = kappa(Rc_val)
    nu_val = nu(Rc_val)
    
    Sigma = surface_density_factor(Rc_val, R0, Rd_thick)
    rot_factor = 1.0 + jnp.tanh(Lz / L0_thick)
    
    f_r = jnp.exp(-kap * Jr / (sigma_r0_thick**2))
    f_z = jnp.exp(-nu_val * Jz / (sigma_z0_thick**2))
    
    return Sigma * rot_factor * f_r * f_z

def df_total_potential(Jr, Jz, Lz, params):
    """
    Computes the total distribution function as a combination of thin and thick disks.
    
    The total DF is a weighted sum of the thin disk DF (with age cohorts) and the thick disk DF.
    
    Parameters:
      - Jr: Radial action.
      - Jz: Vertical action.
      - Lz: Angular momentum about the z-axis.
      - params: Dictionary containing all required parameters, including:
            * frac_thick: Fraction of the thick disk component (e.g. 0.2 for 20%).
    
    Returns:
      The total DF value at (Jr, Jz, Lz).
    """
    f_thin = df_thin_age_potential(Jr, Jz, Lz, params)
    f_thick = df_thick_potential(Jr, Jz, Lz, params)
    frac_thick = params["frac_thick"]
    return (1 - frac_thick) * f_thin + frac_thick * f_thick


#Function to assign age and metallicity to each star
def assign_age_and_metallicity(key, n_samples, params):
    """
    Assigns an age (tau in Gyr) and a metallicity (Z) to each star.
    
    The age is sampled from an exponential star formation rate (SFR) with timescale t0,
    bounded by a maximum age tau_m. The metallicity is then linearly interpolated between
    Z_max and Z_min based on the age.
    
    Parameters:
      - key: PRNG key for random number generation.
      - n_samples: Number of stars (samples) to generate.
      - params: Dictionary containing:
            * tau_m: Maximum allowed age (Gyr)
            * t0: Timescale of the SFR decline (Gyr)
            * Z_max: Maximum metallicity (for very young stars)
            * Z_min: Minimum metallicity (for old stars)
    
    Returns:
      A tuple (tau, Z) where:
        - tau is an array of ages in Gyr,
        - Z is an array of assigned metallicities
    """
    tau_m = params["tau_m"]
    t0 = params["t0"]
    
    #Inverse Transform Sampling from an exponential distribution
    U = random.uniform(key, shape=(n_samples,))
    tau = -t0 * jnp.log(1 - U * (1 - jnp.exp(-tau_m / t0)))
    
    Z_max = params.get("Z_max", 0.03)
    Z_min = params.get("Z_min", 0.005)
    Z = Z_max - (Z_max - Z_min) * (tau / tau_m)
    
    return tau, Z
