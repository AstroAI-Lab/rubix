"""
This module provides a function to map actions to phase-space
coordinates for stars. The conversion uses the guiding center 
approximation and incorporates frequencies derived from the gravitational potentials.

The function is implemented with JAX and uses random sampling of the angles to 
generate a distribution of positions and velocities.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from .potentials import kappa, nu

def actions_to_phase_space(Jr, Jz, Lz, params, key):
    """
    Converts actions into phase-space coordinates.
    
    Parameters:
      - Jr: Radial action
      - Jz: Vertical action
      - Lz: Angular momentum along z
      - params: Dictionary of parameters that must include at least:
          * v0: Circular velocity (km/s)
      - key: PRNGKey for random sampling
    
    Returns:
      A tuple (x, y, z, v_x, v_y, v_z) representing the star's Cartesian coordinates
      and velocity components.
    """
    v0 = params["v0"]
    #Guiding center radius from angular momentum
    epsilon = 0.1 
    Rc_val = jnp.maximum(Lz / params["v0"], epsilon)


    #Obtain dynamical frequencies from potentials
    kap = kappa(Rc_val)
    nu_val = nu(Rc_val)

    #Amplitudes for oscillations in the radial and vertical directions
    A_R = jnp.sqrt(2.0 * Jr / kap)
    A_z = jnp.sqrt(2.0 * Jz / nu_val)
    
    #Sample random angles uniformly in [0, 2π]
    key, subkey = random.split(key)
    theta_R = random.uniform(subkey, minval=0.0, maxval=2 * jnp.pi)
    key, subkey = random.split(key)
    theta_z = random.uniform(subkey, minval=0.0, maxval=2 * jnp.pi)
    key, subkey = random.split(key)
    theta_phi = random.uniform(subkey, minval=0.0, maxval=2 * jnp.pi)
    
    #Compute the instantaneous radial position:
    R = Rc_val + A_R * jnp.cos(theta_R)
    #Azimuthal angle is given by the random angle theta_phi.
    phi = theta_phi
    #Vertical position: oscillates with amplitude A_z
    z = A_z * jnp.cos(theta_z)
    
    #Transform from cylindrical to Cartesian coordinates
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    

    #Radial velocity due to epicyclic motion
    v_R = -A_R * kap * jnp.sin(theta_R)
    #Azimuthal velocity: for a simple model, we use Lz/R 
    v_phi = Lz / R
    #Vertical velocity due to vertical oscillation
    v_z = -A_z * nu_val * jnp.sin(theta_z)
    
    #Transform the radial and azimuthal velocities into Cartesian coordinates
    v_x = v_R * jnp.cos(phi) - v_phi * jnp.sin(phi)
    v_y = v_R * jnp.sin(phi) + v_phi * jnp.cos(phi)
    
    return x, y, z, v_x, v_y, v_z

#Helper function to map an array of action triplets to phase space coordinates
def map_actions_to_phase_space(candidates, params, key):
    """
    Applies actions_to_phase_space to a batch of action candidates.
    
    Parameters:
      - candidates: A JAX array of shape (N, 3) where each row is (Jr, Jz, Lz)
      - params: Dictionary of parameters
      - key: PRNGKey for random sampling
      
    Returns:
      A JAX array of shape (N, 6) where each row is (x, y, z, v_x, v_y, v_z).
    """
    num = candidates.shape[0]
    keys = random.split(key, num)
    mapped = vmap(lambda cand, k: actions_to_phase_space(cand[0], cand[1], cand[2], params, k))(
        candidates, keys)
    
    #mapped is a tuple of 6 arrays, each of shape (N,)
    #we use jnp.column_stack to combine them into a (N, 6) array
    return jnp.column_stack(mapped)
