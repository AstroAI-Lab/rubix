"""
This module implements a differentiable sampling pipeline using a soft acceptance approach.
A sigmoid function is applied to assign acceptance weights to candidate actions generated from the total DF. These weighted candidate actions
can then be transformed into phase-space coordinates.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.nn import sigmoid
from .distribution_functions import df_total_potential

def soft_acceptance(df_vals, rand_vals, envelope_max, tau=0.01):
    """
    Computes a soft acceptance mask using a sigmoid function.

    Parameters:
      - df_vals: An array of DF values for each candidate.
      - rand_vals: An array of uniformly generated random values between 0 and 1.
      - envelope_max: The normalization factor used in the hard acceptance comparison.
      - tau: The temperature parameter controlling the softness of the transition (smaller values result in a harder transition).

    Returns:
      An array of values between 0 and 1 representing the "softness" of the acceptance, 
      where values close to 1 indicate high acceptance and values close to 0 indicate low acceptance.
    """
    return sigmoid((df_vals / envelope_max - rand_vals) / tau)

def sample_df_potential(key, params, n_candidates, envelope_max, tau=0.01):
    """
    Differentiable version of the sampling pipeline.
    
    Instead of a hard acceptance/rejection, returns weighted candidate actions.
    
    Parameters:
      - key: A JAX PRNGKey for random number generation.
      - params: Dictionary with DF parameters.
      - n_candidates: Number of candidate samples to generate.
      - envelope_max: Normalization factor.
      - tau: Temperature parameter for soft acceptance.
      
    Returns:
      A tuple (weighted_candidates, soft_weights):
        - weighted_candidates: Array of shape (n_candidates, 3) 
          where each candidate is multiplied by its soft acceptance weight.
        - soft_weights: Array of shape (n_candidates,) containing the acceptance weights.
    """
    key, subkey = random.split(key)
    Jr_candidates = random.uniform(subkey, shape=(n_candidates,), minval=0.0, maxval=200.0)
    key, subkey = random.split(key)
    Jz_candidates = random.uniform(subkey, shape=(n_candidates,), minval=0.0, maxval=50.0)
    key, subkey = random.split(key)
    Lz_candidates = random.uniform(subkey, shape=(n_candidates,), minval=0.0, maxval=4000.0)
    
    candidates = jnp.stack([Jr_candidates, Jz_candidates, Lz_candidates], axis=1)
    
    #Evaluate the total DF for each candidate
    df_total_vec = jit(vmap(lambda cand: df_total_potential(cand[0], cand[1], cand[2], params)))
    df_vals = df_total_vec(candidates)
    
    #Generate uniform random numbers
    key, subkey = random.split(key)
    rand_vals = random.uniform(subkey, shape=(n_candidates,))
    
    #Compute the soft acceptance weights
    soft_weights = soft_acceptance(df_vals, rand_vals, envelope_max, tau)
    
    #here we multiply each candidate by its weight
    weighted_candidates = candidates * soft_weights[:, None]
    
    return weighted_candidates, soft_weights
