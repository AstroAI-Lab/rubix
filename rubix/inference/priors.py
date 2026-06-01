"""Prior builders for inference-time regularization."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp


def build_sfh_ceh_prior_penalty(
    *,
    age_min_gyr: float = 0.5,
    age_max_gyr: float = 12.0,
    sfh_tau_gyr: float = 4.0,
    ceh_z_old: float = 8e-4,
    ceh_z_young: float = 8e-3,
    ceh_gamma: float = 1.2,
    ceh_sigma: float = 1.2e-3,
) -> Callable[[dict[str, Any]], jnp.ndarray]:
    """Build SFH+CEH prior penalty over stellar ages and metallicities.

    The returned function is differentiable and designed for use as an additive
    objective term in VI/optimization.
    """

    age_min = float(age_min_gyr)
    age_max = float(age_max_gyr)
    age_span = max(age_max - age_min, 1e-3)
    tau = max(float(sfh_tau_gyr), 1e-3)
    z_old = float(ceh_z_old)
    z_young = max(float(ceh_z_young), z_old + 1e-6)
    gamma = max(float(ceh_gamma), 0.2)
    sigma_z = max(float(ceh_sigma), 1e-6)

    # Approximate normalization for truncated exponential p(age).
    norm = tau * (jnp.exp(-age_min / tau) - jnp.exp(-age_max / tau))
    norm = jnp.maximum(norm, 1e-10)

    def penalty_fn(params: dict[str, Any]) -> jnp.ndarray:
        age = params["stars"]["age"]
        met = params["stars"]["metallicity"]

        age_clip = jnp.clip(age, age_min, age_max)
        age_nll = age_clip / tau + jnp.log(norm)
        sfh_penalty = jnp.mean(age_nll)

        age_frac = jnp.clip((age_clip - age_min) / age_span, 0.0, 1.0)
        z_pred = z_young - (z_young - z_old) * (age_frac**gamma)
        ceh_penalty = jnp.mean(((met - z_pred) / sigma_z) ** 2)

        value = sfh_penalty + ceh_penalty
        return jnp.nan_to_num(value, nan=0.0, posinf=1e6, neginf=0.0)

    return penalty_fn
