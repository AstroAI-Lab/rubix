#!/usr/bin/env python
"""Run a realistic synthetic VI cycle for Rubix science validation.

This script creates a synthetic galaxy from particle ICs (optionally AGAMA-backed),
constructs a mock IFU cube with simple physically-motivated assumptions, runs VI,
and writes diagnostics useful for science verification and papers.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.image
import jax.numpy as jnp
import numpy as np


@dataclass
class ParticleICs:
    coords_xy: jnp.ndarray
    velocity_xyz: jnp.ndarray
    mass: jnp.ndarray
    age: jnp.ndarray
    metallicity: jnp.ndarray


class RubixNativeForwardPipeline:
    """RUBIX-native forward adapter with explicit output shape handling."""

    def __init__(
        self,
        user_config: dict,
        out_shape: tuple[int, int, int] | None = None,
        resize_mode: str = "linear",
    ):
        from rubix.core.ifu import get_calculate_datacube_particlewise
        from rubix.core.lsf import get_convolve_lsf
        from rubix.core.psf import get_convolve_psf
        from rubix.core.rotation import get_galaxy_rotation
        from rubix.core.telescope import get_spaxel_assignment, get_telescope
        from rubix.pipeline.linear_pipeline import LinearTransformerPipeline
        from rubix.utils import get_pipeline_config

        pipeline_cfg = get_pipeline_config(user_config["pipeline"]["name"])
        funcs = [
            get_galaxy_rotation(user_config),
            get_spaxel_assignment(user_config),
            get_calculate_datacube_particlewise(user_config),
            get_convolve_psf(user_config),
            get_convolve_lsf(user_config),
        ]
        pipe = LinearTransformerPipeline(pipeline_cfg, funcs)
        pipe.assemble()
        self._compiled = pipe.compile_expression()
        self._spaxel_assignment = get_spaxel_assignment(user_config)
        self.native_n_spaxels = int(get_telescope(user_config).sbin)
        self.out_shape = out_shape
        if resize_mode not in {"linear", "none"}:
            raise ValueError("resize_mode must be one of {'linear', 'none'}")
        self.resize_mode = resize_mode

    def run_sharded(self, rubixdata: Any) -> jnp.ndarray:
        out = self._compiled(rubixdata)
        cube = out.stars.datacube
        if self.out_shape is None:
            return cube
        if tuple(cube.shape) == tuple(self.out_shape):
            return cube
        if self.resize_mode == "none":
            raise ValueError(
                f"native cube shape {tuple(cube.shape)} does not match requested "
                f"out_shape {tuple(self.out_shape)}"
            )
        return jax.image.resize(cube, self.out_shape, method="linear")

    def particle_spaxels(self, rubixdata: Any) -> jnp.ndarray:
        assigned = self._spaxel_assignment(copy.deepcopy(rubixdata))
        return _flat_spaxel_indices_to_spaxels(
            assigned.stars.pixel_assignment, self.native_n_spaxels
        )


class LocalSpaxelSpectralPipeline:
    """Particle contributions with local-spaxel deposit and shared spectral scale.

    Each particle contributes only to one spaxel (no spatial smoothing). Spectrally,
    all particles share the same base wavelength scale and only age/metallicity modulate
    amplitudes through smooth differentiable factors.
    """

    def __init__(
        self,
        particle_spaxels: jnp.ndarray,
        particle_mass: jnp.ndarray,
        spectral_base: jnp.ndarray,
        spectral_age_basis: jnp.ndarray,
        spectral_met_basis: jnp.ndarray,
        nx: int,
        ny: int,
        age_scale: float = 3.0,
        metallicity_pivot: float = 0.0025,
        metallicity_scale: float = 0.0015,
        doppler_c_kms: float = 299792.458,
        noise_level: float = 0.02,
    ):
        self.particle_spaxels = particle_spaxels  # (P, 2) integer ix, iy
        self.particle_mass = particle_mass  # (P,)
        self.spectral_base = spectral_base  # (W,)
        self.spectral_age_basis = spectral_age_basis  # (W,)
        self.spectral_met_basis = spectral_met_basis  # (W,)
        self.nx = nx
        self.ny = ny
        self.nw = int(spectral_base.shape[0])
        self.age_scale = float(age_scale)
        self.metallicity_pivot = float(metallicity_pivot)
        self.metallicity_scale = float(metallicity_scale)
        self.doppler_c_kms = float(doppler_c_kms)
        self.noise_level = float(noise_level)

    def _shift_spectrum_linear(
        self, spectrum: jnp.ndarray, delta_pix: jnp.ndarray
    ) -> jnp.ndarray:
        """Shift spectrum by fractional pixels using linear interpolation."""
        idx = jnp.arange(self.nw, dtype=spectrum.dtype)
        src = idx - delta_pix
        src = jnp.clip(src, 0.0, self.nw - 1.0001)
        i0 = jnp.floor(src).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, self.nw - 1)
        frac = src - i0.astype(spectrum.dtype)
        return (1.0 - frac) * spectrum[i0] + frac * spectrum[i1]

    def run_sharded(self, rubixdata: Any) -> jnp.ndarray:
        age = rubixdata.stars.age
        metallicity = rubixdata.stars.metallicity
        vz = rubixdata.stars.velocity[:, 2]

        # Distinct spectral directions for age and metallicity to improve
        # identifiability in inverse recovery.
        age_coef = jnp.clip((age - 5.0) / self.age_scale, -2.0, 2.0)
        met_coef = jnp.clip(
            (metallicity - self.metallicity_pivot) / self.metallicity_scale, -2.0, 2.0
        )

        cube = jnp.zeros((self.nx, self.ny, self.nw), dtype=self.spectral_base.dtype)

        def add_particle(carry, p):
            ix = self.particle_spaxels[p, 0]
            iy = self.particle_spaxels[p, 1]
            spectrum_shape = (
                self.spectral_base
                + age_coef[p] * self.spectral_age_basis
                + met_coef[p] * self.spectral_met_basis
            )
            delta_pix = (vz[p] / self.doppler_c_kms) * (self.nw - 1)
            shifted = self._shift_spectrum_linear(spectrum_shape, delta_pix)
            spectrum = self.particle_mass[p] * shifted
            return carry.at[ix, iy, :].add(spectrum), None

        cube, _ = jax.lax.scan(add_particle, cube, jnp.arange(age.shape[0]))

        if rubixdata.noise_key is not None:
            noise = jax.random.normal(
                rubixdata.noise_key, shape=cube.shape, dtype=cube.dtype
            )
            cube = cube + self.noise_level * noise

        return cube


def _try_sample_with_agama(
    n_particles: int,
    seed: int,
    qjr: float = 0.0,
    qjphi: float = 0.0,
) -> ParticleICs | None:
    """Try AGAMA-backed IC sampling; return None if unavailable/fails.

    This keeps the workflow robust in environments where AGAMA is not installed.
    """

    try:
        import agama  # type: ignore
    except Exception:
        return None

    try:
        rng = np.random.default_rng(seed)
        qjr = float(qjr)
        qjphi = float(qjphi)
        if not (0.0 <= qjr < 1.0):
            return None
        if not (0.0 <= qjphi < 1.0):
            return None

        # Build a simple axisymmetric MW-like potential and a rotating disk DF.
        # This yields phase-space samples with realistic net rotation.
        pot_disk = agama.Potential(
            type="MiyamotoNagai",
            mass=1.0,
            scaleRadius=2.6,
            scaleHeight=0.30,
        )
        pot_halo = agama.Potential(type="NFW", mass=8.0, scaleRadius=12.0)
        pot = agama.Potential(pot_disk, pot_halo)

        df_thin = agama.DistributionFunction(
            type="QuasiIsothermal",
            Sigma0=1.00,
            Rdisk=2.6,
            sigmaz0=20.0,
            Rsigmaz=9.0,
            sigmar0=32.0,
            Rsigmar=9.0,
            qJr=qjr,
            qJphi=qjphi,
            potential=pot,
        )
        df_thick = agama.DistributionFunction(
            type="QuasiIsothermal",
            Sigma0=0.08,
            Rdisk=2.6,
            sigmaz0=42.0,
            Rsigmaz=9.0,
            sigmar0=58.0,
            Rsigmar=9.0,
            qJr=qjr,
            qJphi=qjphi,
            potential=pot,
        )
        df = agama.DistributionFunction(df_thin, df_thick)
        gm = agama.GalaxyModel(potential=pot, df=df)

        xv, sampled_mass = gm.sample(n_particles)
        xv = np.asarray(xv, dtype=np.float32)
        pts = xv[:, :3]
        vel = xv[:, 3:6]
        sampled_mass = np.asarray(sampled_mass, dtype=np.float32)
        sampled_mass = sampled_mass / np.sum(sampled_mass)

        xy = pts[:, :2]
        r = np.linalg.norm(xy, axis=1)

        mass = sampled_mass
        age = np.clip(
            2.0 + 8.0 * (r / (r.max() + 1e-6)) + 0.3 * rng.normal(size=n_particles),
            0.5,
            12.0,
        )
        metallicity = np.clip(
            0.03 * np.exp(-r / 3.0) + 0.002 * rng.normal(size=n_particles), 0.001, 0.04
        )
        return ParticleICs(
            coords_xy=jnp.asarray(xy, dtype=jnp.float32),
            velocity_xyz=jnp.asarray(vel, dtype=jnp.float32),
            mass=jnp.asarray(mass, dtype=jnp.float32),
            age=jnp.asarray(age, dtype=jnp.float32),
            metallicity=jnp.asarray(metallicity, dtype=jnp.float32),
        )
    except Exception:
        return None


def _sample_fallback_disk(n_particles: int, seed: int) -> ParticleICs:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_particles)
    radius = rng.exponential(scale=2.0, size=n_particles)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    xy = np.stack([x, y], axis=1)
    z = rng.normal(0.0, 0.2, size=n_particles)
    vz = rng.normal(0.0, 12.0, size=n_particles)
    vcirc = 160.0 * (1.0 - np.exp(-radius / 2.5))
    vx = -vcirc * np.sin(theta) + rng.normal(0.0, 20.0, size=n_particles)
    vy = vcirc * np.cos(theta) + rng.normal(0.0, 20.0, size=n_particles)
    vel = np.stack([vx, vy, vz], axis=1).astype(np.float32)

    mass = np.ones(n_particles, dtype=np.float32) / float(n_particles)
    age = np.clip(
        2.0
        + 8.0 * (radius / (radius.max() + 1e-6))
        + 0.25 * rng.normal(size=n_particles),
        0.5,
        12.0,
    )
    metallicity = np.clip(
        0.03 * np.exp(-radius / 3.0) + 0.0015 * rng.normal(size=n_particles),
        0.001,
        0.04,
    )

    return ParticleICs(
        coords_xy=jnp.asarray(xy, dtype=jnp.float32),
        velocity_xyz=jnp.asarray(vel, dtype=jnp.float32),
        mass=jnp.asarray(mass, dtype=jnp.float32),
        age=jnp.asarray(age, dtype=jnp.float32),
        metallicity=jnp.asarray(metallicity, dtype=jnp.float32),
    )


def sample_particle_ics(
    n_particles: int,
    seed: int,
    prefer_agama: bool = True,
    agama_qjr: float = 0.0,
    agama_qjphi: float = 0.0,
) -> tuple[ParticleICs, str]:
    if prefer_agama:
        sampled = _try_sample_with_agama(
            n_particles=n_particles,
            seed=seed,
            qjr=agama_qjr,
            qjphi=agama_qjphi,
        )
        if sampled is not None:
            return sampled, "agama"
    return _sample_fallback_disk(n_particles=n_particles, seed=seed), "fallback_disk"


def _apply_sfh_ceh_population_model(
    ics: ParticleICs,
    *,
    seed: int,
    age_min_gyr: float,
    age_max_gyr: float,
    sfh_tau_gyr: float,
    ceh_z_old: float,
    ceh_z_young: float,
    ceh_gamma: float,
    ceh_sigma: float,
    metallicity_min: float,
    metallicity_max: float,
) -> ParticleICs:
    """Resample (age, metallicity) from a fixed SFH+CEH population model."""
    rng = np.random.default_rng(seed)
    n = int(np.asarray(ics.mass).shape[0])

    age_min = float(age_min_gyr)
    age_max = float(age_max_gyr)
    tau = max(float(sfh_tau_gyr), 1e-3)
    z_old = float(ceh_z_old)
    z_young = max(float(ceh_z_young), z_old + 1e-6)
    gamma = max(float(ceh_gamma), 0.2)
    sigma = max(float(ceh_sigma), 1e-8)
    z_lo = float(metallicity_min)
    z_hi = float(metallicity_max)

    # Truncated exponential sampling for p(age) ~ exp(-age/tau) on [age_min, age_max].
    u = rng.uniform(0.0, 1.0, size=n)
    exp_a = np.exp(-age_min / tau)
    exp_b = np.exp(-age_max / tau)
    age = -tau * np.log(exp_a - u * (exp_a - exp_b))
    age = np.clip(age, age_min, age_max)

    age_frac = np.clip((age - age_min) / max(age_max - age_min, 1e-8), 0.0, 1.0)
    z_pred = z_young - (z_young - z_old) * (age_frac**gamma)
    metallicity = z_pred + rng.normal(0.0, sigma, size=n)
    metallicity = np.clip(metallicity, z_lo, z_hi)

    return ParticleICs(
        coords_xy=ics.coords_xy,
        velocity_xyz=ics.velocity_xyz,
        mass=ics.mass,
        age=jnp.asarray(age, dtype=jnp.float32),
        metallicity=jnp.asarray(metallicity, dtype=jnp.float32),
    )


def _make_identifiability_particle_ics(
    *, preset: str, nx: int, ny: int, seed: int
) -> ParticleICs:
    del seed
    if preset == "single_particle":
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
        mass = np.array([1.0], dtype=np.float32)
        age = np.array([7.5], dtype=np.float32)
        metallicity = np.array([0.008], dtype=np.float32)
        velocity = np.array([[0.0, 0.0, 80.0]], dtype=np.float32)
    elif preset in {"one_per_spaxel", "two_per_spaxel"}:
        xs = np.linspace(-1.0, 1.0, int(nx), dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, int(ny), dtype=np.float32)
        coords_list: list[list[float]] = []
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                coords_list.append([float(x), float(y)])
                if preset == "two_per_spaxel":
                    coords_list.append([float(x), float(y)])
        coords = np.asarray(coords_list, dtype=np.float32)
        n = coords.shape[0]
        frac = np.linspace(0.0, 1.0, n, dtype=np.float32)
        mass = np.ones(n, dtype=np.float32) / float(n)
        age = 2.0 + 8.0 * frac
        metallicity = 0.002 + 0.007 * (1.0 - frac)
        velocity = np.zeros((n, 3), dtype=np.float32)
        velocity[:, 2] = -120.0 + 240.0 * frac
    else:
        raise ValueError(f"unknown identifiability preset: {preset}")

    return ParticleICs(
        coords_xy=jnp.asarray(coords, dtype=jnp.float32),
        velocity_xyz=jnp.asarray(velocity, dtype=jnp.float32),
        mass=jnp.asarray(mass, dtype=jnp.float32),
        age=jnp.asarray(age, dtype=jnp.float32),
        metallicity=jnp.asarray(metallicity, dtype=jnp.float32),
    )


def _coords_to_spaxels(coords_xy: jnp.ndarray, nx: int, ny: int) -> jnp.ndarray:
    x = np.asarray(coords_xy[:, 0])
    y = np.asarray(coords_xy[:, 1])

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    ix = np.clip((x_norm * (nx - 1)).astype(np.int32), 0, nx - 1)
    iy = np.clip((y_norm * (ny - 1)).astype(np.int32), 0, ny - 1)
    return jnp.asarray(np.stack([ix, iy], axis=1))


def _flat_spaxel_indices_to_spaxels(
    flat_indices: jnp.ndarray, n_spaxels: int
) -> jnp.ndarray:
    flat = jnp.asarray(flat_indices, dtype=jnp.int32)
    ix = flat % int(n_spaxels)
    iy = flat // int(n_spaxels)
    return jnp.stack([ix, iy], axis=1)


def _masked_mae_rmse(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    diff = np.asarray(prediction)[mask] - np.asarray(target)[mask]
    if diff.size == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def _mass_weighted_diagnostic_spaxel_maps(
    *,
    spaxels: np.ndarray,
    mass: np.ndarray,
    truth_age: np.ndarray,
    fit_age: np.ndarray,
    truth_met: np.ndarray,
    fit_met: np.ndarray,
    truth_vz: np.ndarray,
    fit_vz: np.ndarray,
    nx: int,
    ny: int,
) -> dict[str, Any]:
    sp = np.asarray(spaxels, dtype=np.int32)
    weights = np.asarray(mass, dtype=np.float64)
    quantities = {
        "true_age": np.asarray(truth_age, dtype=np.float64),
        "fit_age": np.asarray(fit_age, dtype=np.float64),
        "true_metallicity": np.asarray(truth_met, dtype=np.float64),
        "fit_metallicity": np.asarray(fit_met, dtype=np.float64),
        "true_vz": np.asarray(truth_vz, dtype=np.float64),
        "fit_vz": np.asarray(fit_vz, dtype=np.float64),
    }
    mass_map = np.zeros((nx, ny), dtype=np.float64)
    weighted_maps = {name: np.zeros((nx, ny), dtype=np.float64) for name in quantities}

    for p, (ix, iy) in enumerate(sp):
        w = float(weights[p])
        mass_map[ix, iy] += w
        for name, values in quantities.items():
            weighted_maps[name][ix, iy] += w * float(values[p])

    occupied = mass_map > 0.0
    maps = {"mass": mass_map}
    for name, weighted in weighted_maps.items():
        out = np.full((nx, ny), np.nan, dtype=np.float64)
        out[occupied] = weighted[occupied] / mass_map[occupied]
        maps[name] = out

    age = _masked_mae_rmse(maps["fit_age"], maps["true_age"], occupied)
    metallicity = _masked_mae_rmse(
        maps["fit_metallicity"], maps["true_metallicity"], occupied
    )
    vz = _masked_mae_rmse(maps["fit_vz"], maps["true_vz"], occupied)
    metrics = {
        "occupied_spaxels": int(np.sum(occupied)),
        "age_mae": age["mae"],
        "age_rmse": age["rmse"],
        "metallicity_mae": metallicity["mae"],
        "metallicity_rmse": metallicity["rmse"],
        "vz_mae_kms": vz["mae"],
        "vz_rmse_kms": vz["rmse"],
    }
    return {"maps": maps, "metrics": metrics}


def _spaxel_occupancy_summary(spaxels: np.ndarray) -> dict[str, Any]:
    sp = np.asarray(spaxels, dtype=np.int32)
    if sp.size == 0:
        return {
            "occupied_spaxels": 0,
            "max_particles_per_spaxel": 0,
            "mean_particles_per_occupied_spaxel": float("nan"),
            "has_aggregated_spaxels": False,
        }

    _, counts = np.unique(sp, axis=0, return_counts=True)
    return {
        "occupied_spaxels": int(counts.size),
        "max_particles_per_spaxel": int(np.max(counts)),
        "mean_particles_per_occupied_spaxel": float(np.mean(counts)),
        "has_aggregated_spaxels": bool(np.max(counts) > 1),
    }


def _recovery_interpretation(
    *, spaxels: np.ndarray, identifiability_summary: dict[str, Any]
) -> dict[str, Any]:
    occupancy = _spaxel_occupancy_summary(spaxels)
    aggregated = bool(occupancy["has_aggregated_spaxels"])
    weak_by_parameter = {
        str(row.get("parameter", "")).split("_", maxsplit=1)[0]: bool(
            row.get("weakly_identified", False)
        )
        for row in identifiability_summary.get("parameters", [])
    }

    rows = []
    for parameter, suffix in (
        ("age", "age_mae"),
        ("metallicity", "metallicity_mae"),
        ("vz", "vz_mae_kms"),
    ):
        primary_metric = (
            f"diagnostic_spaxel_recovery.{suffix}"
            if aggregated
            else f"recovery.{suffix}"
        )
        weak = weak_by_parameter.get(parameter)
        notes = []
        if aggregated:
            notes.append(
                "particle-level recovery is underdetermined within occupied spaxels"
            )
        if weak is True:
            notes.append(
                "finite-difference signal is below the configured weak threshold"
            )
        rows.append(
            {
                "parameter": parameter,
                "primary_metric": primary_metric,
                "particle_recovery_primary": bool(not aggregated),
                "spaxel_recovery_primary": bool(aggregated),
                "weakly_identified": weak,
                "notes": notes,
            }
        )

    return {
        "spaxel_occupancy": occupancy,
        "primary_recovery_level": (
            "spaxel_mass_weighted" if aggregated else "particle"
        ),
        "parameter_recovery": rows,
    }


def _classify_sensitivity_identifiability(
    sensitivity_summary: dict[str, Any],
    *,
    rms_over_sigma_threshold: float = 0.1,
    relative_l2_threshold: float = 1e-3,
) -> dict[str, Any]:
    if not sensitivity_summary.get("enabled", False):
        return {"enabled": False, "parameters": []}

    rows = []
    for row in sensitivity_summary.get("perturbations", []):
        rms_over_sigma = float(row.get("rms_over_sigma", 0.0))
        relative_l2 = float(row.get("relative_l2", 0.0))
        weak = rms_over_sigma < float(rms_over_sigma_threshold)
        rows.append(
            {
                "parameter": row.get("parameter"),
                "weakly_identified": bool(weak),
                "rms_over_sigma": rms_over_sigma,
                "relative_l2": relative_l2,
            }
        )

    return {
        "enabled": True,
        "rms_over_sigma_threshold": float(rms_over_sigma_threshold),
        "relative_l2_threshold": float(relative_l2_threshold),
        "parameters": rows,
    }


def _auto_update_scales_from_sensitivity(
    sensitivity_summary: dict[str, Any], *, max_scale: float
) -> dict[str, float]:
    scales = {"age": 1.0, "metallicity": 1.0, "vz": 1.0}
    if not sensitivity_summary.get("enabled", False):
        return scales

    key_map = {
        "age_all_particles_gyr": "age",
        "metallicity_all_particles": "metallicity",
        "vz_all_particles_kms": "vz",
    }
    values: dict[str, float] = {}
    for row in sensitivity_summary.get("perturbations", []):
        key = key_map.get(row.get("parameter"))
        if key is None:
            continue
        values[key] = max(float(row.get("rms_over_sigma", 0.0)), 1e-12)

    if not values:
        return scales

    reference = max(values.values())
    cap = max(float(max_scale), 1.0)
    for key, value in values.items():
        scales[key] = float(np.clip(reference / value, 1.0, cap))
    return scales


def _constrained_gradient_signal_summary(
    *,
    forward_fn: Callable[[dict[str, Any]], jnp.ndarray],
    target: jnp.ndarray,
    sigma: jnp.ndarray,
    params: dict[str, Any],
    zero_threshold: float = 1e-12,
) -> dict[str, Any]:
    stars = params["stars"]
    age0 = jnp.asarray(stars["age"])
    met0 = jnp.asarray(stars["metallicity"])
    velocity0 = jnp.asarray(stars["velocity"])

    def loss(age: jnp.ndarray, metallicity: jnp.ndarray, vz: jnp.ndarray):
        velocity = velocity0.at[:, 2].set(vz)
        cube = forward_fn(
            {
                "stars": {
                    "age": age,
                    "metallicity": metallicity,
                    "velocity": velocity,
                }
            }
        )
        resid = (cube - target) / sigma
        return 0.5 * jnp.mean(resid**2)

    grads = jax.grad(loss, argnums=(0, 1, 2))(age0, met0, velocity0[:, 2])
    rows = []
    for name, grad in zip(("age", "metallicity", "vz"), grads):
        grad_abs = jnp.abs(grad)
        norm = float(jnp.linalg.norm(grad))
        max_abs = float(jnp.max(grad_abs)) if grad.size else 0.0
        rows.append(
            {
                "parameter": name,
                "l2_norm": norm,
                "mean_abs": float(jnp.mean(grad_abs)) if grad.size else 0.0,
                "max_abs": max_abs,
                "zero_gradient": bool(max_abs <= float(zero_threshold)),
            }
        )

    return {
        "enabled": True,
        "loss": float(loss(age0, met0, velocity0[:, 2])),
        "zero_threshold": float(zero_threshold),
        "parameters": rows,
    }


def _build_custom_wavelength_template(nw: int) -> jnp.ndarray:
    """Fallback synthetic wavelength template used when SSP lookup is unavailable."""
    w = np.linspace(3600.0, 7000.0, nw, dtype=np.float32)
    base = 0.8 + 0.15 * np.sin((w - w.min()) / 500.0)

    def gauss(mu: float, sig: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((w - mu) / sig) ** 2)

    template = base - gauss(4861.0, 4.5, 0.08) - gauss(6563.0, 5.5, 0.06)
    return jnp.asarray(template / np.mean(template), dtype=jnp.float32)


def _build_spectral_components(
    nw: int,
    ssp_template_name: str = "BruzualCharlot2003",
    ssp_age_gyr: float | None = None,
    ssp_metallicity: float | None = None,
    age_basis_scale: float = 0.35,
    met_basis_scale: float = 0.25,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, str]:
    """Build spectral base and age/metallicity basis from RUBIX SSP lookup.

    Falls back to a custom synthetic template if SSP loading/interpolation fails.
    """
    try:
        from rubix.spectra.ssp.factory import get_ssp_template

        ssp = get_ssp_template(ssp_template_name)
        age_grid = np.asarray(ssp.age, dtype=np.float64)
        met_grid = np.asarray(ssp.metallicity, dtype=np.float64)
        wave_grid = np.asarray(ssp.wavelength, dtype=np.float64)

        target_age = float(np.median(age_grid) if ssp_age_gyr is None else ssp_age_gyr)
        target_met = float(
            np.median(met_grid) if ssp_metallicity is None else ssp_metallicity
        )
        target_age = float(np.clip(target_age, age_grid.min(), age_grid.max()))
        target_met = float(np.clip(target_met, met_grid.min(), met_grid.max()))

        lookup = ssp.get_lookup_interpolation(method="linear")

        # Build local spectral basis around fiducial point.
        d_age = max(0.2, 0.1 * (age_grid.max() - age_grid.min()))
        d_met = max(1e-3, 0.1 * (met_grid.max() - met_grid.min()))
        age_lo = float(np.clip(target_age - d_age, age_grid.min(), age_grid.max()))
        age_hi = float(np.clip(target_age + d_age, age_grid.min(), age_grid.max()))
        met_lo = float(np.clip(target_met - d_met, met_grid.min(), met_grid.max()))
        met_hi = float(np.clip(target_met + d_met, met_grid.min(), met_grid.max()))

        spec_native = np.asarray(lookup(target_met, target_age), dtype=np.float64)
        spec_age_lo = np.asarray(lookup(target_met, age_lo), dtype=np.float64)
        spec_age_hi = np.asarray(lookup(target_met, age_hi), dtype=np.float64)
        spec_met_lo = np.asarray(lookup(met_lo, target_age), dtype=np.float64)
        spec_met_hi = np.asarray(lookup(met_hi, target_age), dtype=np.float64)

        wave_out = np.linspace(wave_grid.min(), wave_grid.max(), nw, dtype=np.float64)
        base = np.interp(wave_out, wave_grid, spec_native)
        age_basis = np.interp(wave_out, wave_grid, spec_age_hi - spec_age_lo)
        met_basis = np.interp(wave_out, wave_grid, spec_met_hi - spec_met_lo)

        base = base / (np.mean(base) + 1e-12)
        age_basis = age_basis - np.mean(age_basis)
        met_basis = met_basis - np.mean(met_basis)
        age_norm = np.linalg.norm(age_basis) + 1e-12
        met_norm = np.linalg.norm(met_basis) + 1e-12
        age_basis = float(age_basis_scale) * age_basis / age_norm
        met_basis = float(met_basis_scale) * met_basis / met_norm

        return (
            jnp.asarray(base, dtype=jnp.float32),
            jnp.asarray(age_basis, dtype=jnp.float32),
            jnp.asarray(met_basis, dtype=jnp.float32),
            "ssp_lookup_basis",
        )
    except Exception:
        base = _build_custom_wavelength_template(nw)
        w = np.linspace(-1.0, 1.0, nw, dtype=np.float32)
        age_basis = float(age_basis_scale) * w
        met_basis = float(met_basis_scale) * (w**2 - np.mean(w**2))
        return (
            base,
            jnp.asarray(age_basis, dtype=jnp.float32),
            jnp.asarray(met_basis, dtype=jnp.float32),
            "custom_fallback_basis",
        )


def _make_static_data(ics: ParticleICs) -> Any:
    from rubix.core.data import Galaxy, GasData, RubixData, StarsData

    p = int(ics.mass.shape[0])
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.concatenate(
                [ics.coords_xy, jnp.zeros((p, 1), dtype=ics.coords_xy.dtype)], axis=1
            ),
            velocity=ics.velocity_xyz,
            mass=ics.mass,
            age=jnp.ones(p),
            metallicity=jnp.ones(p) * 0.02,
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)), velocity=jnp.zeros((1, 3)), mass=jnp.ones(1)
        ),
    )


def _build_rubix_native_config(
    *, pipeline_name: str, telescope_name: str, dist_z: float
) -> dict[str, Any]:
    from rubix import config as rubix_base_config

    cfg = copy.deepcopy(rubix_base_config)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["name"] = pipeline_name
    cfg.setdefault("telescope", {})
    cfg["telescope"]["name"] = telescope_name
    # Ensure required convolution settings exist for calc_gradient.
    cfg["telescope"].setdefault("psf", {"name": "gaussian", "size": 5, "sigma": 0.6})
    cfg["telescope"].setdefault("lsf", {"sigma": 0.5})
    cfg.setdefault("simulation", {})
    cfg["simulation"]["name"] = "IllustrisTNG"
    cfg.setdefault("cosmology", {})
    cfg["cosmology"]["name"] = "PLANCK15"
    cfg.setdefault("galaxy", {})
    cfg["galaxy"]["dist_z"] = float(dist_z)
    cfg["galaxy"]["rotation"] = {"type": "face-on"}
    cfg.setdefault("data", {})
    cfg["data"].setdefault("args", {})
    cfg["data"]["args"]["particle_type"] = ["stars"]
    cfg.setdefault("ssp", {})
    cfg["ssp"].setdefault("template", {})
    cfg["ssp"]["template"]["name"] = "BruzualCharlot2003"
    cfg["ssp"]["method"] = "linear"
    cfg["logger"] = None
    return cfg


def _build_parameter_penalty_fn(
    *,
    particle_spaxels: jnp.ndarray,
    age_anchor: float,
    metallicity_anchor: float,
    smoothness_weight: float,
    amr_weight: float,
    mean_age_weight: float,
    mean_metallicity_weight: float,
    sfh_ceh_weight: float,
    ceh_relation_weight: float,
    sfh_ceh_penalty_fn: Any | None = None,
    ceh_relation_penalty_fn: Any | None = None,
) -> Any:
    """Build a lightweight physics-inspired regularization penalty.

    The penalty combines:
    - local smoothness over adjacent-spaxel particles for age/metallicity,
    - weak anti-correlation preference between age and metallicity residuals,
    - weak global anchors on mean age/metallicity.
    """
    sp = np.asarray(particle_spaxels, dtype=np.int32)
    n_particles = int(sp.shape[0])
    edges_i: list[int] = []
    edges_j: list[int] = []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = abs(int(sp[i, 0]) - int(sp[j, 0]))
            dy = abs(int(sp[i, 1]) - int(sp[j, 1]))
            if max(dx, dy) <= 1:
                edges_i.append(i)
                edges_j.append(j)
    if len(edges_i) == 0:
        edges_i = list(range(n_particles))
        edges_j = list(range(n_particles))
    edge_i = jnp.asarray(edges_i, dtype=jnp.int32)
    edge_j = jnp.asarray(edges_j, dtype=jnp.int32)
    age_anchor_jnp = jnp.asarray(age_anchor, dtype=jnp.float32)
    met_anchor_jnp = jnp.asarray(metallicity_anchor, dtype=jnp.float32)

    def penalty_fn(params: dict[str, Any]) -> jnp.ndarray:
        age = params["stars"]["age"]
        met = params["stars"]["metallicity"]
        age_diff = age[edge_i] - age[edge_j]
        met_diff = met[edge_i] - met[edge_j]
        smooth = (
            jnp.mean(age_diff**2) + 10.0 * jnp.mean(met_diff**2)
            if smoothness_weight > 0.0
            else jnp.asarray(0.0, dtype=age.dtype)
        )
        age_centered = age - jnp.mean(age)
        met_centered = met - jnp.mean(met)
        age_scale = jnp.sqrt(jnp.maximum(jnp.mean(age_centered**2), 1e-4))
        met_scale = jnp.sqrt(jnp.maximum(jnp.mean(met_centered**2), 1e-10))
        denom = age_scale * met_scale
        corr = jnp.clip(jnp.mean(age_centered * met_centered) / denom, -1.0, 1.0)
        corr = jnp.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        amr = jnp.maximum(corr + 0.25, 0.0) ** 2 if amr_weight > 0.0 else 0.0
        mean_age_pen = (
            (jnp.mean(age) - age_anchor_jnp) ** 2 if mean_age_weight > 0.0 else 0.0
        )
        mean_met_pen = (
            (jnp.mean(met) - met_anchor_jnp) ** 2
            if mean_metallicity_weight > 0.0
            else 0.0
        )
        value = (
            smoothness_weight * smooth
            + amr_weight * amr
            + mean_age_weight * mean_age_pen
            + mean_metallicity_weight * mean_met_pen
        )
        if sfh_ceh_penalty_fn is not None and sfh_ceh_weight > 0.0:
            value = value + sfh_ceh_weight * sfh_ceh_penalty_fn(params)
        if ceh_relation_penalty_fn is not None and ceh_relation_weight > 0.0:
            value = value + ceh_relation_weight * ceh_relation_penalty_fn(params)
        return jnp.nan_to_num(value, nan=0.0, posinf=1e6, neginf=0.0)

    return penalty_fn


def _build_vi_optimizer(args: argparse.Namespace) -> Any:
    import optax

    transforms = []
    if args.grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(args.grad_clip_norm)))

    if args.optimizer == "adam":
        transforms.append(
            optax.adam(
                learning_rate=args.vi_lr,
                b1=args.adam_b1,
                b2=args.adam_b2,
                eps=args.adam_eps,
            )
        )
    elif args.optimizer == "adamw":
        transforms.append(
            optax.adamw(
                learning_rate=args.vi_lr,
                b1=args.adam_b1,
                b2=args.adam_b2,
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        )
    elif args.optimizer == "lbfgs":
        if args.lbfgs_disable_linesearch:
            transforms.append(
                optax.lbfgs(
                    learning_rate=args.vi_lr,
                    memory_size=args.lbfgs_memory_size,
                    linesearch=None,
                )
            )
        else:
            transforms.append(optax.lbfgs(memory_size=args.lbfgs_memory_size))
    else:
        raise ValueError(f"unsupported optimizer: {args.optimizer}")

    update_scaler = _build_vi_mean_update_scaler(
        age_scale=args.age_update_scale,
        metallicity_scale=args.metallicity_update_scale,
        vz_scale=args.vz_update_scale,
    )
    if update_scaler is not None:
        transforms.append(update_scaler)

    if len(transforms) == 1:
        return transforms[0]
    return optax.chain(*transforms)


def _build_vi_mean_update_scaler(
    *, age_scale: float, metallicity_scale: float, vz_scale: float
) -> Any | None:
    import optax

    scales = {
        "age": float(age_scale),
        "metallicity": float(metallicity_scale),
        "vz": float(vz_scale),
    }
    if all(np.isclose(value, 1.0) for value in scales.values()):
        return None
    if any(value <= 0.0 for value in scales.values()):
        raise ValueError("parameter update scales must be strictly positive")

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        scaled = copy.deepcopy(updates)
        stars = scaled.get("mean", {}).get("stars", {})
        if "age" in stars:
            stars["age"] = stars["age"] * scales["age"]
        if "metallicity" in stars:
            stars["metallicity"] = stars["metallicity"] * scales["metallicity"]
        if "velocity" in stars:
            velocity = stars["velocity"]
            stars["velocity"] = velocity.at[:, 2].multiply(scales["vz"])
        return scaled, state

    return optax.GradientTransformation(init_fn, update_fn)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    arr = np.asarray(value) if isinstance(value, (jnp.ndarray, np.ndarray)) else None
    if arr is not None:
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run realistic synthetic Rubix VI cycle."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/realistic_synthetic_vi_cycle"
    )
    parser.add_argument(
        "--forward-model",
        type=str,
        default="rubix_native",
        choices=["rubix_native", "legacy_local"],
    )
    parser.add_argument("--pipeline-name", type=str, default="calc_gradient")
    parser.add_argument("--telescope-name", type=str, default="MUSE")
    parser.add_argument("--galaxy-dist-z", type=float, default=0.05)
    parser.add_argument(
        "--native-resize-mode",
        type=str,
        default="linear",
        choices=["linear", "none"],
        help=(
            "How the RUBIX-native adapter handles cube shape mismatches. "
            "'linear' preserves the historical resized output; 'none' raises "
            "if the requested shape is not the telescope-native shape."
        ),
    )
    parser.add_argument("--n-particles", type=int, default=64)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nw", type=int, default=64)
    parser.add_argument(
        "--ic-preset",
        type=str,
        default="sampled",
        choices=["sampled", "single_particle", "one_per_spaxel", "two_per_spaxel"],
        help=(
            "Particle initial condition preset. Use non-sampled presets for "
            "deterministic identifiability ladder checks."
        ),
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="mean",
        choices=["mean", "truth", "truth_zero_vz"],
        help=(
            "Initialization for VI posterior means in constrained space. "
            "'mean' keeps the historical population mean start; 'truth' is a "
            "sanity check; 'truth_zero_vz' isolates velocity recovery."
        ),
    )
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=1e-6,
        help="Numerical floor for the assumed per-voxel Gaussian sigma.",
    )
    parser.add_argument(
        "--noise-relative",
        type=float,
        default=0.0,
        help=(
            "Relative (flux-scaled) noise for the assumed sigma cube; its "
            "inverse is the bright-end S/N. When >0 (or --noise-poisson-scale "
            ">0) the per-voxel sigma is tied to flux instead of a constant, so "
            "it need not be retuned per cube brightness."
        ),
    )
    parser.add_argument(
        "--noise-poisson-scale",
        type=float,
        default=0.0,
        help="Shot-noise variance per unit flux for the assumed sigma cube.",
    )
    parser.add_argument(
        "--add-observational-noise",
        action="store_true",
        help=(
            "Add Gaussian noise to the target at the assumed per-voxel sigma "
            "(seeded by --seed). Required for well-posed calibration/coverage: "
            "without it the target is noise-free, so the assumed sigma is "
            "arbitrary and coverage is ill-defined."
        ),
    )
    parser.add_argument("--vi-steps", type=int, default=300)
    parser.add_argument("--vi-lr", type=float, default=8e-3)
    parser.add_argument("--num-vi-samples", type=int, default=4)
    parser.add_argument("--init-log-std", type=float, default=-2.0)
    parser.add_argument(
        "--posterior-rank",
        type=int,
        default=0,
        help=(
            "Factor rank for a low-rank-plus-diagonal Gaussian posterior "
            "(0 = diagonal mean-field). Rank >= 1 captures correlated "
            "age-metallicity geometry."
        ),
    )
    parser.add_argument(
        "--posterior-block",
        action="store_true",
        help=(
            "Use a per-particle block-diagonal Gaussian posterior coupling "
            "age, metallicity, and v_z into a dense 3x3 block each. Addresses "
            "the mean-field variance underestimation on calibrated (--beta-kl "
            ">0) runs. Mutually exclusive with --posterior-rank."
        ),
    )
    parser.add_argument("--num-posterior-samples", type=int, default=16)
    parser.add_argument(
        "--beta-kl",
        type=float,
        default=0.0,
        help=(
            "KL weight. Default 0.0 (MAP) is best for parameter *recovery*: with "
            "the summed likelihood, beta_kl=1.0 gives a calibrated ELBO but the "
            "standard-normal unconstrained prior is informative for sigmoid- "
            "bounded parameters and biases the mean (measured age recovery on "
            "the native 2x2 rung degrades ~8x). Use beta_kl=1.0 only for "
            "calibrated-posterior studies, and check coverage with "
            "run_vi_calibration.py."
        ),
    )
    parser.add_argument(
        "--prior-std",
        type=float,
        default=1.814,
        help=(
            "Std of the zero-mean Gaussian prior on unconstrained latents in the "
            "KL term (only active when --beta-kl > 0). Default 1.814 (pi/sqrt(3)) "
            "variance-matches the logistic prior that induces a uniform physical "
            "prior for sigmoid-bounded parameters, avoiding the midpoint bias of "
            "a standard-normal (std=1.0) prior."
        ),
    )
    parser.add_argument(
        "--normalize-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the per-voxel-mean Gaussian loss. Default is the summed "
            "(calibrated, velocity-capable) likelihood; pass --normalize-loss "
            "only for MAP-style point fits with --beta-kl 0."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "lbfgs"],
    )
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--age-update-scale", type=float, default=1.0)
    parser.add_argument("--metallicity-update-scale", type=float, default=1.0)
    parser.add_argument("--vz-update-scale", type=float, default=1.0)
    parser.add_argument(
        "--auto-update-scales-from-sensitivity",
        action="store_true",
        help=(
            "When --sensitivity-check is enabled, multiply parameter update "
            "scales to equalize perturbation RMS-over-sigma signals."
        ),
    )
    parser.add_argument("--max-auto-update-scale", type=float, default=300.0)
    parser.add_argument("--lbfgs-memory-size", type=int, default=10)
    parser.add_argument(
        "--lbfgs-disable-linesearch",
        action="store_true",
        help="Use --vi-lr as a fixed L-BFGS step instead of Optax line search.",
    )
    parser.add_argument(
        "--map-warmup-steps",
        type=int,
        default=5,
        help=(
            "Run a MAP-style optimizer warm start before the main VI pass. "
            "Defaults to a 5-step L-BFGS warmup, which with the summed "
            "likelihood is the current best recipe for line-of-sight velocity "
            "recovery. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--map-warmup-optimizer",
        type=str,
        default="lbfgs",
        choices=["adam", "adamw", "lbfgs"],
    )
    parser.add_argument("--map-warmup-lr", type=float, default=1e-3)
    parser.add_argument("--map-warmup-init-log-std", type=float, default=-10.0)
    parser.add_argument("--map-warmup-num-samples", type=int, default=1)
    parser.add_argument("--map-warmup-beta-kl", type=float, default=0.0)
    parser.add_argument(
        "--map-warmup-use-priors",
        action="store_true",
        help="Apply configured parameter priors during MAP warmup.",
    )
    parser.add_argument(
        "--ssp-template-name",
        type=str,
        default="BruzualCharlot2003",
        help="RUBIX SSP template key used to build the wavelength template.",
    )
    parser.add_argument(
        "--ssp-age-gyr",
        type=float,
        default=None,
        help="Optional SSP age (Gyr) used for spectral lookup.",
    )
    parser.add_argument(
        "--ssp-metallicity",
        type=float,
        default=None,
        help="Optional SSP metallicity used for spectral lookup.",
    )
    parser.add_argument(
        "--age-basis-scale",
        type=float,
        default=0.35,
        help="Scale factor for age spectral basis amplitude.",
    )
    parser.add_argument(
        "--met-basis-scale",
        type=float,
        default=0.25,
        help="Scale factor for metallicity spectral basis amplitude.",
    )
    parser.add_argument(
        "--age-response-scale",
        type=float,
        default=3.0,
        help="Denominator controlling age response sensitivity.",
    )
    parser.add_argument(
        "--met-response-pivot",
        type=float,
        default=0.0025,
        help="Pivot metallicity for metallicity response.",
    )
    parser.add_argument(
        "--met-response-scale",
        type=float,
        default=0.0015,
        help="Denominator controlling metallicity response sensitivity.",
    )
    parser.add_argument("--vz-lower-kms", type=float, default=-300.0)
    parser.add_argument("--vz-upper-kms", type=float, default=300.0)
    parser.add_argument(
        "--prior-smoothness-weight",
        type=float,
        default=0.0,
        help="Weight for local spatial smoothness regularization.",
    )
    parser.add_argument(
        "--prior-amr-weight",
        type=float,
        default=0.0,
        help="Weight for weak age-metallicity anti-correlation prior.",
    )
    parser.add_argument(
        "--prior-mean-age-weight",
        type=float,
        default=0.0,
        help="Weight for weak prior on global mean stellar age.",
    )
    parser.add_argument(
        "--prior-mean-met-weight",
        type=float,
        default=0.0,
        help="Weight for weak prior on global mean stellar metallicity.",
    )
    parser.add_argument(
        "--prior-ramp-steps",
        type=int,
        default=80,
        help="Linear ramp-in steps for prior terms.",
    )
    parser.add_argument(
        "--prior-sfh-ceh-weight",
        type=float,
        default=0.0,
        help="Weight for SFH+CEH joint prior penalty.",
    )
    parser.add_argument(
        "--prior-ceh-relation-weight",
        type=float,
        default=0.0,
        help="Weight for CEH age-metallicity relation penalty without SFH age prior.",
    )
    parser.add_argument(
        "--prior-sfh-tau-gyr",
        type=float,
        default=4.0,
        help="Exponential SFH timescale (Gyr).",
    )
    parser.add_argument(
        "--prior-ceh-z-old",
        type=float,
        default=8e-4,
        help="Old-population metallicity anchor for CEH prior.",
    )
    parser.add_argument(
        "--prior-ceh-z-young",
        type=float,
        default=8e-3,
        help="Young-population metallicity anchor for CEH prior.",
    )
    parser.add_argument(
        "--prior-ceh-gamma",
        type=float,
        default=1.2,
        help="Shape exponent for monotonic age-metallicity trend.",
    )
    parser.add_argument(
        "--prior-ceh-sigma",
        type=float,
        default=1.2e-3,
        help="Scatter scale for CEH prior residuals.",
    )
    parser.add_argument(
        "--synthetic-population-model",
        type=str,
        default="legacy",
        choices=["legacy", "sfh_ceh"],
        help="Population generator for true age-metallicity labels.",
    )
    parser.add_argument("--synthetic-age-min-gyr", type=float, default=0.5)
    parser.add_argument("--synthetic-age-max-gyr", type=float, default=12.0)
    parser.add_argument("--synthetic-sfh-tau-gyr", type=float, default=4.0)
    parser.add_argument("--synthetic-ceh-z-old", type=float, default=8e-4)
    parser.add_argument("--synthetic-ceh-z-young", type=float, default=8e-3)
    parser.add_argument("--synthetic-ceh-gamma", type=float, default=1.2)
    parser.add_argument("--synthetic-ceh-sigma", type=float, default=1.2e-3)
    parser.add_argument("--synthetic-metallicity-min", type=float, default=5e-4)
    parser.add_argument("--synthetic-metallicity-max", type=float, default=0.01)
    parser.add_argument(
        "--agama-qjr",
        type=float,
        default=0.0,
        help="AGAMA QuasiIsothermal qJr parameter in [0,1).",
    )
    parser.add_argument(
        "--agama-qjphi",
        type=float,
        default=0.0,
        help="AGAMA QuasiIsothermal qJphi parameter in [0,1).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-agama",
        action="store_true",
        help="Disable AGAMA IC sampling even if installed.",
    )
    parser.add_argument(
        "--skip-gradient-check",
        action="store_true",
        help="Skip finite-difference gradient check (recommended for routine runs).",
    )
    parser.add_argument(
        "--sensitivity-check",
        action="store_true",
        help="Compute finite perturbation cube sensitivities at the truth point.",
    )
    parser.add_argument(
        "--gradient-signal-check",
        action="store_true",
        help=(
            "Compute local autodiff gradient norms at the VI initialization; "
            "useful for detecting non-differentiable or flat native model directions."
        ),
    )
    parser.add_argument("--gradient-zero-threshold", type=float, default=1e-12)
    parser.add_argument("--sensitivity-age-delta", type=float, default=1.0)
    parser.add_argument("--sensitivity-metallicity-delta", type=float, default=1e-3)
    parser.add_argument("--sensitivity-vz-delta-kms", type=float, default=100.0)
    parser.add_argument("--weak-rms-over-sigma-threshold", type=float, default=0.1)
    parser.add_argument("--weak-relative-l2-threshold", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    from rubix.inference import (
        apply_params,
        build_age_metallicity_velocity_transforms,
        build_ceh_relation_prior_penalty,
        build_sfh_ceh_prior_penalty,
        compare_gradients,
        compute_residual_products,
        finite_difference_grad,
        flux_scaled_sigma,
        optimize_variational_ifu_cube,
        sample_posterior_parameters,
        sample_posterior_predictive_cubes,
        summarize_masked_metrics,
        summarize_predictive_cube_samples,
    )

    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}
    t0_total = time.perf_counter()

    t_stage = time.perf_counter()
    if args.ic_preset == "sampled":
        ics, sampler = sample_particle_ics(
            n_particles=args.n_particles,
            seed=args.seed,
            prefer_agama=not args.no_agama,
            agama_qjr=args.agama_qjr,
            agama_qjphi=args.agama_qjphi,
        )
    else:
        ics = _make_identifiability_particle_ics(
            preset=args.ic_preset,
            nx=args.nx,
            ny=args.ny,
            seed=args.seed,
        )
        sampler = f"identifiability_{args.ic_preset}"
        args.n_particles = int(ics.mass.shape[0])

    if args.synthetic_population_model == "sfh_ceh":
        ics = _apply_sfh_ceh_population_model(
            ics,
            seed=args.seed + 17,
            age_min_gyr=args.synthetic_age_min_gyr,
            age_max_gyr=args.synthetic_age_max_gyr,
            sfh_tau_gyr=args.synthetic_sfh_tau_gyr,
            ceh_z_old=args.synthetic_ceh_z_old,
            ceh_z_young=args.synthetic_ceh_z_young,
            ceh_gamma=args.synthetic_ceh_gamma,
            ceh_sigma=args.synthetic_ceh_sigma,
            metallicity_min=args.synthetic_metallicity_min,
            metallicity_max=args.synthetic_metallicity_max,
        )
    stage_times["sample_particle_ics_s"] = time.perf_counter() - t_stage
    print(
        (
            f"[stage] sampled ICs via {sampler} (n_particles={args.n_particles}, "
            f"population_model={args.synthetic_population_model})"
        ),
        flush=True,
    )
    t_stage = time.perf_counter()
    template_source = "legacy_local"
    diagnostic_spaxels_source = "coordinate_rescaled_to_output_grid"
    if args.forward_model == "rubix_native":
        user_cfg = _build_rubix_native_config(
            pipeline_name=args.pipeline_name,
            telescope_name=args.telescope_name,
            dist_z=args.galaxy_dist_z,
        )
        pipe = RubixNativeForwardPipeline(
            user_config=user_cfg,
            out_shape=(args.nx, args.ny, args.nw),
            resize_mode=args.native_resize_mode,
        )
        static_data = _make_static_data(ics)
        static_data.galaxy.redshift = jnp.asarray(args.galaxy_dist_z, dtype=jnp.float32)
        static_data.galaxy.center = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)
        static_data.galaxy.halfmassrad_stars = jnp.asarray(5.0, dtype=jnp.float32)
        spaxels = _coords_to_spaxels(ics.coords_xy, nx=args.nx, ny=args.ny)
        if args.nx == pipe.native_n_spaxels and args.ny == pipe.native_n_spaxels:
            spaxels = pipe.particle_spaxels(static_data)
            diagnostic_spaxels_source = "rubix_native_pixel_assignment"
        template_source = "rubix_ssp_lookup_native"
    else:
        spaxels = _coords_to_spaxels(ics.coords_xy, nx=args.nx, ny=args.ny)
        diagnostic_spaxels_source = "local_forward_model_coordinate_rescaled_to_grid"
        (
            spectral_base,
            spectral_age_basis,
            spectral_met_basis,
            template_source,
        ) = _build_spectral_components(
            args.nw,
            ssp_template_name=args.ssp_template_name,
            ssp_age_gyr=args.ssp_age_gyr,
            ssp_metallicity=args.ssp_metallicity,
            age_basis_scale=args.age_basis_scale,
            met_basis_scale=args.met_basis_scale,
        )
        pipe = LocalSpaxelSpectralPipeline(
            particle_spaxels=spaxels,
            particle_mass=ics.mass,
            spectral_base=spectral_base,
            spectral_age_basis=spectral_age_basis,
            spectral_met_basis=spectral_met_basis,
            nx=args.nx,
            ny=args.ny,
            age_scale=args.age_response_scale,
            metallicity_pivot=args.met_response_pivot,
            metallicity_scale=args.met_response_scale,
            noise_level=args.noise_level,
        )
        static_data = _make_static_data(ics)
    stage_times["build_spaxels_template_s"] = time.perf_counter() - t_stage
    print(f"[stage] forward model: {args.forward_model}", flush=True)
    print(f"[stage] wavelength template source: {template_source}", flush=True)

    true_params = {
        "stars": {
            "age": ics.age,
            "metallicity": ics.metallicity,
            "velocity": ics.velocity_xyz,
        }
    }
    if args.init_mode == "truth":
        init_params = true_params
    elif args.init_mode == "truth_zero_vz":
        velocity_init = np.asarray(ics.velocity_xyz).copy()
        velocity_init[:, 2] = 0.0
        init_params = {
            "stars": {
                "age": ics.age,
                "metallicity": ics.metallicity,
                "velocity": jnp.asarray(velocity_init, dtype=jnp.float32),
            }
        }
    else:
        velocity_init = np.asarray(ics.velocity_xyz).copy()
        velocity_init[:, 2] = 0.0
        init_params = {
            "stars": {
                "age": jnp.full_like(ics.age, jnp.mean(ics.age)),
                "metallicity": jnp.full_like(
                    ics.metallicity, jnp.mean(ics.metallicity)
                ),
                "velocity": jnp.asarray(velocity_init, dtype=jnp.float32),
            }
        }

    initial_params = init_params

    t_stage = time.perf_counter()
    target = pipe.run_sharded(
        apply_params(
            static_data,
            {
                "stars": {
                    "age": true_params["stars"]["age"],
                    "metallicity": true_params["stars"]["metallicity"],
                    "velocity": true_params["stars"]["velocity"],
                }
            },
        )
    )
    stage_times["build_target_cube_s"] = time.perf_counter() - t_stage
    print("[stage] built synthetic target cube", flush=True)

    if args.noise_relative > 0.0 or args.noise_poisson_scale > 0.0:
        # Flux-scaled per-voxel sigma: floor set by max(noise_level, sigma_floor)
        # so the constant term is preserved and the flux terms add on top.
        sigma = flux_scaled_sigma(
            target,
            relative_noise=args.noise_relative,
            floor=max(float(args.noise_level), float(args.sigma_floor)),
            poisson_scale=args.noise_poisson_scale,
        )
        sigma_level = float(jnp.median(sigma))
    else:
        sigma_level = max(float(args.noise_level), float(args.sigma_floor))
        sigma = jnp.ones_like(target) * sigma_level

    if args.add_observational_noise:
        # Inject a noise realization at the assumed per-voxel sigma so the
        # likelihood is correctly scaled and coverage/SBC over seeds is
        # well-posed. Keyed off the seed but distinct from the VI sampling key.
        noise_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 8191)
        target = target + sigma * jax.random.normal(
            noise_key, shape=target.shape, dtype=target.dtype
        )
    target_norm = float(jnp.linalg.norm(target))

    def _cube_sensitivity(
        label: str, params: dict[str, Any], delta: float
    ) -> dict[str, float | str]:
        pred = pipe.run_sharded(apply_params(static_data, params))
        diff = pred - target
        l2 = float(jnp.linalg.norm(diff))
        rms = float(jnp.sqrt(jnp.mean(diff**2)))
        max_abs = float(jnp.max(jnp.abs(diff)))
        sigma_safe = float(sigma_level)
        return {
            "parameter": label,
            "delta": float(delta),
            "l2": l2,
            "rms": rms,
            "max_abs": max_abs,
            "relative_l2": l2 / max(target_norm, 1e-12),
            "rms_over_sigma": rms / max(sigma_safe, 1e-12),
            "max_abs_over_sigma": max_abs / max(sigma_safe, 1e-12),
            "l2_per_unit": l2 / max(abs(float(delta)), 1e-12),
            "rms_per_unit": rms / max(abs(float(delta)), 1e-12),
            "max_abs_per_unit": max_abs / max(abs(float(delta)), 1e-12),
        }

    if args.sensitivity_check:
        t_sens = time.perf_counter()
        age_delta = float(args.sensitivity_age_delta)
        met_delta = float(args.sensitivity_metallicity_delta)
        vz_delta = float(args.sensitivity_vz_delta_kms)
        sensitivity_summary = {
            "enabled": True,
            "target_l2": target_norm,
            "perturbations": [
                _cube_sensitivity(
                    "age_all_particles_gyr",
                    {
                        "stars": {
                            "age": true_params["stars"]["age"] + age_delta,
                            "metallicity": true_params["stars"]["metallicity"],
                            "velocity": true_params["stars"]["velocity"],
                        }
                    },
                    age_delta,
                ),
                _cube_sensitivity(
                    "metallicity_all_particles",
                    {
                        "stars": {
                            "age": true_params["stars"]["age"],
                            "metallicity": true_params["stars"]["metallicity"]
                            + met_delta,
                            "velocity": true_params["stars"]["velocity"],
                        }
                    },
                    met_delta,
                ),
                _cube_sensitivity(
                    "vz_all_particles_kms",
                    {
                        "stars": {
                            "age": true_params["stars"]["age"],
                            "metallicity": true_params["stars"]["metallicity"],
                            "velocity": true_params["stars"]["velocity"]
                            .at[:, 2]
                            .add(vz_delta),
                        }
                    },
                    vz_delta,
                ),
            ],
        }
        stage_times["sensitivity_check_s"] = time.perf_counter() - t_sens
        print("[stage] completed finite perturbation sensitivity check", flush=True)
    else:
        sensitivity_summary = {
            "enabled": False,
            "target_l2": target_norm,
            "perturbations": [],
        }
        stage_times["sensitivity_check_s"] = 0.0
    identifiability_summary = _classify_sensitivity_identifiability(
        sensitivity_summary,
        rms_over_sigma_threshold=args.weak_rms_over_sigma_threshold,
        relative_l2_threshold=args.weak_relative_l2_threshold,
    )
    if args.gradient_signal_check:
        t_grad_signal = time.perf_counter()
        gradient_signal_summary = _constrained_gradient_signal_summary(
            forward_fn=lambda params: pipe.run_sharded(
                apply_params(static_data, params)
            ),
            target=target,
            sigma=sigma,
            params=init_params,
            zero_threshold=args.gradient_zero_threshold,
        )
        stage_times["gradient_signal_check_s"] = time.perf_counter() - t_grad_signal
        print("[stage] completed local gradient signal check", flush=True)
    else:
        gradient_signal_summary = {
            "enabled": False,
            "loss": None,
            "zero_threshold": args.gradient_zero_threshold,
            "parameters": [],
        }
        stage_times["gradient_signal_check_s"] = 0.0

    if args.auto_update_scales_from_sensitivity:
        auto_scales = _auto_update_scales_from_sensitivity(
            sensitivity_summary, max_scale=args.max_auto_update_scale
        )
        args.age_update_scale *= auto_scales["age"]
        args.metallicity_update_scale *= auto_scales["metallicity"]
        args.vz_update_scale *= auto_scales["vz"]
    else:
        auto_scales = {"age": 1.0, "metallicity": 1.0, "vz": 1.0}

    # Gradient consistency check on the full cube objective.
    def objective(params: dict[str, Any]) -> jnp.ndarray:
        pred = pipe.run_sharded(apply_params(static_data, params))
        return jnp.mean((pred - target) ** 2)

    objective_jit = jax.jit(objective)
    objective_and_grad_jit = jax.jit(jax.value_and_grad(objective))

    if args.skip_gradient_check:
        grad_summary = None
        print("[stage] skipped finite-difference gradient check", flush=True)
        stage_times["gradient_check_s"] = 0.0
    else:
        t_stage = time.perf_counter()
        print("[stage] running autodiff gradient", flush=True)
        _, grads_auto = objective_and_grad_jit(init_params)
        print("[stage] running finite-difference gradient (can be slow)", flush=True)
        grads_fd = finite_difference_grad(
            objective_jit,
            init_params,
            eps=1e-4,
            batch_size=16,
            jit_compile=True,
        )
        grad_summary = compare_gradients(grads_auto, grads_fd)
        stage_times["gradient_check_s"] = time.perf_counter() - t_stage
        print("[stage] completed gradient comparison", flush=True)

    print("[stage] preparing variational inference", flush=True)
    metallicity_upper = (
        args.synthetic_metallicity_max
        if args.synthetic_population_model == "sfh_ceh"
        else 0.04
    )
    transforms = build_age_metallicity_velocity_transforms(
        fixed_velocity_xy=static_data.stars.velocity[:, :2],
        age_lower=0.5,
        age_upper=12.0,
        metallicity_lower=5e-4,
        metallicity_upper=metallicity_upper,
        vz_lower=args.vz_lower_kms,
        vz_upper=args.vz_upper_kms,
    )
    sfh_ceh_penalty_fn = build_sfh_ceh_prior_penalty(
        age_min_gyr=0.5,
        age_max_gyr=12.0,
        sfh_tau_gyr=args.prior_sfh_tau_gyr,
        ceh_z_old=args.prior_ceh_z_old,
        ceh_z_young=args.prior_ceh_z_young,
        ceh_gamma=args.prior_ceh_gamma,
        ceh_sigma=args.prior_ceh_sigma,
    )
    ceh_relation_penalty_fn = build_ceh_relation_prior_penalty(
        age_min_gyr=0.5,
        age_max_gyr=12.0,
        ceh_z_old=args.prior_ceh_z_old,
        ceh_z_young=args.prior_ceh_z_young,
        ceh_gamma=args.prior_ceh_gamma,
        ceh_sigma=args.prior_ceh_sigma,
    )
    param_penalty_fn = _build_parameter_penalty_fn(
        particle_spaxels=spaxels,
        age_anchor=float(np.mean(np.asarray(true_params["stars"]["age"]))),
        metallicity_anchor=float(
            np.mean(np.asarray(true_params["stars"]["metallicity"]))
        ),
        smoothness_weight=args.prior_smoothness_weight,
        amr_weight=args.prior_amr_weight,
        mean_age_weight=args.prior_mean_age_weight,
        mean_metallicity_weight=args.prior_mean_met_weight,
        sfh_ceh_weight=args.prior_sfh_ceh_weight,
        ceh_relation_weight=args.prior_ceh_relation_weight,
        sfh_ceh_penalty_fn=sfh_ceh_penalty_fn,
        ceh_relation_penalty_fn=ceh_relation_penalty_fn,
    )
    warmup_result = None
    if args.map_warmup_steps > 0:
        print("[stage] running MAP warmup", flush=True)
        t_warmup = time.perf_counter()
        warmup_args = copy.copy(args)
        warmup_args.optimizer = args.map_warmup_optimizer
        warmup_args.vi_lr = args.map_warmup_lr
        warmup_optimizer = _build_vi_optimizer(warmup_args)
        warmup_result = optimize_variational_ifu_cube(
            pipeline=pipe,
            params_init=init_params,
            static_data=static_data,
            target=target,
            sigma=sigma,
            learning_rate=args.map_warmup_lr,
            max_steps=args.map_warmup_steps,
            tol=1e-8,
            num_samples=args.map_warmup_num_samples,
            beta_kl=args.map_warmup_beta_kl,
            init_log_std=args.map_warmup_init_log_std,
            transforms=transforms,
            optimizer=warmup_optimizer,
            param_penalty_fn=param_penalty_fn if args.map_warmup_use_priors else None,
            param_penalty_weight=1.0 if args.map_warmup_use_priors else 0.0,
            param_penalty_ramp_steps=args.prior_ramp_steps,
            normalize_loss=args.normalize_loss,
            seed=args.seed,
        )
        init_params = warmup_result.best_posterior_mean_constrained_params
        stage_times["map_warmup_s"] = time.perf_counter() - t_warmup
        print("[stage] completed MAP warmup", flush=True)
    else:
        stage_times["map_warmup_s"] = 0.0

    posterior_block_couplings = (
        [("stars", "age"), ("stars", "metallicity"), ("stars", "velocity")]
        if args.posterior_block
        else None
    )

    print("[stage] running main variational inference", flush=True)
    t_stage = time.perf_counter()
    optimizer = _build_vi_optimizer(args)
    vi_result = optimize_variational_ifu_cube(
        pipeline=pipe,
        params_init=init_params,
        static_data=static_data,
        target=target,
        sigma=sigma,
        learning_rate=args.vi_lr,
        max_steps=args.vi_steps,
        tol=1e-8,
        num_samples=args.num_vi_samples,
        beta_kl=args.beta_kl,
        init_log_std=args.init_log_std,
        transforms=transforms,
        optimizer=optimizer,
        param_penalty_fn=param_penalty_fn,
        param_penalty_weight=1.0,
        param_penalty_ramp_steps=args.prior_ramp_steps,
        normalize_loss=args.normalize_loss,
        posterior_rank=args.posterior_rank,
        posterior_block_couplings=posterior_block_couplings,
        prior_std=args.prior_std,
        seed=args.seed,
    )
    stage_times["variational_inference_s"] = time.perf_counter() - t_stage
    print("[stage] completed variational inference", flush=True)

    print("[stage] sampling posterior predictive cubes", flush=True)
    t_stage = time.perf_counter()
    samples = sample_posterior_predictive_cubes(
        pipeline=pipe,
        posterior_mean_params=vi_result.posterior_mean_params,
        posterior_log_std_params=vi_result.posterior_log_std_params,
        static_data=static_data,
        num_samples=args.num_posterior_samples,
        transforms=transforms,
        seed=args.seed + 1,
    )
    pred_summary = summarize_predictive_cube_samples(samples)

    # Posterior parameter samples in constrained (physical) space. These are the
    # inputs to cross-seed calibration/coverage diagnostics (see
    # scripts/run_vi_calibration.py); persisting them keeps posterior *widths*,
    # not just the point estimate, available downstream.
    posterior_param_samples = sample_posterior_parameters(
        posterior_mean_params=vi_result.posterior_mean_params,
        posterior_log_std_params=vi_result.posterior_log_std_params,
        num_samples=args.num_posterior_samples,
        transforms=transforms,
        seed=args.seed + 2,
    )
    post_samples_age = np.asarray(posterior_param_samples["stars"]["age"])
    post_samples_met = np.asarray(posterior_param_samples["stars"]["metallicity"])
    post_samples_vz = np.asarray(posterior_param_samples["stars"]["velocity"][:, :, 2])

    residuals = compute_residual_products(
        prediction=pred_summary["mean"], target=target, sigma=sigma
    )
    metrics = summarize_masked_metrics(prediction=pred_summary["mean"], target=target)
    stage_times["posterior_diagnostics_s"] = time.perf_counter() - t_stage
    print("[stage] computed posterior diagnostics", flush=True)

    truth_age = np.asarray(true_params["stars"]["age"])
    truth_met = np.asarray(true_params["stars"]["metallicity"])
    truth_vz = np.asarray(true_params["stars"]["velocity"][:, 2])
    fit_age = np.asarray(vi_result.posterior_mean_constrained_params["stars"]["age"])
    fit_met = np.asarray(
        vi_result.posterior_mean_constrained_params["stars"]["metallicity"]
    )
    fit_vz = np.asarray(
        vi_result.posterior_mean_constrained_params["stars"]["velocity"][:, 2]
    )

    recovery = {
        "age_mae": float(np.mean(np.abs(fit_age - truth_age))),
        "age_rmse": float(np.sqrt(np.mean((fit_age - truth_age) ** 2))),
        "metallicity_mae": float(np.mean(np.abs(fit_met - truth_met))),
        "metallicity_rmse": float(np.sqrt(np.mean((fit_met - truth_met) ** 2))),
        "vz_mae_kms": float(np.mean(np.abs(fit_vz - truth_vz))),
        "vz_rmse_kms": float(np.sqrt(np.mean((fit_vz - truth_vz) ** 2))),
    }
    final_params = vi_result.posterior_mean_constrained_params

    def _prior_eval(params: dict[str, Any]) -> dict[str, float]:
        return {
            "sfh_ceh_penalty": float(sfh_ceh_penalty_fn(params)),
            "ceh_relation_penalty": float(ceh_relation_penalty_fn(params)),
            "combined_weighted_penalty": float(param_penalty_fn(params)),
        }

    prior_diagnostics = {
        "truth": _prior_eval(true_params),
        "initial": _prior_eval(initial_params),
        "fit": _prior_eval(final_params),
    }
    diagnostic_spaxels = _mass_weighted_diagnostic_spaxel_maps(
        spaxels=np.asarray(spaxels),
        mass=np.asarray(ics.mass),
        truth_age=truth_age,
        fit_age=fit_age,
        truth_met=truth_met,
        fit_met=fit_met,
        truth_vz=truth_vz,
        fit_vz=fit_vz,
        nx=args.nx,
        ny=args.ny,
    )
    diagnostic_spaxel_maps = diagnostic_spaxels["maps"]
    diagnostic_spaxel_recovery = diagnostic_spaxels["metrics"]

    t_stage = time.perf_counter()
    np.savez(
        out / "science_cycle_outputs.npz",
        coords_xy=np.asarray(ics.coords_xy),
        velocity_xyz=np.asarray(ics.velocity_xyz),
        spaxels=np.asarray(spaxels),
        mass=np.asarray(ics.mass),
        true_age=truth_age,
        true_metallicity=truth_met,
        true_vz=truth_vz,
        fit_age=fit_age,
        fit_metallicity=fit_met,
        fit_vz=fit_vz,
        post_samples_age=post_samples_age,
        post_samples_metallicity=post_samples_met,
        post_samples_vz=post_samples_vz,
        diagnostic_spaxel_mass_map=diagnostic_spaxel_maps["mass"],
        diagnostic_spaxel_true_age_map=diagnostic_spaxel_maps["true_age"],
        diagnostic_spaxel_fit_age_map=diagnostic_spaxel_maps["fit_age"],
        diagnostic_spaxel_true_metallicity_map=diagnostic_spaxel_maps[
            "true_metallicity"
        ],
        diagnostic_spaxel_fit_metallicity_map=diagnostic_spaxel_maps["fit_metallicity"],
        diagnostic_spaxel_true_vz_map=diagnostic_spaxel_maps["true_vz"],
        diagnostic_spaxel_fit_vz_map=diagnostic_spaxel_maps["fit_vz"],
        target_cube=np.asarray(target),
        pred_mean_cube=np.asarray(pred_summary["mean"]),
        residual_cube=np.asarray(residuals["residual"]),
        chi2_cube=np.asarray(residuals["chi2"]),
        vi_objective=np.asarray(vi_result.objective_history),
        vi_reconstruction=np.asarray(vi_result.reconstruction_history),
        vi_kl=np.asarray(vi_result.kl_history),
        vi_grad_norm=np.asarray(vi_result.grad_norm_history),
    )
    stage_times["write_npz_s"] = time.perf_counter() - t_stage

    summary = {
        "sampler": sampler,
        "config": {
            "n_particles": args.n_particles,
            "cube_shape": [args.nx, args.ny, args.nw],
            "noise_level": args.noise_level,
            "sigma_floor": args.sigma_floor,
            "noise_relative": args.noise_relative,
            "noise_poisson_scale": args.noise_poisson_scale,
            "add_observational_noise": args.add_observational_noise,
            "effective_sigma_level": sigma_level,
            "ic_preset": args.ic_preset,
            "init_mode": args.init_mode,
            "ssp_template_name": args.ssp_template_name,
            "ssp_age_gyr": args.ssp_age_gyr,
            "ssp_metallicity": args.ssp_metallicity,
            "vi_steps": args.vi_steps,
            "vi_lr": args.vi_lr,
            "num_vi_samples": args.num_vi_samples,
            "init_log_std": args.init_log_std,
            "posterior_rank": args.posterior_rank,
            "posterior_block": args.posterior_block,
            "num_posterior_samples": args.num_posterior_samples,
            "beta_kl": args.beta_kl,
            "prior_std": args.prior_std,
            "normalize_loss": args.normalize_loss,
            "optimizer": args.optimizer,
            "adam_b1": args.adam_b1,
            "adam_b2": args.adam_b2,
            "adam_eps": args.adam_eps,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "age_update_scale": args.age_update_scale,
            "metallicity_update_scale": args.metallicity_update_scale,
            "vz_update_scale": args.vz_update_scale,
            "auto_update_scales_from_sensitivity": (
                args.auto_update_scales_from_sensitivity
            ),
            "max_auto_update_scale": args.max_auto_update_scale,
            "auto_update_scale_multipliers": auto_scales,
            "lbfgs_memory_size": args.lbfgs_memory_size,
            "lbfgs_disable_linesearch": args.lbfgs_disable_linesearch,
            "map_warmup_steps": args.map_warmup_steps,
            "map_warmup_optimizer": args.map_warmup_optimizer,
            "map_warmup_lr": args.map_warmup_lr,
            "map_warmup_init_log_std": args.map_warmup_init_log_std,
            "map_warmup_num_samples": args.map_warmup_num_samples,
            "map_warmup_beta_kl": args.map_warmup_beta_kl,
            "map_warmup_use_priors": args.map_warmup_use_priors,
            "sensitivity_check": args.sensitivity_check,
            "gradient_signal_check": args.gradient_signal_check,
            "gradient_zero_threshold": args.gradient_zero_threshold,
            "sensitivity_age_delta": args.sensitivity_age_delta,
            "sensitivity_metallicity_delta": args.sensitivity_metallicity_delta,
            "sensitivity_vz_delta_kms": args.sensitivity_vz_delta_kms,
            "weak_rms_over_sigma_threshold": args.weak_rms_over_sigma_threshold,
            "weak_relative_l2_threshold": args.weak_relative_l2_threshold,
            "agama_qjr": args.agama_qjr,
            "agama_qjphi": args.agama_qjphi,
            "seed": args.seed,
            "forward_model": args.forward_model,
            "pipeline_name": args.pipeline_name,
            "telescope_name": args.telescope_name,
            "native_resize_mode": args.native_resize_mode,
            "galaxy_dist_z": args.galaxy_dist_z,
            "age_basis_scale": args.age_basis_scale,
            "met_basis_scale": args.met_basis_scale,
            "age_response_scale": args.age_response_scale,
            "met_response_pivot": args.met_response_pivot,
            "met_response_scale": args.met_response_scale,
            "inference_metallicity_upper": metallicity_upper,
            "vz_lower_kms": args.vz_lower_kms,
            "vz_upper_kms": args.vz_upper_kms,
            "prior_smoothness_weight": args.prior_smoothness_weight,
            "prior_amr_weight": args.prior_amr_weight,
            "prior_mean_age_weight": args.prior_mean_age_weight,
            "prior_mean_met_weight": args.prior_mean_met_weight,
            "prior_ramp_steps": args.prior_ramp_steps,
            "prior_sfh_ceh_weight": args.prior_sfh_ceh_weight,
            "prior_ceh_relation_weight": args.prior_ceh_relation_weight,
            "prior_sfh_tau_gyr": args.prior_sfh_tau_gyr,
            "prior_ceh_z_old": args.prior_ceh_z_old,
            "prior_ceh_z_young": args.prior_ceh_z_young,
            "prior_ceh_gamma": args.prior_ceh_gamma,
            "prior_ceh_sigma": args.prior_ceh_sigma,
            "synthetic_population_model": args.synthetic_population_model,
            "synthetic_age_min_gyr": args.synthetic_age_min_gyr,
            "synthetic_age_max_gyr": args.synthetic_age_max_gyr,
            "synthetic_sfh_tau_gyr": args.synthetic_sfh_tau_gyr,
            "synthetic_ceh_z_old": args.synthetic_ceh_z_old,
            "synthetic_ceh_z_young": args.synthetic_ceh_z_young,
            "synthetic_ceh_gamma": args.synthetic_ceh_gamma,
            "synthetic_ceh_sigma": args.synthetic_ceh_sigma,
            "synthetic_metallicity_min": args.synthetic_metallicity_min,
            "synthetic_metallicity_max": args.synthetic_metallicity_max,
        },
        "provenance": {
            "wavelength_template_source": template_source,
            "diagnostic_spaxels_source": diagnostic_spaxels_source,
        },
        "gradient_check": (
            {
                "enabled": True,
                "max_abs_error": float(grad_summary.max_abs_error),
                "relative_l2_error": float(grad_summary.relative_l2_error),
            }
            if grad_summary is not None
            else {
                "enabled": False,
                "max_abs_error": None,
                "relative_l2_error": None,
            }
        ),
        "sensitivity_check": _to_jsonable(sensitivity_summary),
        "gradient_signal_check": _to_jsonable(gradient_signal_summary),
        "identifiability": _to_jsonable(identifiability_summary),
        "vi": {
            "final_objective": float(vi_result.final_objective),
            "final_reconstruction": float(vi_result.final_reconstruction),
            "final_kl": float(vi_result.final_kl),
            "steps_run": int(vi_result.steps_run),
            "converged": bool(vi_result.converged),
        },
        "map_warmup": (
            {
                "enabled": True,
                "final_objective": float(warmup_result.final_objective),
                "final_reconstruction": float(warmup_result.final_reconstruction),
                "final_kl": float(warmup_result.final_kl),
                "best_objective": float(warmup_result.best_objective),
                "best_step": int(warmup_result.best_step),
                "steps_run": int(warmup_result.steps_run),
                "converged": bool(warmup_result.converged),
            }
            if warmup_result is not None
            else {
                "enabled": False,
                "final_objective": None,
                "final_reconstruction": None,
                "final_kl": None,
                "best_objective": None,
                "best_step": None,
                "steps_run": 0,
                "converged": False,
            }
        ),
        "recovery_interpretation": _recovery_interpretation(
            spaxels=np.asarray(spaxels),
            identifiability_summary=identifiability_summary,
        ),
        "recovery": recovery,
        "diagnostic_spaxel_recovery": diagnostic_spaxel_recovery,
        "prior_diagnostics": _to_jsonable(prior_diagnostics),
        "metrics": _to_jsonable(metrics),
        "timings_seconds": {
            **{k: float(v) for k, v in stage_times.items()},
            "total_runtime_s": float(time.perf_counter() - t0_total),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
