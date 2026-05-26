#!/usr/bin/env python
"""Run a realistic synthetic VI cycle for Rubix science validation.

This script creates a synthetic galaxy from particle ICs (optionally AGAMA-backed),
constructs a mock IFU cube with simple physically-motivated assumptions, runs VI,
and writes diagnostics useful for science verification and papers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ParticleICs:
    coords_xy: jnp.ndarray
    velocity_xyz: jnp.ndarray
    mass: jnp.ndarray
    age: jnp.ndarray
    metallicity: jnp.ndarray


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
        wavelength_template: jnp.ndarray,
        nx: int,
        ny: int,
        noise_level: float = 0.02,
    ):
        self.particle_spaxels = particle_spaxels  # (P, 2) integer ix, iy
        self.particle_mass = particle_mass  # (P,)
        self.wavelength_template = wavelength_template  # (W,)
        self.nx = nx
        self.ny = ny
        self.nw = int(wavelength_template.shape[0])
        self.noise_level = float(noise_level)

    def run_sharded(self, rubixdata: Any) -> jnp.ndarray:
        age = rubixdata.stars.age
        metallicity = rubixdata.stars.metallicity

        # Smooth synthetic response factors for age/Z around physical ranges.
        age_factor = 0.8 + 0.4 * jnp.tanh((age - 5.0) / 4.0)
        met_factor = 0.7 + 0.5 * jnp.sqrt(jnp.clip(metallicity, 1e-8, None) / 0.02)
        amp = self.particle_mass * age_factor * met_factor

        cube = jnp.zeros(
            (self.nx, self.ny, self.nw), dtype=self.wavelength_template.dtype
        )

        def add_particle(carry, p):
            ix = self.particle_spaxels[p, 0]
            iy = self.particle_spaxels[p, 1]
            spectrum = amp[p] * self.wavelength_template
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


def _coords_to_spaxels(coords_xy: jnp.ndarray, nx: int, ny: int) -> jnp.ndarray:
    x = np.asarray(coords_xy[:, 0])
    y = np.asarray(coords_xy[:, 1])

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    ix = np.clip((x_norm * (nx - 1)).astype(np.int32), 0, nx - 1)
    iy = np.clip((y_norm * (ny - 1)).astype(np.int32), 0, ny - 1)
    return jnp.asarray(np.stack([ix, iy], axis=1))


def _build_custom_wavelength_template(nw: int) -> jnp.ndarray:
    """Fallback synthetic wavelength template used when SSP lookup is unavailable."""
    w = np.linspace(3600.0, 7000.0, nw, dtype=np.float32)
    base = 0.8 + 0.15 * np.sin((w - w.min()) / 500.0)

    def gauss(mu: float, sig: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((w - mu) / sig) ** 2)

    template = base - gauss(4861.0, 4.5, 0.08) - gauss(6563.0, 5.5, 0.06)
    return jnp.asarray(template / np.mean(template), dtype=jnp.float32)


def _build_wavelength_template(
    nw: int,
    ssp_template_name: str = "BruzualCharlot2003",
    ssp_age_gyr: float | None = None,
    ssp_metallicity: float | None = None,
) -> tuple[jnp.ndarray, str]:
    """Build wavelength template from RUBIX SSP lookup tables.

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
        spec_native = np.asarray(lookup(target_met, target_age), dtype=np.float64)

        wave_out = np.linspace(wave_grid.min(), wave_grid.max(), nw, dtype=np.float64)
        spec_out = np.interp(wave_out, wave_grid, spec_native)
        spec_out = spec_out / (np.mean(spec_out) + 1e-12)
        return jnp.asarray(spec_out, dtype=jnp.float32), "ssp_lookup"
    except Exception:
        return _build_custom_wavelength_template(nw), "custom_fallback"


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


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
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
    parser.add_argument("--n-particles", type=int, default=64)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--nw", type=int, default=64)
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument("--vi-steps", type=int, default=300)
    parser.add_argument("--vi-lr", type=float, default=2e-2)
    parser.add_argument("--num-vi-samples", type=int, default=4)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
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
    return parser.parse_args()


def main() -> None:
    from rubix.core.data import RubixData, StarsData
    from rubix.inference import (
        compare_gradients,
        compute_residual_products,
        finite_difference_grad,
        optimize_variational_ifu_cube,
        sample_posterior_predictive_cubes,
        summarize_masked_metrics,
        summarize_predictive_cube_samples,
    )

    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ics, sampler = sample_particle_ics(
        n_particles=args.n_particles,
        seed=args.seed,
        prefer_agama=not args.no_agama,
        agama_qjr=args.agama_qjr,
        agama_qjphi=args.agama_qjphi,
    )
    print(
        f"[stage] sampled ICs via {sampler} (n_particles={args.n_particles})",
        flush=True,
    )
    spaxels = _coords_to_spaxels(ics.coords_xy, nx=args.nx, ny=args.ny)
    template, template_source = _build_wavelength_template(
        args.nw,
        ssp_template_name=args.ssp_template_name,
        ssp_age_gyr=args.ssp_age_gyr,
        ssp_metallicity=args.ssp_metallicity,
    )
    print(f"[stage] wavelength template source: {template_source}", flush=True)

    pipe = LocalSpaxelSpectralPipeline(
        particle_spaxels=spaxels,
        particle_mass=ics.mass,
        wavelength_template=template,
        nx=args.nx,
        ny=args.ny,
        noise_level=args.noise_level,
    )

    static_data = _make_static_data(ics)

    true_params = {"stars": {"age": ics.age, "metallicity": ics.metallicity}}
    init_params = {
        "stars": {
            "age": jnp.full_like(ics.age, jnp.mean(ics.age)),
            "metallicity": jnp.full_like(ics.metallicity, jnp.mean(ics.metallicity)),
        }
    }

    target = pipe.run_sharded(
        RubixData(
            galaxy=static_data.galaxy,
            stars=StarsData(
                coords=static_data.stars.coords,
                velocity=static_data.stars.velocity,
                mass=static_data.stars.mass,
                age=true_params["stars"]["age"],
                metallicity=true_params["stars"]["metallicity"],
            ),
            gas=static_data.gas,
        )
    )
    print("[stage] built synthetic target cube", flush=True)

    sigma = jnp.ones_like(target) * args.noise_level

    # Gradient consistency check on the full cube objective.
    def objective(params: dict[str, Any]) -> jnp.ndarray:
        pred = pipe.run_sharded(
            RubixData(
                galaxy=static_data.galaxy,
                stars=StarsData(
                    coords=static_data.stars.coords,
                    velocity=static_data.stars.velocity,
                    mass=static_data.stars.mass,
                    age=params["stars"]["age"],
                    metallicity=params["stars"]["metallicity"],
                ),
                gas=static_data.gas,
            )
        )
        return jnp.mean((pred - target) ** 2)

    objective_jit = jax.jit(objective)
    objective_and_grad_jit = jax.jit(jax.value_and_grad(objective))

    if args.skip_gradient_check:
        grad_summary = None
        print("[stage] skipped finite-difference gradient check", flush=True)
    else:
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
        print("[stage] completed gradient comparison", flush=True)

    print("[stage] running variational inference", flush=True)
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
        seed=args.seed,
    )
    print("[stage] completed variational inference", flush=True)

    print("[stage] sampling posterior predictive cubes", flush=True)
    samples = sample_posterior_predictive_cubes(
        pipeline=pipe,
        posterior_mean_params=vi_result.posterior_mean_params,
        posterior_log_std_params=vi_result.posterior_log_std_params,
        static_data=static_data,
        num_samples=16,
        seed=args.seed + 1,
    )
    pred_summary = summarize_predictive_cube_samples(samples)
    residuals = compute_residual_products(
        prediction=pred_summary["mean"], target=target, sigma=sigma
    )
    metrics = summarize_masked_metrics(prediction=pred_summary["mean"], target=target)
    print("[stage] computed posterior diagnostics", flush=True)

    truth_age = np.asarray(true_params["stars"]["age"])
    truth_met = np.asarray(true_params["stars"]["metallicity"])
    fit_age = np.asarray(vi_result.posterior_mean_params["stars"]["age"])
    fit_met = np.asarray(vi_result.posterior_mean_params["stars"]["metallicity"])

    recovery = {
        "age_mae": float(np.mean(np.abs(fit_age - truth_age))),
        "age_rmse": float(np.sqrt(np.mean((fit_age - truth_age) ** 2))),
        "metallicity_mae": float(np.mean(np.abs(fit_met - truth_met))),
        "metallicity_rmse": float(np.sqrt(np.mean((fit_met - truth_met) ** 2))),
    }

    np.savez(
        out / "science_cycle_outputs.npz",
        coords_xy=np.asarray(ics.coords_xy),
        spaxels=np.asarray(spaxels),
        mass=np.asarray(ics.mass),
        true_age=truth_age,
        true_metallicity=truth_met,
        fit_age=fit_age,
        fit_metallicity=fit_met,
        target_cube=np.asarray(target),
        pred_mean_cube=np.asarray(pred_summary["mean"]),
        residual_cube=np.asarray(residuals["residual"]),
        chi2_cube=np.asarray(residuals["chi2"]),
        vi_objective=np.asarray(vi_result.objective_history),
        vi_reconstruction=np.asarray(vi_result.reconstruction_history),
        vi_kl=np.asarray(vi_result.kl_history),
        vi_grad_norm=np.asarray(vi_result.grad_norm_history),
    )

    summary = {
        "sampler": sampler,
        "config": {
            "n_particles": args.n_particles,
            "cube_shape": [args.nx, args.ny, args.nw],
            "noise_level": args.noise_level,
            "ssp_template_name": args.ssp_template_name,
            "ssp_age_gyr": args.ssp_age_gyr,
            "ssp_metallicity": args.ssp_metallicity,
            "vi_steps": args.vi_steps,
            "vi_lr": args.vi_lr,
            "num_vi_samples": args.num_vi_samples,
            "beta_kl": args.beta_kl,
            "agama_qjr": args.agama_qjr,
            "agama_qjphi": args.agama_qjphi,
            "seed": args.seed,
        },
        "provenance": {
            "wavelength_template_source": template_source,
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
        "vi": {
            "final_objective": float(vi_result.final_objective),
            "final_reconstruction": float(vi_result.final_reconstruction),
            "final_kl": float(vi_result.final_kl),
            "steps_run": int(vi_result.steps_run),
            "converged": bool(vi_result.converged),
        },
        "recovery": recovery,
        "metrics": _to_jsonable(metrics),
    }
    (out / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
