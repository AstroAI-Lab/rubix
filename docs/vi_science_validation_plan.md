# RUBIX VI Science Validation Plan

This plan defines the staged work needed to make RUBIX variational inference
credible for real science use. The central rule is that a good cube fit is not
by itself enough: each stage must prove what physical quantities are identifiable,
calibrated, and robust under the relevant RUBIX forward model.

## Phase 0: Environment and Runner Reliability

Goal: every VI result is reproducible from a known runtime.

Acceptance gates:

- Commands run in the `rubix` conda environment.
- JAX backend selection is explicit. For local CPU validation, use
  `JAX_PLATFORMS=cpu`.
- `python -c "import rubix, jax, optax, h5py"` succeeds.
- `pytest --collect-only` and targeted inference tests do not hang.
- The realistic synthetic VI script can be imported and invoked with `--help`.
- The notebook kernel and VS Code interpreter point at the `rubix` environment.

Evidence to record:

- Python executable, conda environment, JAX version, active JAX devices.
- Exact command lines for validation runs.
- Runtime and pass/fail status for targeted checks.

## Phase 1: Forward Model Contract Tests

Goal: prove that VI optimizes the same RUBIX-native forward model used to
generate the synthetic truth.

Acceptance gates:

- A native forward adapter returns the telescope-native cube shape unless an
  explicit, documented downsampling operator is requested.
- Perturbing stellar age, metallicity, mass, and line-of-sight velocity changes
  the cube in measurable and expected ways.
- Autodiff gradients match finite differences on a tiny native configuration.
- Posterior predictive sampling applies the same parameter transforms used by
  VI optimization.

Recommended tests:

- One-particle SSP sensitivity test.
- One-particle Doppler sensitivity test with km/s-scale velocity.
- Tiny native pipeline gradient check with PSF/LSF disabled, then enabled.
- Transform-aware posterior predictive regression test.

## Phase 2: Identifiability Ladder

Goal: separate optimizer bugs from inverse problems that are intrinsically
underdetermined.

Run the following ladder in order:

1. One particle, one spaxel, no PSF, no LSF, no noise.
2. One particle per spaxel, no PSF, no LSF, no noise.
3. Multiple particles per spaxel, no PSF, no LSF, no noise.
4. Add LSF.
5. Add PSF.
6. Add observational noise.
7. Add realistic particle population and aperture geometry.

Acceptance gates:

- Particle-level recovery is required only for stages where the observation
  identifies particle-level labels.
- Once particles are aggregated into shared spaxels, primary metrics shift to
  mass-weighted spaxel quantities and posterior predictive cube diagnostics.
- Failures include Jacobian or sensitivity diagnostics showing whether the
  missing recovery signal is numerical or physical.

Metrics:

- Cube negative log likelihood and residual summaries.
- Particle age, metallicity, and velocity recovery where identifiable.
- Mass-weighted spaxel age, metallicity, and velocity recovery.
- Posterior interval coverage on repeated simulations.
- Gradient norm and update norm traces.

## Phase 3: Synthetic Physics Calibration

Goal: make synthetic experiments physically meaningful enough to support real
science claims.

Acceptance gates:

- AGAMA or fallback velocities are in km/s before entering RUBIX Doppler logic.
- Particle coordinates are scaled or filtered to the telescope aperture instead
  of being silently clipped into edge spaxels.
- Synthetic age and metallicity priors are documented and separable from the
  inference priors being tested.
- Noise models are tied to flux or S/N, not only a constant sigma cube.
- Wavelength handling uses the native telescope grid or a documented spectral
  coarsening operator.

## Phase 4: VI Objective and Posterior Calibration

Goal: posterior summaries are calibrated, not just numerically stable.

Acceptance gates:

- Mean-field VI matches a MAP solution on deterministic low-noise cases.
- Tiny cases compare against grid or MCMC reference posteriors.
- Simulation-based calibration passes on low-dimensional problems.
- Prior predictive and posterior predictive checks are generated for each
  science benchmark.
- KL and physical priors are scaled in interpretable units.

Recommended additions:

- Optional physics-space prior terms for age, metallicity, and velocity.
- Per-parameter posterior summaries in constrained space.
- Calibration plots for credible intervals.
- Guardrails on posterior collapse and unconstrained-space saturation.

## Phase 5: Science Readiness Benchmarks

Goal: define the evidence required before using RUBIX VI for a science result.

Required benchmark outputs:

- A recovery table across the identifiability ladder.
- Posterior predictive residual plots.
- Calibration and coverage plots.
- Runtime and memory profiles.
- Documented failure modes and non-identifiable regimes.
- Versioned guardrail thresholds for objective quality, calibration, and runtime.

Minimum readiness criteria:

- Repeated synthetic experiments recover identifiable quantities within declared
  tolerances.
- Non-identifiable quantities are reported as such and are not treated as
  particle-level successes.
- Posterior predictive checks pass on held-out synthetic configurations.
- Results are reproducible from a committed config and environment record.

## Immediate Work Package

Completed/active findings:

- Local validation commands are standardized on:

   ```bash
   conda run -n rubix env JAX_PLATFORMS=cpu <command>
   ```

- The BC03 template age metadata must be treated as log-years and converted to
  Gyr. Loading it as linear Gyr produced zero local native age gradients around
  the VI mean initialization.
- The native 2x2 test telescope (`TESTGRADIENT_2X2`) is the current fast
  full-cube validation rung.
- The runner now records finite perturbation sensitivity and local autodiff
  gradient signal. Both are needed: finite perturbations show nonlocal cube
  response, while local gradient signal tells us whether gradient VI can move
  from the chosen initialization.
- For the native 2x2 one-particle-per-spaxel case at `sigma_floor=1e-6`, age
  and metallicity are identifiable by finite sensitivity, while line-of-sight
  velocity remains weak.
- Optimizer sweep result: a 5-step L-BFGS warmup followed by Adam with
  `age_update_scale=20` gives the best current native 2x2 age recovery
  (`age_mae ~0.64 Gyr`). Plain Adam remains much worse (`age_mae ~2.29 Gyr`).
  Metallicities still collapse toward a central compromise, indicating an
  age-metallicity degeneracy rather than a mere cube-fit failure.
- The SFH+CEH population model now applies to deterministic ladder presets as
  well as sampled populations when `--synthetic-population-model sfh_ceh` is
  selected. This makes prior-regularized tests fair: truth and regularizer can
  come from the same age-metallicity family.
- On the native 2x2 SFH+CEH truth case, `age_update_scale=20` with CEH prior
  weights in the `0.1-1.0` range starts to improve metallicity recovery. Weight
  `0.1` gives a small metallicity gain with little age cost; weight `1.0`
  improves metallicity more but begins to bias age. A full CEH weight sweep is
  useful, but can be deferred until the smaller reference posterior checks are
  in place.
- A native one-particle reference grid now runs for the full RUBIX native
  pipeline. For the legacy single-particle case with truth
  `(age=7.5 Gyr, Z=0.008, vz=80 km/s)`, a `60 x 60` age-metallicity grid at
  fixed true velocity found a best grid point near the truth
  `(age~7.32 Gyr, Z~0.00903)`, while the weighted marginal posterior remained
  broad along the age-metallicity ridge
  `(age p16/p50/p84 ~7.52/8.88/10.44 Gyr, Z p16/p50/p84
  ~0.00259/0.00565/0.00855)`.
- The matching native single-particle VI run with Adam and `age_update_scale=20`
  recovered age and metallicity accurately
  (`age_mae~0.005 Gyr`, `metallicity_mae~2.5e-5`) and fit the cube nearly
  exactly (`mse~3.2e-16`), but still left line-of-sight velocity weakly
  constrained (`vz_mae~41 km/s`). This supports treating velocity as a
  physical identifiability problem in this fast telescope setup, not merely an
  optimizer failure.
- A coarse native `24 x 24 x 41` age-metallicity-velocity reference grid over
  `vz=-120..240 km/s` found its best velocity near the truth
  (`best vz=69 km/s`, truth `80 km/s`), but the velocity marginal remained very
  broad (`p16/p50/p84 ~ -66/60/186 km/s`). The profile likelihood changes by
  only `~8.5e-6` between `vz=69` and `vz=78 km/s` and by only `~1.2e-4` out to
  roughly `vz=33..105 km/s`, confirming that this telescope/configuration does
  not provide a sharp velocity constraint for the current single-particle test.
- The reference-grid runner now records per-parameter profile-likelihood
  summaries so future checks expose broad or flat likelihood directions directly
  in `summary.json`.
- A comparison wrapper now runs matched native single-particle reference grids
  and VI jobs across seeds and writes `comparison_summary.json`. A tiny smoke
  run verified the end-to-end harness and summary extraction; science-grade
  repeated-seed use should use a finer grid and enough VI steps to match the
  validated single-particle run.
- The first repeated-seed SFH+CEH comparison (`seeds=0,1`, native
  single-particle, `16 x 16 x 31` reference grid, `vi_steps=300`) completed
  without failures. VI recovered age and metallicity tightly
  (`mean age_mae~0.0015 Gyr`, `mean metallicity_mae~3.6e-6`) while velocity
  remained poor (`mean vz_mae~41 km/s`). The reference grid best velocities
  were closer to truth on average (`mean abs error~24 km/s`), but the
  profile-likelihood width at `delta<=1e-3` averaged `~120 km/s`, reinforcing
  that velocity is broadly constrained in this setup.
- The comparison wrapper also supports a native 2x2 diagnostic rung. On the
  SFH+CEH one-particle-per-spaxel case with `seeds=0,1`, `vi_steps=300`,
  `age_update_scale=20`, and no inference CEH prior, VI reached
  `mean age_mae~0.485 Gyr`, `mean metallicity_mae~0.00143`, and
  `mean vz_mae~74 km/s`, with velocity marked weakly identified in both seeds.
  Adding `prior_sfh_ceh_weight=0.1` gave a small metallicity and velocity gain
  (`mean metallicity_mae~0.00139`, `mean vz_mae~73.7 km/s`) at a small age cost
  (`mean age_mae~0.504 Gyr`). This supports using CEH as a regularizer, but not
  yet treating the current weight as calibrated.
- A native 2x2 block-profile diagnostic now scans one particle and one
  parameter at a time while holding the rest of the cube truth fixed. For
  SFH+CEH seed `0`, the no-prior profile found sharp local age/metallicity
  likelihoods near truth (`mean age best error~0.14 Gyr`,
  `mean metallicity best error~9.1e-5`) but broad velocity profiles
  (`mean vz width at delta<=1e-3 ~324 km/s`). Repeating the profile with
  `prior_sfh_ceh_weight=0.1` shifted some local age/metallicity optima away
  from the pure-likelihood truth (`mean age best error~0.27 Gyr`,
  `mean metallicity best error~7.3e-4`) while leaving velocity width unchanged.
  This makes the CEH tradeoff explicit: it regularizes the coupled VI solution
  but can bias local truth if weighted too strongly.
- The block-profile diagnostic can now scan around a fitted VI center loaded
  from `science_cycle_outputs.npz`. For native 2x2 SFH+CEH seed `0`, no-prior
  VI-centered profiles kept age/metallicity near the fitted ridge rather than
  returning to truth (`mean age best error~0.59 Gyr`, `mean metallicity best
  error~0.0026`, with small `delta_at_center` for metallicity). The
  CEH-regularized fit-centered profile behaved similarly but modestly improved
  metallicity (`mean metallicity best error~0.0016`). This distinguishes the
  remaining 2x2 error from a simple local-gradient failure: truth is locally
  sharp when the other particles are held at truth, but the coupled VI solution
  lands on a different age-metallicity ridge where one-at-a-time scans do not
  recover the true particle labels.
- Coupled 2D age-metallicity profiles are now available via
  `--scan-age-metallicity-pairs`. On native 2x2 SFH+CEH seed `0` with a coarse
  `16x16` global grid, truth-centered scans already found non-truth
  age-metallicity combinations with lower objective for all four particles
  (`mean age best error~0.22 Gyr`, `mean metallicity best error~0.0026`,
  `mean delta_at_truth~0.28`, `mean corr~-0.22`). Fit-centered scans improved
  over the VI center in objective (`mean delta_at_center~0.88`) but still
  preferred a biased region rather than returning to truth (`mean age best
  error~0.51 Gyr`, `mean metallicity best error~0.0042`). The coarse grid means
  the absolute offsets should not yet be overinterpreted, but the qualitative
  result is important: the native 2x2 failure is a coupled age-metallicity
  posterior-geometry problem, not just an Adam/L-BFGS convergence problem.
- The 2D profiler now supports local grids via
  `--age-metallicity-grid-mode={truth,center}` plus age/metallicity half-widths.
  With a local `25x25` grid around truth (`half_width_age=1 Gyr`,
  `half_width_Z=0.003`), truth-centered native 2x2 SFH+CEH profiles collapsed
  close to truth (`mean age best error~0.03 Gyr`, `mean metallicity best
  error~1.8e-4`, `mean delta_at_truth~0.022`). This shows the coarse global
  grid exaggerated the truth-centered displacement. However, when the rest of
  the cube was held at the VI fit, local scans still preferred biased coupled
  age-metallicity combinations: fit-local scans gave `mean age best
  error~0.55 Gyr`, `mean metallicity best error~0.0032`, while truth-local
  scans with the same fitted background reduced this to `mean age best
  error~0.32 Gyr`, `mean metallicity best error~0.0017`. The refined diagnosis
  is therefore: truth is locally recoverable in the truth basin, but the fitted
  multi-particle state creates a nearby low-objective basin/ridge that remains
  biased.
- `run_vi_block_profile.py` now prints a compact terminal summary by default
  while preserving the full per-grid payload in `summary.json`. Use
  `--print-full-summary` only when the full JSON dump is needed. Local 2D
  profile rows also record whether truth and center are inside the scanned
  window, preventing edge-nearest deltas from being mistaken for exact
  reference evaluations.
- The established chemical enrichment relation can now be used independently
  from the marginal SFH age prior via `build_ceh_relation_prior_penalty` and
  `--prior-ceh-relation-weight`. This isolates the age-metallicity regularizer
  from the age-distribution prior. On native 2x2 SFH+CEH seed `0`, the
  CEH-only relation penalty was lower at truth than at the no-prior VI fit
  (`1.25` vs `1.89`). In local fit-held 2D profiles, weight `0.1` reduced
  metallicity bias substantially: fit-local `mean metallicity best error`
  improved from `~0.0032` to `~0.0015`, and truth-local with fitted background
  improved from `~0.0017` to `~0.0011`. A full VI rerun with
  `--prior-ceh-relation-weight 0.1` improved seed-0 metallicity from
  `0.001753` to `0.001673`, close to the previous joint SFH+CEH `0.1` result
  (`0.001676`), while age remained essentially unchanged
  (`0.5489 -> 0.5438 Gyr`) and velocity stayed weak (`~74.8 km/s`). Repeating
  the CEH-only run over the same two seeds as the no-prior and joint-prior
  native 2x2 comparisons gave nearly the same aggregate as joint SFH+CEH
  weight `0.1`: no-prior `mean age_mae~0.485`,
  `mean metallicity_mae~0.001434`, `mean vz_mae~74.04`; joint SFH+CEH `0.1`
  `mean age_mae~0.504`, `mean metallicity_mae~0.001394`,
  `mean vz_mae~73.68`; CEH-only relation `0.1`
  `mean age_mae~0.505`, `mean metallicity_mae~0.001392`,
  `mean vz_mae~73.67`. This suggests the useful part of the prior at this rung
  is the relation itself, not the marginal age prior, but also that the
  relation prior alone is not enough to solve the age-metallicity basin.
- Velocity is present in the RUBIX native forward model, but the previous
  diagnostics were suppressing its evidence. The profile and VI runners now can
  use summed Gaussian data terms (`--likelihood-normalization sum` for block
  profiles and `--no-normalize-loss` for full VI). On the single-particle
  native velocity-isolation test, mean-normalized profiles stayed broad
  (`delta<=1e-3` width `~200 km/s` for both coarse `466`-bin and high-res
  `4652`-bin telescope grids). Summed likelihood made the same profile sharp:
  coarse grid widths were `0/20/90/270 km/s` at
  `delta<=1e-3/1e-2/1e-1/1`, and high-res widths were
  `0/0/30/90 km/s`. With summed loss and truth age/metallicity but `v_z`
  initialized to zero, Adam still stalled at `vz_mae~52.6 km/s`, while L-BFGS
  recovered velocity accurately (`~0.002 km/s` on the coarse grid and
  `~0.8 km/s` on the high-res grid). The velocity failure is therefore not an
  absence of RUBIX Doppler information; it is a likelihood normalization plus
  optimizer/scaling problem.

- **ELBO units are now calibrated by default.** The VI objective is
  `E_q[reconstruction] + beta_kl * KL(q || N(0, I))`. For posterior *widths* to
  be trustworthy this must equal the negative ELBO, which requires the summed
  (not per-voxel-mean) Gaussian NLL and `beta_kl = 1.0`. Accordingly the library
  defaults were changed: `optimize_variational_ifu_cube` now defaults to
  `normalize_loss=False` and `beta_kl=1.0`, the experiment config templates use
  the same pair, and calling the VI cube optimizer with `normalize_loss=True`
  and `beta_kl > 0` now emits a warning. Use `beta_kl=0` for an explicit MAP
  point estimate (mean meaningful, width not). The MAP-oriented diagnostic
  scripts that intentionally set `beta_kl=0` are unaffected; only their reported
  posterior *widths* should still be treated as uncalibrated.

- **Posterior calibration harness is now available (Phase 4).** The
  `rubix.inference.calibration` module computes central credible-interval
  coverage, standardized errors (`mean_z`, `rms_z`), and simulation-based
  calibration (SBC) rank statistics from posterior parameter samples in
  constrained space. The science-cycle runner now persists
  `post_samples_{age,metallicity,vz}` in `science_cycle_outputs.npz`, and
  `scripts/run_vi_calibration.py` aggregates repeated-seed runs into a coverage
  and SBC report. Unit tests confirm the diagnostics recover nominal coverage
  and uniform SBC ranks for a correctly specified Gaussian, and flag an
  under-dispersed (over-confident) posterior. This is the tooling for the
  Phase 4 gate "simulation-based calibration passes on low-dimensional problems"
  and "calibration plots for credible intervals"; it must now be *run* on the
  identifiability-ladder rungs with calibrated ELBO units to close the gate.

- **Structured posterior family added (addresses the coupled age-metallicity
  geometry).** `rubix.inference.posterior_family` implements a
  low-rank-plus-diagonal Gaussian posterior, ``Sigma = diag(exp(2 log_std)) +
  W W^T``, over the raveled unconstrained latent. It reduces exactly to the
  diagonal mean-field posterior when ``W = 0`` and costs ``O(D r + r^3)``.
  ``optimize_variational_posterior``/``optimize_variational_ifu_cube`` take a
  ``posterior_rank`` argument (0 = mean-field), and the science-cycle runner
  exposes ``--posterior-rank``. The optimizer samples and computes the KL with
  the full low-rank covariance, reports marginal widths in
  ``posterior_log_std_params`` (so diagonal downstream samplers keep correct
  marginals) and returns the factor in ``posterior_factor_params`` for joint
  correlated sampling. A unit test confirms that on a ridge likelihood the
  rank-1 posterior recovers the strong negative age-metallicity correlation a
  diagonal posterior cannot. Next: run the native 2x2 SFH+CEH rung with
  ``--posterior-rank 1`` (or a per-spaxel block rank) and compare recovery and
  calibration against the diagonal fit.

- **The velocity-capable recipe is now the science-cycle default.** Based on the
  finding that summed likelihood plus an L-BFGS/MAP warmup recovers line-of-sight
  velocity, ``run_realistic_synthetic_vi_cycle.py`` now defaults to the summed
  Gaussian likelihood (``--normalize-loss`` is opt-in), a 5-step L-BFGS warmup
  (``--map-warmup-steps 5``), and ``--beta-kl 1.0`` for a calibrated ELBO. Pass
  ``--normalize-loss --beta-kl 0 --map-warmup-steps 0`` to recover the old
  MAP-style per-voxel-mean recipe. A tiny end-to-end smoke run confirmed the new
  defaults execute and that the persisted posterior samples feed the calibration
  harness.

- **Flux/S-N-scaled noise model added (Phase 3 gate).**
  ``rubix.inference.flux_scaled_sigma`` builds a per-voxel sigma from
  ``sqrt((relative_noise*|flux|)^2 + poisson_scale*max(flux,0) + floor^2)``,
  reducing to the constant floor when the flux terms are zero. The science-cycle
  runner exposes ``--noise-relative`` (inverse bright-end S/N) and
  ``--noise-poisson-scale``; when either is set the assumed sigma cube is tied to
  flux instead of a hand-tuned constant. Default behavior is unchanged.

- **Robust best-step selection and objective caveats documented.** The
  ``best`` posterior mean/step is now selected on an exponential moving average
  of the stochastic objective (``best_selection_ema_decay``, default 0.9) so a
  single lucky Monte Carlo draw is no longer recorded as the best step; set the
  decay to 0 for deterministic optimizers such as L-BFGS. Docstrings now warn
  that (a) a nonzero Huber term makes the objective a non-log-likelihood so
  posterior widths are uncalibrated while it is on, and (b) the ``converged``
  flag is unreliable under a stochastic objective.

- **Transform/prior robustness cleanups.** ``VelocityZBoundsTransform`` now uses
  a single-channel ``(N, 1)`` unconstrained latent for ``v_z`` (x/y are held
  fixed and are no longer free latents), removing phantom parameters and their
  spurious KL contribution. ``SoftplusLowerBound.inverse`` uses the stable
  identity ``log(expm1(x)) = x + log1p(-exp(-x))`` to avoid overflow and
  nan-gradients for large values. The SFH age prior replaces its hard age clip
  with a soft quadratic barrier so out-of-range ages keep a restoring gradient.

- **Empirical follow-up: calibrated defaults vs recovery, and posterior-family
  effect on the native 2x2 SFH+CEH rung.** Running diagonal (`--posterior-rank
  0`) vs low-rank (`--posterior-rank 1`) at matched seeds gave *identical* mean
  recovery (`age_mae~3.13`, `metallicity_mae~0.00174`, `vz_mae~80`): the point
  estimate is set by the deterministic L-BFGS/MAP warmup, so the posterior family
  does not move it. The low-rank and block-covariance families are therefore
  *calibration* tools (correct correlated uncertainty), not recovery tools.
- **The calibrated default (`beta_kl=1.0`) badly hurts recovery here.** Rerunning
  with `--beta-kl 0` (MAP) recovered `age_mae~0.36` (vs `~3.13` at `beta_kl=1.0`),
  an ~8x improvement, and improved `vz_mae` (80 -> 59). Cause: the standard-normal
  prior in *unconstrained* space is informative for sigmoid-bounded parameters
  (it concentrates ages near the bound midpoint). Consequence:
  `run_realistic_synthetic_vi_cycle.py` now defaults to `--beta-kl 0` (MAP) for
  its recovery mission; calibrated posteriors are an explicit opt-in. The generic
  library default stays `beta_kl=1.0` (correct ELBO weight) since the prior is the
  user's to choose.
- **The calibration harness catches the miscalibration.** On the `beta_kl=1.0`
  low-rank runs, `run_vi_calibration.py` reported severe age under-coverage
  (`cov@0.9~0.375`, `rms_z~2.23`) and over-coverage for weakly-identified `vz`
  (`cov@0.9~1.0`), exactly the biased+over-confident signature expected from the
  informative unconstrained prior. This validates the Phase 4 tooling and makes a
  weakly-informative / physics-based prior over the unconstrained latents the next
  thing to fix before `beta_kl=1.0` posteriors can be treated as calibrated.
- **Per-particle block-covariance posterior added (math + tests).**
  `posterior_family.py` now provides `build_particle_block_index_map`,
  `init_block_cholesky`, `sample_block_gaussian`, `kl_block_to_standard_normal`,
  and `block_marginal_log_std`: a block-diagonal Gaussian giving each coupled
  group (e.g. per-particle age/metallicity/vz) its own dense covariance while the
  rest stay diagonal. It is the natural family for P independent per-particle
  ridges that a single global low-rank factor cannot represent. KL is verified
  against a dense reference and sampling against the empirical block covariance.
  Optimizer wiring is deferred until the unconstrained-prior issue is resolved so
  calibrated coverage can be evaluated meaningfully.

- **Prior width made configurable; the deeper issue is likelihood strength.** The
  KL now supports a configurable isotropic prior std (`prior_std`, default 1.0)
  in `kl_diag`/`kl_low_rank`/`kl_block` and both optimizer entry points, with the
  recipe defaulting calibrated runs to `prior_std = 1.814 = pi/sqrt(3)`. That is
  the variance of the logistic prior which induces an exactly uniform physical
  prior for any sigmoid-bounded parameter (bounds only rescale the physical
  value, not the latent), so a single value de-biases age, metallicity, and vz at
  once. Empirically, however, `prior_std` alone did *not* change recovery or
  coverage on the native 2x2 rung: with a near-noise-free target and an
  over-large assumed `sigma=0.02`, the summed reconstruction NLL at the fit is
  ~1e-4 while the KL is ~O(10), so `beta_kl=1` collapses the posterior onto the
  prior regardless of its width (`age_mae~3.13`). Sharpening the likelihood
  (`sigma=1e-5`) restored `beta_kl=1` recovery to `age_mae~0.376` (~MAP). So the
  dominant lever is likelihood strength / the assumed noise level, not prior
  width.
- **Calibration needs observational noise in the target.** A deterministic
  noise-free target makes the assumed `sigma` arbitrary *and* makes coverage/SBC
  ill-posed (no noise realizations to cover). `run_realistic_synthetic_vi_cycle.py`
  now has `--add-observational-noise`, which injects a seed-keyed noise
  realization at the assumed per-voxel `sigma`, giving a correctly-scaled
  likelihood and well-posed cross-seed coverage. Note the flux-scaled sigma floor
  is `max(noise_level, sigma_floor)`, so calibrated runs must lower
  `--noise-level` (the default 0.02 floor otherwise dominates and re-creates the
  prior-collapsed regime).
- **Coverage result (native 2x2, 5 noise-injected seeds, `sigma=1e-4`,
  `beta_kl=1`, `prior_std=1.814`, 20 pooled trials).** Moving from the
  prior-collapsed `sigma=0.02` regime to a data-constraining `sigma=1e-4`
  improved age coverage markedly: `cov@0.9` rose from `0.35` to `0.60`,
  `cov@0.95` to `0.80`, and the SBC reduced chi-square fell from `10.8` to `4.4`.
  Coverage is still below nominal with `rms_z~1.9` (age) and `~1.7`
  (metallicity), i.e. the posteriors remain ~2x too narrow. This is the expected
  mean-field VI variance underestimation, not a bug: the diagonal Gaussian cannot
  capture the coupled age-metallicity width. Line-of-sight velocity over-covers
  (`cov@0.9~1.0`, `rms_z~1.0`) because it is weakly identified (broad posterior).
  Conclusion: with a well-scaled likelihood the calibration pipeline is now
  well-posed and the prior is unbiased; closing the remaining gap needs a
  richer posterior (wire the block-covariance family into the optimizer) and/or
  importance-weighted width correction, not further prior tuning.
- **Block-covariance posterior wired in, but it does NOT close the marginal
  coverage gap (informative negative result).** `optimize_variational_*` now take
  `posterior_block_couplings` and the recipe exposes `--posterior-block`
  (age/metallicity/vz coupled per particle). Repeating the native 2x2 coverage
  test (`sigma=1e-4`, `beta_kl=1`, `prior_std=1.814`, 5 seeds, 20 trials) with the
  block posterior vs the diagonal baseline gave essentially unchanged marginal
  coverage: age `cov@0.9` `0.50` (block) vs `0.60` (diag), age `rms_z` `2.22` vs
  `1.93`; metallicity was marginally better (`rms_z` `1.59` vs `1.70`, SBC reduced
  chi-square `7.2` vs `8.9`) and age SBC uniformity improved (`3.3` vs `4.4`).
  The reason is structural: per-parameter *marginal* coverage depends only on the
  marginal *width*, not on the age-metallicity *correlation* the block adds, so a
  richer covariance cannot fix it. The residual under-coverage (`rms_z~2`, ~2x
  too narrow) is mean-field/Gaussian VI variance underestimation plus likely mean
  bias from the always-on SFH/CEH physics penalty (`param_penalty_weight=1.0`).
  The block family remains the right tool for *joint* credible regions (not
  measured by the current marginal harness). The correct next levers for marginal
  calibration are therefore: (1) importance-weighted / multi-sample VI to inflate
  the marginal width, (2) reduce or anneal the physics-penalty mean bias, and
  (3) add a joint (2D age-metallicity) coverage diagnostic to actually exercise
  the block posterior's strength.
- **Bias-vs-width decomposition: the gap is under-dispersion, not bias.**
  Decomposing the native 2x2 age z-scores (5 seeds, sigma=1e-4) gave a small
  systematic bias (`mean_z ~ +0.4`) but a large dispersion (`std_z ~ 1.9`), i.e.
  the posteriors are ~1.9x too narrow. The physics penalty was already off in
  these runs, so it is not the source. Widening the KL prior to `prior_std=5`
  made it *worse* (`std_z 1.9 -> 2.4`, recovery worse) by letting the fit overfit
  the injected noise, confirming the residual is not KL-prior pull. `prior_std=
  1.814` is retained.
- **Importance-weighted VI (IWAE) did not help here (negative result).** Added
  `importance_weighted` (IWAE bound) to the optimizer and `--importance-weighted`
  to the recipe. On the native rung (`K=16`) it slightly *narrowed* age
  (`post_std 1.01 -> 0.92`, `rms_z 1.93 -> 2.50`); metallicity/vz SBC improved
  marginally. This matches the known IWAE failure mode (Rainforth et al. 2018):
  the inference-network gradient signal-to-noise falls as `1/sqrt(K)`, so a large
  `K` with a stochastic forward model does not widen (can shrink) the proposal.
- **Joint 2D coverage diagnostic added; block correlation now sampled.**
  `joint_credible_coverage` (Mahalanobis highest-density regions) and a joint
  age-metallicity row in `run_vi_calibration.py`; the recipe now draws saved
  posterior samples from the FULL correlated posterior when a structured family
  is used. Joint age-Z coverage is (as expected) below the marginals
  (`cov@0.9 ~ 0.40`) because it compounds the under-dispersed marginals.
- **Root cause found: it was ``log_std`` under-convergence, not the posterior
  family.** The ~1.9x under-dispersion was robust across diagonal, block, and
  IWAE posteriors and across prior widths -- because none of those addressed the
  real issue. Re-running the native rung with more steps and a wider posterior
  init (``--vi-steps 800 --init-log-std -0.5`` vs the previous ``200`` steps from
  ``init_log_std=-2``) roughly *doubled* the recovered age posterior width
  (`post_std 1.0 -> 2.07`) and dropped `rms_z` from ~1.9 to ~1.44 on the same
  seed. Starting from ``exp(-2)~0.14`` the log-std simply had not grown to the
  data-supported width in 200 steps. Practical guidance for calibrated runs:
  use a wider ``init_log_std`` (~ -0.5) and enough VI steps for the width to
  converge; verify with the coverage/SBC harness. This reframes the earlier
  block/IWAE results: both are sound, generally-useful tools, but the native 2x2
  calibration gap was an optimization-convergence artifact, not a
  posterior-expressiveness limit.
- **5-seed aggregate confirms the fix (Phase 4 gate essentially closed for this
  rung).** At ``--vi-steps 800 --init-log-std -0.5`` (5 noise-injected seeds,
  ``sigma=1e-4``, ``beta_kl=1``, ``prior_std=1.814``, 20 pooled trials) the age
  dispersion becomes calibrated: ``std_z`` fell ``1.89 -> 1.06`` (nominal 1.0),
  age ``cov@0.9`` rose ``0.60 -> 0.85`` and ``cov@0.95`` ``0.80 -> 0.90``, and
  the joint age-metallicity ``cov@0.9`` rose ``0.45 -> 0.85``. The residual age
  ``rms_z=1.24`` is now a small systematic bias (``mean_z=+0.65``, ages slightly
  old), not under-dispersion. Metallicity moved from under- to slightly
  over-covering (``cov@0.9 0.55 -> 1.0``) and the weakly-identified velocity
  over-covers. These are now folded into a single ``--calibrated`` recipe preset
  (``beta_kl=1``, ``init_log_std=-0.5``, ``vi_steps=800`` on top of the summed
  likelihood and ``prior_std=1.814`` defaults; explicit flags still win). The
  recipe otherwise defaults to MAP recovery -- note ``init_log_std=-0.5`` mildly
  degrades MAP point recovery (``age_mae 0.36 -> 0.49``), which is why the wider
  init is applied only under ``--calibrated``.
- **Residual age mean-shift is legitimate prior influence, not a bug.** After
  convergence the only residual is a ~+0.6 sigma age mean-shift. It is *not* a
  reporting artifact (the posterior median gives the same shift; the mild +0.44
  posterior skew does not explain it) -- it is the correct Bayesian pull of the
  ``beta_kl=1`` prior on a finite-information likelihood, and coverage being
  near-nominal confirms the posterior (including that shift) is calibrated.
  Removing it would make the posterior *less* Bayesian-correct, so no de-biasing
  is applied. The metallicity/velocity over-coverage is within 20-trial sampling
  noise and is the conservative (safe) failure direction; genuine per-parameter
  *width* control would need per-parameter ``log_std`` learning rates, deferred
  as a non-minor feature.

Next work package:

1. Apply summed likelihood and L-BFGS/MAP warmup to the native 2x2 SFH+CEH rung
   and check whether velocity recovery improves without destabilizing
   age-metallicity recovery.
2. Repeat the local 2D profile comparison over multiple seeds with the CEH-only
   relation prior and, if needed, add a midpoint/wide local mode that spans both
   truth and fit in one scan.
3. Introduce a posterior family that can represent the observed coupled
   age-metallicity geometry, starting with block covariance per spaxel or a
   low-rank Gaussian before considering flows.
4. Introduce a flux- or S/N-scaled sigma model so `sigma_floor` is not manually
   tuned per cube brightness.
5. Extend the native ladder from one-particle-per-spaxel to two-particles-per
   spaxel using spaxel-mass-weighted recovery as the primary metric.
6. Defer a broad CEH weight sweep until after the local profile and structured
   prior diagnostics explain which part of the ridge the regularizer should
   constrain.
