Inference Workflows
===================

This page summarizes the production-ready inference interfaces added to Rubix
for gradient-based optimization and variational inference on IFU models.

Overview
--------

The inference stack lives in ``rubix.inference`` and provides:

- parameter application and forward/loss wrappers
- deterministic vs stochastic gradient pipeline mode selection
- constrained parameter transforms (age/metallicity defaults)
- Optax-based optimization loops
- finite-difference gradient validation helpers
- a first mean-field variational inference scaffold

Deterministic vs Stochastic Modes
---------------------------------

Use deterministic mode for stable gradient-based fitting and finite-difference
validation, and stochastic mode for likelihood/noise-aware workflows.
For ``calc_gradient``, both modes use the same transformer graph and the
post-aggregation noise switch is configured via
``pipeline_config.yml -> calc_gradient.options.post_aggregation_noise_by_mode``.

.. code-block:: python

   from rubix.inference import make_inference_pipeline

   pipe_det = make_inference_pipeline(config, mode="deterministic")
   pipe_sto = make_inference_pipeline(config, mode="stochastic")


Gradient-Based Optimization
---------------------------

The standard optimization entrypoint is ``optimize_params``.
For full-cube fitting with voxel masks/weights, use ``optimize_ifu_cube``.

.. code-block:: python

   import jax.numpy as jnp
   from rubix.inference import (
       build_age_metallicity_transforms,
       optimize_params,
   )

   transforms = build_age_metallicity_transforms(
       age_lower=0.0,
       age_upper=20.0,
       metallicity_lower=0.0,
       metallicity_upper=0.05,
   )

   result = optimize_params(
       pipeline=pipe_det,
       params_init=params_init,
       static_data=static_data,
       target=target_cube,
       transforms=transforms,
       learning_rate=1e-2,
       max_steps=500,
       tol=1e-6,
   )

   optimized = result.params

.. code-block:: python

   from rubix.inference import optimize_ifu_cube

   result = optimize_ifu_cube(
       pipeline=pipe_det,
       params_init=params_init,
       static_data=static_data,
       target=target_cube,
       mask=valid_voxel_mask,
       weights=inverse_variance_weights,
       normalize_loss=True,
       learning_rate=1e-2,
       max_steps=500,
   )


Science Loss Functions
----------------------

Rubix provides reusable loss builders for robust and probabilistic IFU fitting.

.. code-block:: python

   from rubix.inference import combine_loss_fns, huber_data_loss, masked_gaussian_nll

   def gaussian_term(pred, target):
       return masked_gaussian_nll(
           pred,
           target,
           inv_variance=inverse_variance_cube,
           mask=valid_voxel_mask,
           normalize=True,
       )

   def robust_term(pred, target):
       return huber_data_loss(pred, target, delta=0.2, mask=valid_voxel_mask)

   loss_fn = combine_loss_fns(
       [gaussian_term, robust_term],
       weights=[1.0, 0.1],
   )


Finite-Difference Gradient Validation
-------------------------------------

Use the validation helpers to compare autodiff gradients and central
finite-difference estimates.

.. code-block:: python

   from rubix.inference import compare_gradients, finite_difference_grad, loss, value_and_grad

   def objective(p):
       return loss(
           pipeline=pipe_det,
           params=p,
           static_data=static_data,
           target=target_cube,
       )

   _, auto_grad = value_and_grad(
       pipeline=pipe_det,
       params=params_init,
       static_data=static_data,
       target=target_cube,
   )
   fd_grad = finite_difference_grad(objective, params_init, eps=1e-4)
   summary = compare_gradients(auto_grad, fd_grad)

   print(summary.max_abs_error, summary.relative_l2_error)


Mean-Field Variational Inference
--------------------------------

Use ``optimize_variational_posterior`` for a first diagonal-Gaussian posterior.

.. code-block:: python

   from rubix.inference import optimize_variational_posterior

   vi = optimize_variational_posterior(
       pipeline=pipe_det,
       params_init=params_init,
       static_data=static_data,
       target=target_cube,
       learning_rate=5e-3,
       num_samples=4,
       beta_kl=1e-3,
       max_steps=500,
   )

   posterior_mean = vi.posterior_mean_params
   posterior_log_std = vi.posterior_log_std_params

For full IFU cubes, use ``optimize_variational_ifu_cube`` with uncertainty
maps and optional robust Huber regularization in the reconstruction term.
The returned ``VariationalResult`` includes diagnostics such as
``best_step``, ``final_objective``, ``final_reconstruction``, ``final_kl``,
and per-step ``grad_norm_history``/``update_norm_history``.

To benchmark full-cube VI performance:

.. code-block:: bash

   python bench/benchmark_variational_inference.py \
     --nx 25 --ny 25 --nw 256 \
     --repeats 3 \
     --max-steps 300 \
     --num-samples 4 \
     --use-mask --use-huber


Performance Notes
-----------------

For large particle counts, configure optional IFU accumulation controls:

- ``performance.particle_chunk_size``: chunked particle accumulation
- ``performance.remat_particlewise``: rematerialization/checkpointing of the
  particle step function for memory/computation tradeoffs

These settings are used by the particlewise IFU builders in ``rubix.core.ifu``.

Benchmarking Full-IFU Optimization
----------------------------------

Use the benchmark harness to profile optimization runtime and objective-side
memory diagnostics for full IFU cubes.

.. code-block:: bash

   python bench/benchmark_ifu_cube_optimization.py \
     --nx 25 --ny 25 --nw 256 \
     --repeats 3 \
     --max-steps 200 \
     --use-mask --use-weights
