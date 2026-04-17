# Gradient Production PR Plan

This document tracks the structured PR stack for scaling Rubix gradients from
proof-of-concept notebooks to production full-IFU workflows.

## Branching strategy

- Base branch: `multi-spaxel-gradients`
- Preferred pattern for isolated review: create short-lived sub-branches from
  `multi-spaxel-gradients` and open PRs back into `multi-spaxel-gradients`.
- Integration options:
  - squash/merge stacked PRs into `multi-spaxel-gradients`, then one PR to
    `main`
  - or promote selected sub-branches directly to `main` when fully independent

## PR stack

1. `feat(inference-api)` (completed)
- Add `rubix.inference` module with:
  - parameter application without mutating baseline `RubixData`
  - `forward`, `loss`, and `value_and_grad` API
- Add unit tests for copy semantics, deterministic loss calls, and analytical
  gradient checks.

2. `feat(parameterization)` (current)
- Add constrained parameter transforms (age/metallicity first) and tests.

3. `refactor(gradient-modes)`
- Separate deterministic and stochastic (PRNG-keyed) gradient paths.

4. `feat(optimizer-loop)`
- Add reusable Optax optimization loop with histories/checkpoints.

5. `test(gradient-vs-fd)`
- Add finite-difference validation suite for gradient correctness.

6. `perf(full-ifu-scaling)`
- Add chunking/checkpointing controls and consistency tests.

7. `feat(variational-inference)`
- Add first VI scaffold (mean-field baseline) on top of inference API.

8. `docs(examples-and-guides)`
- Add production examples and API docs for inverse modeling workflows.
