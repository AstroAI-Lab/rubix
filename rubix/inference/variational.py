import warnings
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import optax
from beartype.typing import Any

from rubix.core.data import RubixData

from .api import LossFn, loss
from .losses import combine_loss_fns, huber_data_loss, masked_gaussian_nll
from .parameterization import TransformTree, apply_transforms
from .posterior_family import (
    init_low_rank_factor,
    kl_low_rank_to_standard_normal,
    low_rank_marginal_log_std,
    sample_low_rank_gaussian,
)

ParamsTree = Mapping[str, Mapping[str, Any]]


@dataclass
class VariationalResult:
    """Container for mean-field variational optimization outputs."""

    posterior_mean_params: dict[str, dict[str, Any]]
    posterior_log_std_params: dict[str, dict[str, Any]]
    posterior_mean_constrained_params: dict[str, dict[str, Any]]
    best_posterior_mean_params: dict[str, dict[str, Any]]
    best_posterior_mean_constrained_params: dict[str, dict[str, Any]]
    objective_history: list[float]
    reconstruction_history: list[float]
    kl_history: list[float]
    grad_norm_history: list[float]
    update_norm_history: list[float]
    best_objective: float
    best_step: int
    final_objective: float
    final_reconstruction: float
    final_kl: float = float("nan")
    steps_run: int = -1
    converged: bool = False
    posterior_factor_params: Optional[jnp.ndarray] = None


@dataclass
class VariationalState:
    """Serializable variational optimization state for checkpoint/resume."""

    variational_params: dict[str, dict[str, dict[str, Any]]]
    opt_state: Any
    best_mean: dict[str, dict[str, Any]]
    best_objective: float
    best_step: int
    key: jnp.ndarray
    objective_history: list[float]
    reconstruction_history: list[float]
    kl_history: list[float]
    grad_norm_history: list[float]
    update_norm_history: list[float]
    steps_run: int
    best_selection_ema: float = float("inf")


def _tree_to_dict(tree: ParamsTree) -> dict[str, dict[str, Any]]:
    """Return a mutable dictionary copy from a nested parameter tree."""
    return {component: dict(fields) for component, fields in tree.items()}


def initialize_mean_field_params(
    params_init: ParamsTree,
    init_log_std: float = -2.0,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Initialize diagonal Gaussian variational parameters.

    Args:
        params_init (ParamsTree): Initial point for posterior means.
        init_log_std (float, optional): Initial log standard deviation for all
            leaves. Defaults to -2.0.

    Returns:
        tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
            Posterior mean and posterior log-std pytrees.
    """
    mean = _tree_to_dict(params_init)
    log_std = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) + init_log_std,  # noqa: B023
        mean,
    )
    return mean, _tree_to_dict(log_std)


def sample_diag_gaussian(
    mean: ParamsTree,
    log_std: ParamsTree,
    key: jnp.ndarray,
) -> dict[str, dict[str, Any]]:
    """Sample a pytree from a diagonal Gaussian posterior."""
    # Reconstruct a key tree matching ``mean`` leaves.
    treedef = jax.tree_util.tree_structure(mean)
    key_tree = jax.tree_util.tree_unflatten(
        treedef,
        list(jax.random.split(key, treedef.num_leaves)),
    )
    eps = jax.tree_util.tree_map(
        lambda x, k: jax.random.normal(k, shape=x.shape, dtype=x.dtype),
        mean,
        key_tree,
    )
    sample = jax.tree_util.tree_map(
        lambda m, ls, e: m + jnp.exp(ls) * e,  # noqa: B023
        mean,
        log_std,
        eps,
    )
    return _tree_to_dict(sample)


def kl_diag_gaussian_to_standard_normal(
    mean: ParamsTree,
    log_std: ParamsTree,
    prior_std: float = 1.0,
) -> jnp.ndarray:
    """Compute KL[q||p] for diagonal q vs a zero-mean Gaussian prior p.

    Args:
        mean (ParamsTree): Posterior mean pytree.
        log_std (ParamsTree): Posterior log-std pytree.
        prior_std (float, optional): Standard deviation of the isotropic
            zero-mean Gaussian prior ``N(0, prior_std^2 I)`` in unconstrained
            space. Defaults to 1.0 (standard normal). For sigmoid-bounded
            parameters ``prior_std ~= 1.814`` (``pi/sqrt(3)``) variance-matches
            the logistic prior that induces a uniform physical prior, which
            removes most of the midpoint bias of the standard normal.

    Returns:
        jnp.ndarray: Scalar KL divergence.
    """
    mean_flat, _ = jax.flatten_util.ravel_pytree(mean)
    log_std_flat, _ = jax.flatten_util.ravel_pytree(log_std)
    var_flat = jnp.exp(2.0 * log_std_flat)
    tau2 = prior_std**2
    log_tau = jnp.log(prior_std)
    kl = 0.5 * jnp.sum(
        (var_flat + mean_flat**2) / tau2 - 1.0 + 2.0 * log_tau - 2.0 * log_std_flat
    )
    return kl


def optimize_variational_posterior(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    learning_rate: float = 5e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    num_samples: int = 4,
    beta_kl: float = 1.0,
    init_log_std: float = -2.0,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    seed: int = 0,
    state_init: Optional[VariationalState] = None,
    return_state: bool = False,
    param_penalty_fn: Optional[Callable[[ParamsTree], jnp.ndarray]] = None,
    param_penalty_weight: float = 0.0,
    param_penalty_ramp_steps: int = 0,
    posterior_rank: int = 0,
    posterior_factor_init_scale: float = 1e-2,
    best_selection_ema_decay: float = 0.9,
    prior_std: float = 1.0,
) -> Any:
    """Optimize a mean-field variational posterior with reparameterization.

    Units contract (important for calibrated posteriors): the minimized
    objective is ``E_q[reconstruction] + beta_kl * KL(q || N(0, I))``. For the
    posterior *widths* to be calibrated this must equal the negative ELBO, which
    requires ``reconstruction`` to be the **summed** (not per-voxel-mean)
    negative log-likelihood and ``beta_kl = 1.0``. A per-voxel-mean data term
    (see ``normalize_loss`` in :func:`optimize_variational_ifu_cube`) shrinks the
    likelihood by the voxel count, so any ``beta_kl > 0`` then over-regularizes
    and the reported ``posterior_log_std_params`` are not trustworthy. Set
    ``beta_kl = 0`` to obtain an explicit MAP point estimate (the posterior mean
    is still meaningful, but its width is not).

    Args:
        pipeline (Any): Pipeline-like object consumed by :func:`rubix.inference.loss`.
        params_init (ParamsTree): Initial constrained parameter point.
            **Ignored when** ``state_init`` **is provided**; the optimizer
            resumes from ``state_init.variational_params`` instead.
        static_data (RubixData): Baseline RubixData passed to the forward model.
        target (jnp.ndarray): Target datacube or statistic.
        learning_rate (float, optional): Step size for default Adam optimizer.
            Defaults to 5e-3.
        max_steps (int, optional): Maximum optimization steps. Defaults to 500.
        tol (float, optional): Convergence threshold on the update norm.
            Defaults to 1e-6. **Caveat:** with a stochastic (Monte Carlo)
            objective the per-step update norm rarely falls below a tight
            ``tol``, so the returned ``converged`` flag is usually ``False``
            even for well-optimized runs; treat it as a best-effort signal and
            rely on the objective/gradient traces to judge convergence.
        num_samples (int, optional): Monte Carlo samples per step. Defaults to 4.
        beta_kl (float, optional): KL weight. Defaults to 1.0, the only value
            that yields a calibrated ELBO (with a summed-NLL reconstruction).
            Use 0.0 for an explicit MAP point estimate. See the units contract
            in the summary above.
        init_log_std (float, optional): Initial posterior log-std.  **Ignored
            when** ``state_init`` **is provided**; the posterior is resumed
            from the persisted variational parameters.  Defaults to -2.0.
        loss_fn (Optional[LossFn], optional): Optional custom reconstruction loss.
            Defaults to ``None`` (sum-of-squares).
        noise_key (Optional[jnp.ndarray], optional): Optional key for stochastic
            pipelines. Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional transform tree
            to map unconstrained latent variables to constrained parameters.
            Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Custom
            optimizer. Defaults to ``None`` (Adam).
        seed (int, optional): Random seed for VI sampling.  **Ignored when**
            ``state_init`` **is provided**; the PRNG key is resumed from
            ``state_init.key``.  Defaults to 0.
        state_init (Optional[VariationalState], optional): Optional resumable
            variational state.  When provided, ``params_init``, ``init_log_std``,
            and ``seed`` are all ignored and the run continues exactly from the
            persisted variational parameters, optimizer state, and PRNG key.
            Defaults to ``None``.
        return_state (bool, optional): If ``True``, also return updated
            :class:`VariationalState`. Defaults to ``False``.
        param_penalty_fn (Optional[Callable[[ParamsTree], jnp.ndarray]], optional):
            Optional penalty on constrained parameters (e.g. smoothness priors).
            Defaults to ``None``.
        param_penalty_weight (float, optional): Global multiplier for
            ``param_penalty_fn``. Defaults to 0.0.
        param_penalty_ramp_steps (int, optional): Number of initial steps over
            which to linearly ramp ``param_penalty_weight`` from 0 to full.
            Defaults to 0 (no ramp).
        posterior_rank (int, optional): If ``> 0``, use a low-rank-plus-diagonal
            Gaussian posterior with this factor rank instead of the diagonal
            mean-field posterior, allowing correlated parameter geometry (e.g.
            the age--metallicity ridge). ``posterior_log_std_params`` is then
            reported as the marginal log-std and ``posterior_factor_params``
            holds the low-rank factor. Defaults to 0 (mean-field).
        posterior_factor_init_scale (float, optional): Initialization scale for
            the low-rank factor. Small values start near the diagonal posterior.
            Defaults to 1e-2.
        best_selection_ema_decay (float, optional): Decay in ``[0, 1)`` for the
            exponential moving average of the (stochastic) objective used to
            select the ``best`` step/mean. Smoothing prevents a single
            lucky-noise Monte Carlo evaluation from being recorded as the best
            step. ``0`` disables smoothing and selects on the raw per-step value
            (``best_objective`` then equals ``min(objective_history)``). Defaults
            to 0.9.
        prior_std (float, optional): Std of the isotropic zero-mean Gaussian
            prior ``N(0, prior_std^2 I)`` on the unconstrained latents used in
            the KL term. Defaults to 1.0 (standard normal). For sigmoid-bounded
            parameters ``prior_std ~= 1.814`` approximates a uniform physical
            prior, removing most of the midpoint bias a standard normal imposes
            on calibrated (``beta_kl > 0``) runs.

    Raises:
        ValueError: If ``num_samples`` is not strictly positive, if
            ``posterior_rank`` is negative, or if ``best_selection_ema_decay``
            is not in ``[0, 1)``.

    Returns:
        VariationalResult: Posterior statistics and optimization traces.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be strictly positive")
    if posterior_rank < 0:
        raise ValueError("posterior_rank must be non-negative")
    if not 0.0 <= best_selection_ema_decay < 1.0:
        raise ValueError("best_selection_ema_decay must be in [0, 1)")

    use_low_rank = posterior_rank > 0

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    if state_init is None:
        if transforms is None:
            unconstrained_init = _tree_to_dict(params_init)
        else:
            unconstrained_init = apply_transforms(
                params=params_init,
                transforms=transforms,
                direction="inverse",
            )

        mean, log_std = initialize_mean_field_params(
            params_init=unconstrained_init,
            init_log_std=init_log_std,
        )
        variational_params = {"mean": mean, "log_std": log_std}
        if use_low_rank:
            factor_key = jax.random.PRNGKey(seed + 1)
            variational_params["factor"] = init_low_rank_factor(
                mean=mean,
                rank=posterior_rank,
                key=factor_key,
                init_scale=posterior_factor_init_scale,
            )
        opt_state = optimizer.init(variational_params)

        objective_history: list[float] = []
        reconstruction_history: list[float] = []
        kl_history: list[float] = []
        grad_norm_history: list[float] = []
        update_norm_history: list[float] = []

        best_objective = jnp.inf
        best_mean = mean
        best_step = -1
        best_ema = jnp.inf
        steps_run = 0
        key = jax.random.PRNGKey(seed)
    else:
        variational_params = state_init.variational_params
        opt_state = state_init.opt_state
        best_mean = state_init.best_mean
        best_objective = jnp.asarray(state_init.best_objective)
        best_step = int(state_init.best_step)
        key = state_init.key
        objective_history = list(state_init.objective_history)
        reconstruction_history = list(state_init.reconstruction_history)
        kl_history = list(state_init.kl_history)
        grad_norm_history = list(state_init.grad_norm_history)
        update_norm_history = list(state_init.update_norm_history)
        steps_run = int(state_init.steps_run)
        best_ema = jnp.asarray(
            getattr(state_init, "best_selection_ema", state_init.best_objective)
        )
        # Resume honors the persisted posterior family regardless of the
        # ``posterior_rank`` argument, which is ignored (like ``params_init``).
        use_low_rank = "factor" in variational_params

    converged = False
    initial_steps_run = steps_run

    if param_penalty_weight < 0:
        raise ValueError("param_penalty_weight must be non-negative")
    if param_penalty_ramp_steps < 0:
        raise ValueError("param_penalty_ramp_steps must be non-negative")

    def objective_fn(current_params, step_key, step_index):
        current_mean = current_params["mean"]
        current_log_std = current_params["log_std"]
        current_factor = current_params.get("factor") if use_low_rank else None
        sample_keys = jax.random.split(step_key, num_samples)

        def sample_reconstruction(sample_key):
            if use_low_rank:
                sampled_unconstrained = sample_low_rank_gaussian(
                    current_mean, current_log_std, current_factor, sample_key
                )
            else:
                sampled_unconstrained = sample_diag_gaussian(
                    current_mean, current_log_std, sample_key
                )
            if transforms is None:
                sampled_constrained = sampled_unconstrained
            else:
                sampled_constrained = apply_transforms(
                    params=sampled_unconstrained,
                    transforms=transforms,
                    direction="forward",
                )
            return loss(
                pipeline=pipeline,
                params=sampled_constrained,
                static_data=static_data,
                target=target,
                loss_fn=loss_fn,
                noise_key=noise_key,
            )

        reconstructions = jax.vmap(sample_reconstruction)(sample_keys)
        reconstruction = jnp.mean(reconstructions)
        if use_low_rank:
            kl = kl_low_rank_to_standard_normal(
                current_mean, current_log_std, current_factor, prior_std=prior_std
            )
        else:
            kl = kl_diag_gaussian_to_standard_normal(
                current_mean, current_log_std, prior_std=prior_std
            )
        prior_penalty = jnp.asarray(0.0, dtype=reconstruction.dtype)
        if param_penalty_fn is not None and param_penalty_weight > 0.0:
            if transforms is None:
                constrained_mean = current_mean
            else:
                constrained_mean = apply_transforms(
                    params=current_mean,
                    transforms=transforms,
                    direction="forward",
                )
            ramp = jnp.asarray(1.0, dtype=reconstruction.dtype)
            if param_penalty_ramp_steps > 0:
                ramp = jnp.minimum(
                    (step_index + 1.0) / float(param_penalty_ramp_steps),
                    1.0,
                )
            prior_penalty = (
                jnp.asarray(param_penalty_weight, dtype=reconstruction.dtype)
                * ramp
                * param_penalty_fn(constrained_mean)
            )
        objective = reconstruction + beta_kl * kl + prior_penalty
        return objective, (reconstruction, kl, prior_penalty)

    for step in range(max_steps):
        key, step_key = jax.random.split(key)
        step_index = jnp.asarray(initial_steps_run + step, dtype=jnp.float32)
        (value, (reconstruction_value, kl_value, _)), grads = jax.value_and_grad(
            objective_fn, has_aux=True
        )(variational_params, step_key, step_index)

        current_mean = variational_params["mean"]
        # Smooth the stochastic objective before selecting the best step so a
        # single low-variance Monte Carlo draw cannot masquerade as the best.
        if jnp.isinf(best_ema):
            step_ema = value
        else:
            step_ema = (
                best_selection_ema_decay * best_ema
                + (1.0 - best_selection_ema_decay) * value
            )
        best_ema = step_ema
        if step_ema < best_objective:
            best_objective = step_ema
            best_mean = current_mean
            best_step = initial_steps_run + step

        if isinstance(optimizer, optax.GradientTransformationExtraArgs):
            step_value_fn = lambda params: objective_fn(params, step_key, step_index)[0]
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                variational_params,
                value=value,
                grad=grads,
                value_fn=step_value_fn,
            )
        else:
            updates, opt_state = optimizer.update(grads, opt_state, variational_params)
        variational_params = optax.apply_updates(variational_params, updates)

        grad_norm = float(optax.global_norm(grads))
        update_norm = float(optax.global_norm(updates))

        objective_history.append(float(value))
        reconstruction_history.append(float(reconstruction_value))
        kl_history.append(float(kl_value))
        grad_norm_history.append(grad_norm)
        update_norm_history.append(update_norm)

        steps_run = steps_run + 1
        if update_norm < tol:
            converged = True
            break

    final_mean = variational_params["mean"]
    final_diag_log_std = variational_params["log_std"]
    if use_low_rank:
        final_factor = variational_params["factor"]
        # Report marginal widths so diagonal downstream samplers reproduce the
        # correct per-parameter posterior spread; the factor is returned for
        # joint (correlated) sampling.
        final_log_std = low_rank_marginal_log_std(final_diag_log_std, final_factor)
    else:
        final_factor = None
        final_log_std = final_diag_log_std

    if transforms is None:
        posterior_mean_constrained = final_mean
        best_posterior_mean_constrained = best_mean
    else:
        posterior_mean_constrained = apply_transforms(
            params=final_mean,
            transforms=transforms,
            direction="forward",
        )
        best_posterior_mean_constrained = apply_transforms(
            params=best_mean,
            transforms=transforms,
            direction="forward",
        )

    if len(objective_history) == 0:
        final_objective = float("nan")
        final_reconstruction = float("nan")
        final_kl = float("nan")
        state_key = key
    else:
        # Capture the training-loop key state before the final evaluation so
        # that the VariationalState records the PRNG state at which the loop
        # ended.  This ensures resumed runs continue with the same key
        # sequence as a continuous run of the same total length would.
        state_key = key
        key, final_eval_key = jax.random.split(key)
        final_step_index = jnp.asarray(max(steps_run - 1, 0), dtype=jnp.float32)
        final_value, (final_reconstruction_value, final_kl_value, _) = objective_fn(
            variational_params, final_eval_key, final_step_index
        )
        final_objective = float(final_value)
        final_reconstruction = float(final_reconstruction_value)
        final_kl = float(final_kl_value)

    result = VariationalResult(
        posterior_mean_params=_tree_to_dict(final_mean),
        posterior_log_std_params=_tree_to_dict(final_log_std),
        posterior_mean_constrained_params=_tree_to_dict(posterior_mean_constrained),
        best_posterior_mean_params=_tree_to_dict(best_mean),
        best_posterior_mean_constrained_params=_tree_to_dict(
            best_posterior_mean_constrained
        ),
        objective_history=objective_history,
        reconstruction_history=reconstruction_history,
        kl_history=kl_history,
        grad_norm_history=grad_norm_history,
        update_norm_history=update_norm_history,
        best_objective=float(best_objective),
        best_step=best_step,
        final_objective=final_objective,
        final_reconstruction=final_reconstruction,
        final_kl=final_kl,
        steps_run=steps_run,
        converged=converged,
        posterior_factor_params=final_factor,
    )

    serialized_variational_params = {
        "mean": _tree_to_dict(variational_params["mean"]),
        "log_std": _tree_to_dict(variational_params["log_std"]),
    }
    if "factor" in variational_params:
        serialized_variational_params["factor"] = variational_params["factor"]

    state = VariationalState(
        variational_params=serialized_variational_params,
        opt_state=opt_state,
        best_mean=_tree_to_dict(best_mean),
        best_objective=float(best_objective),
        best_step=best_step,
        key=state_key,
        objective_history=objective_history,
        reconstruction_history=reconstruction_history,
        kl_history=kl_history,
        grad_norm_history=grad_norm_history,
        update_norm_history=update_norm_history,
        steps_run=steps_run,
        best_selection_ema=float(best_ema),
    )

    if return_state:
        return result, state
    return result


def optimize_variational_ifu_cube(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    sigma: Optional[jnp.ndarray] = None,
    inv_variance: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    normalize_loss: bool = False,
    huber_delta: Optional[float] = None,
    huber_weight: float = 0.0,
    learning_rate: float = 5e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    num_samples: int = 4,
    beta_kl: float = 1.0,
    init_log_std: float = -2.0,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    seed: int = 0,
    state_init: Optional[VariationalState] = None,
    return_state: bool = False,
    param_penalty_fn: Optional[Callable[[ParamsTree], jnp.ndarray]] = None,
    param_penalty_weight: float = 0.0,
    param_penalty_ramp_steps: int = 0,
    posterior_rank: int = 0,
    posterior_factor_init_scale: float = 1e-2,
    best_selection_ema_decay: float = 0.9,
    prior_std: float = 1.0,
) -> Any:
    """Optimize a VI posterior against full IFU cubes with science losses.

    Args:
        pipeline (Any): Pipeline-like object consumed by VI objective.
        params_init (ParamsTree): Initial constrained parameter point.
        static_data (RubixData): Baseline RubixData passed to the forward model.
        target (jnp.ndarray): Target IFU datacube.
        sigma (Optional[jnp.ndarray], optional): Per-voxel uncertainty cube.
            Defaults to ``None``.
        inv_variance (Optional[jnp.ndarray], optional): Per-voxel inverse
            variance cube. Defaults to ``None``.
        mask (Optional[jnp.ndarray], optional): Optional binary voxel mask.
            Defaults to ``None``.
        normalize_loss (bool, optional): If ``True`` the Gaussian data term is
            divided by the number of active voxels (per-voxel mean NLL). This
            breaks the ELBO units, so it should only be combined with
            ``beta_kl = 0`` (MAP). Defaults to ``False`` (summed NLL), which is
            the calibrated choice paired with ``beta_kl = 1.0``.
        huber_delta (Optional[float], optional): Optional Huber transition.
            Defaults to ``None`` (disabled unless ``huber_weight > 0``).
        huber_weight (float, optional): Weight of robust Huber data term.
            Defaults to 0.0. **Caveat:** a nonzero Huber term adds a
            non-Gaussian, non-log-likelihood component to the objective, so the
            ELBO is no longer a valid negative log-evidence and posterior widths
            are not calibrated while ``huber_weight > 0``. Keep it at 0 for
            calibration runs; use it only as a robustness aid for point fits.
        learning_rate (float, optional): Adam learning rate. Defaults to 5e-3.
        max_steps (int, optional): Maximum optimization steps. Defaults to 500.
        tol (float, optional): Convergence threshold. Defaults to 1e-6.
        num_samples (int, optional): Monte Carlo samples per step. Defaults to 4.
        beta_kl (float, optional): KL weight. Defaults to 1.0 (calibrated ELBO
            with the summed-NLL default). Use 0.0 for an explicit MAP estimate.
        init_log_std (float, optional): Initial posterior log-std.
            Defaults to -2.0.
        noise_key (Optional[jnp.ndarray], optional): Optional stochastic key.
            Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional transform tree.
            Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Optional
            optimizer override. Defaults to ``None``.
        seed (int, optional): Random seed for VI sampling. Defaults to 0.
        state_init (Optional[VariationalState], optional): Optional resumable
            state for exact continuation. Defaults to ``None``.
        return_state (bool, optional): If ``True``, also return updated
            :class:`VariationalState`. Defaults to ``False``.
        param_penalty_fn (Optional[Callable[[ParamsTree], jnp.ndarray]], optional):
            Optional penalty on constrained parameters. Defaults to ``None``.
        param_penalty_weight (float, optional): Global multiplier for
            ``param_penalty_fn``. Defaults to 0.0.
        param_penalty_ramp_steps (int, optional): Number of steps to ramp
            penalty weight. Defaults to 0.
        posterior_rank (int, optional): If ``> 0``, use a low-rank-plus-diagonal
            Gaussian posterior of this factor rank instead of mean-field, to
            capture correlated geometry such as the age--metallicity ridge.
            Defaults to 0 (mean-field).
        posterior_factor_init_scale (float, optional): Initialization scale for
            the low-rank factor. Defaults to 1e-2.
        best_selection_ema_decay (float, optional): EMA decay in ``[0, 1)`` for
            smoothing the stochastic objective when selecting the best step.
            Defaults to 0.9; use 0 to select on the raw per-step value.
        prior_std (float, optional): Std of the isotropic zero-mean Gaussian
            prior on unconstrained latents in the KL term. Defaults to 1.0; use
            ``~1.814`` to approximate a uniform physical prior for sigmoid-bounded
            parameters on calibrated runs.

    Raises:
        ValueError: If ``target`` is not 3D, if Huber settings are invalid, if
            both ``sigma`` and ``inv_variance`` are provided, or if the shape of
            ``sigma``, ``inv_variance``, or ``mask`` does not match ``target``.

    Returns:
        VariationalResult: Posterior statistics and optimization traces.
    """
    if target.ndim != 3:
        raise ValueError("target must be a 3D IFU datacube")

    if normalize_loss and beta_kl > 0.0:
        warnings.warn(
            "optimize_variational_ifu_cube called with normalize_loss=True and "
            "beta_kl > 0: the per-voxel-mean data term and the KL term are on "
            "inconsistent scales, so the resulting posterior widths are NOT "
            "calibrated (the KL over-regularizes by roughly the active-voxel "
            "count). For a calibrated ELBO use normalize_loss=False (summed "
            "Gaussian NLL) with beta_kl=1.0; use beta_kl=0.0 for an explicit "
            "MAP point estimate.",
            stacklevel=2,
        )

    if sigma is not None and inv_variance is not None:
        raise ValueError("only one of sigma or inv_variance may be provided, not both")

    if sigma is not None and sigma.shape != target.shape:
        raise ValueError(
            f"sigma shape {sigma.shape} does not match target shape {target.shape}"
        )

    if inv_variance is not None and inv_variance.shape != target.shape:
        raise ValueError(
            f"inv_variance shape {inv_variance.shape} does not match target shape "
            f"{target.shape}"
        )

    if mask is not None and mask.shape != target.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match target shape {target.shape}"
        )

    huber_weight = float(huber_weight)

    if huber_weight < 0.0:
        raise ValueError("huber_weight must be non-negative")

    if huber_weight > 0.0 and huber_delta is None:
        raise ValueError("huber_delta must be provided when huber_weight > 0")

    if huber_weight > 0.0:
        huber_delta = float(huber_delta)

    if huber_weight > 0.0 and huber_delta <= 0.0:
        raise ValueError("huber_delta must be > 0 when huber_weight > 0")

    gaussian_loss: LossFn = lambda pred, truth: masked_gaussian_nll(
        prediction=pred,
        target=truth,
        sigma=sigma,
        inv_variance=inv_variance,
        mask=mask,
        normalize=normalize_loss,
    )

    if huber_weight > 0.0:
        huber_loss: LossFn = lambda pred, truth: huber_data_loss(
            prediction=pred,
            target=truth,
            delta=huber_delta,
            mask=mask,
            normalize=normalize_loss,
        )
        reconstruction_loss_fn = combine_loss_fns(
            [gaussian_loss, huber_loss], weights=[1.0, huber_weight]
        )
    else:
        reconstruction_loss_fn = gaussian_loss

    return optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tol=tol,
        num_samples=num_samples,
        beta_kl=beta_kl,
        init_log_std=init_log_std,
        loss_fn=reconstruction_loss_fn,
        noise_key=noise_key,
        transforms=transforms,
        optimizer=optimizer,
        seed=seed,
        state_init=state_init,
        return_state=return_state,
        param_penalty_fn=param_penalty_fn,
        param_penalty_weight=param_penalty_weight,
        param_penalty_ramp_steps=param_penalty_ramp_steps,
        posterior_rank=posterior_rank,
        posterior_factor_init_scale=posterior_factor_init_scale,
        best_selection_ema_decay=best_selection_ema_decay,
        prior_std=prior_std,
    )
