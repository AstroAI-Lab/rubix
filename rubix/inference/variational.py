from dataclasses import dataclass
from typing import Mapping, Optional

import jax
import jax.numpy as jnp
import optax
from beartype.typing import Any

from rubix.core.data import RubixData

from .api import LossFn, loss
from .losses import combine_loss_fns, huber_data_loss, masked_gaussian_nll
from .parameterization import TransformTree, apply_transforms

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
    best_objective: float
    steps_run: int
    converged: bool


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
) -> jnp.ndarray:
    """Compute KL[q||p] for diagonal q vs standard normal prior p."""
    mean_flat, _ = jax.flatten_util.ravel_pytree(mean)
    log_std_flat, _ = jax.flatten_util.ravel_pytree(log_std)
    var_flat = jnp.exp(2.0 * log_std_flat)
    kl = 0.5 * jnp.sum(var_flat + mean_flat**2 - 1.0 - 2.0 * log_std_flat)
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
    beta_kl: float = 1e-3,
    init_log_std: float = -2.0,
    loss_fn: Optional[LossFn] = None,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    seed: int = 0,
) -> VariationalResult:
    """Optimize a mean-field variational posterior with reparameterization.

    Args:
        pipeline (Any): Pipeline-like object consumed by :func:`rubix.inference.loss`.
        params_init (ParamsTree): Initial constrained parameter point.
        static_data (RubixData): Baseline RubixData passed to the forward model.
        target (jnp.ndarray): Target datacube or statistic.
        learning_rate (float, optional): Step size for default Adam optimizer.
            Defaults to 5e-3.
        max_steps (int, optional): Maximum optimization steps. Defaults to 500.
        tol (float, optional): Convergence threshold on update norm.
            Defaults to 1e-6.
        num_samples (int, optional): Monte Carlo samples per step. Defaults to 4.
        beta_kl (float, optional): KL weight. Defaults to 1e-3.
        init_log_std (float, optional): Initial posterior log-std. Defaults to -2.0.
        loss_fn (Optional[LossFn], optional): Optional custom reconstruction loss.
            Defaults to ``None`` (sum-of-squares).
        noise_key (Optional[jnp.ndarray], optional): Optional key for stochastic
            pipelines. Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional transform tree
            to map unconstrained latent variables to constrained parameters.
            Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Custom
            optimizer. Defaults to ``None`` (Adam).
        seed (int, optional): Random seed for VI sampling. Defaults to 0.

    Raises:
        ValueError: If ``num_samples`` is not strictly positive.

    Returns:
        VariationalResult: Posterior statistics and optimization traces.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be strictly positive")

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

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
    opt_state = optimizer.init(variational_params)

    objective_history: list[float] = []
    reconstruction_history: list[float] = []
    kl_history: list[float] = []

    best_objective = jnp.inf
    best_mean = mean
    converged = False
    steps_run = 0
    key = jax.random.PRNGKey(seed)

    def objective_fn(current_params, step_key):
        current_mean = current_params["mean"]
        current_log_std = current_params["log_std"]
        sample_keys = jax.random.split(step_key, num_samples)

        def sample_reconstruction(sample_key):
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
        kl = kl_diag_gaussian_to_standard_normal(current_mean, current_log_std)
        objective = reconstruction + beta_kl * kl
        return objective, (reconstruction, kl)

    for step in range(max_steps):
        key, step_key = jax.random.split(key)
        (value, (reconstruction_value, kl_value)), grads = jax.value_and_grad(
            objective_fn, has_aux=True
        )(variational_params, step_key)

        current_mean = variational_params["mean"]
        if value < best_objective:
            best_objective = value
            best_mean = current_mean

        updates, opt_state = optimizer.update(grads, opt_state, variational_params)
        variational_params = optax.apply_updates(variational_params, updates)

        objective_history.append(float(value))
        reconstruction_history.append(float(reconstruction_value))
        kl_history.append(float(kl_value))

        steps_run = step + 1
        if float(optax.global_norm(updates)) < tol:
            converged = True
            break

    final_mean = variational_params["mean"]
    final_log_std = variational_params["log_std"]

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

    return VariationalResult(
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
        best_objective=float(best_objective),
        steps_run=steps_run,
        converged=converged,
    )


def optimize_variational_ifu_cube(
    pipeline: Any,
    params_init: ParamsTree,
    static_data: RubixData,
    target: jnp.ndarray,
    sigma: Optional[jnp.ndarray] = None,
    inv_variance: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    normalize_loss: bool = True,
    huber_delta: Optional[float] = None,
    huber_weight: float = 0.0,
    learning_rate: float = 5e-3,
    max_steps: int = 500,
    tol: float = 1e-6,
    num_samples: int = 4,
    beta_kl: float = 1e-3,
    init_log_std: float = -2.0,
    noise_key: Optional[jnp.ndarray] = None,
    transforms: Optional[TransformTree] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    seed: int = 0,
) -> VariationalResult:
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
        normalize_loss (bool, optional): Whether to normalize data term.
            Defaults to ``True``.
        huber_delta (Optional[float], optional): Optional Huber transition.
            Defaults to ``None`` (disabled unless ``huber_weight > 0``).
        huber_weight (float, optional): Weight of robust Huber data term.
            Defaults to 0.0.
        learning_rate (float, optional): Adam learning rate. Defaults to 5e-3.
        max_steps (int, optional): Maximum optimization steps. Defaults to 500.
        tol (float, optional): Convergence threshold. Defaults to 1e-6.
        num_samples (int, optional): Monte Carlo samples per step. Defaults to 4.
        beta_kl (float, optional): KL weight. Defaults to 1e-3.
        init_log_std (float, optional): Initial posterior log-std.
            Defaults to -2.0.
        noise_key (Optional[jnp.ndarray], optional): Optional stochastic key.
            Defaults to ``None``.
        transforms (Optional[TransformTree], optional): Optional transform tree.
            Defaults to ``None``.
        optimizer (Optional[optax.GradientTransformation], optional): Optional
            optimizer override. Defaults to ``None``.
        seed (int, optional): Random seed for VI sampling. Defaults to 0.

    Raises:
        ValueError: If ``target`` is not 3D, or if Huber settings are invalid.

    Returns:
        VariationalResult: Posterior statistics and optimization traces.
    """
    if target.ndim != 3:
        raise ValueError("target must be a 3D IFU datacube")

    if huber_weight < 0.0:
        raise ValueError("huber_weight must be non-negative")

    if huber_weight > 0.0 and huber_delta is None:
        raise ValueError("huber_delta must be provided when huber_weight > 0")

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
            delta=float(huber_delta),
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
    )
