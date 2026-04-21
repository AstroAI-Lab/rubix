from collections.abc import Mapping
from typing import Any, Optional

import jax.numpy as jnp

from .api import LossFn
from .losses import combine_loss_fns, huber_data_loss, masked_gaussian_nll
from .objectives import build_ifu_cube_loss

TensorMap = Mapping[str, jnp.ndarray]


def _get_tensor(
    tensors: Optional[TensorMap],
    key: Optional[str],
    field_name: str,
) -> Optional[jnp.ndarray]:
    """Resolve optional tensor by key from a tensor map."""
    if key is None:
        return None

    if tensors is None:
        raise ValueError(f"{field_name} key provided but tensors mapping is None")

    if key not in tensors:
        raise ValueError(f"{field_name} key '{key}' not found in tensors mapping")

    return tensors[key]


def _build_single_loss_from_config(
    term_cfg: Mapping[str, Any],
    tensors: Optional[TensorMap],
) -> LossFn:
    """Build a single prediction-target loss from config."""
    kind = term_cfg.get("kind", "mse")

    if kind == "mse":
        mask = _get_tensor(tensors, term_cfg.get("mask_key"), "mask")
        weights = _get_tensor(tensors, term_cfg.get("weights_key"), "weights")
        normalize = bool(term_cfg.get("normalize", True))
        eps = float(term_cfg.get("eps", 1e-12))
        return build_ifu_cube_loss(
            mask=mask,
            weights=weights,
            normalize=normalize,
            eps=eps,
        )

    if kind == "gaussian_nll":
        sigma = _get_tensor(tensors, term_cfg.get("sigma_key"), "sigma")
        inv_variance = _get_tensor(
            tensors, term_cfg.get("inv_variance_key"), "inv_variance"
        )
        mask = _get_tensor(tensors, term_cfg.get("mask_key"), "mask")
        normalize = bool(term_cfg.get("normalize", True))
        eps = float(term_cfg.get("eps", 1e-12))

        def _loss_fn(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
            return masked_gaussian_nll(
                prediction=prediction,
                target=target,
                sigma=sigma,
                inv_variance=inv_variance,
                mask=mask,
                normalize=normalize,
                eps=eps,
            )

        return _loss_fn

    if kind == "huber":
        delta = float(term_cfg.get("delta", 1.0))
        mask = _get_tensor(tensors, term_cfg.get("mask_key"), "mask")
        weights = _get_tensor(tensors, term_cfg.get("weights_key"), "weights")
        normalize = bool(term_cfg.get("normalize", True))
        eps = float(term_cfg.get("eps", 1e-12))

        def _loss_fn(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
            return huber_data_loss(
                prediction=prediction,
                target=target,
                delta=delta,
                mask=mask,
                weights=weights,
                normalize=normalize,
                eps=eps,
            )

        return _loss_fn

    raise ValueError(
        "Unsupported objective kind "
        f"'{kind}'. Supported kinds: {'mse', 'gaussian_nll', 'huber'}"
    )


def build_loss_from_config(
    objective_config: Mapping[str, Any],
    tensors: Optional[TensorMap] = None,
) -> LossFn:
    """Build a composable loss function from objective configuration.

    The schema supports a single term or a weighted combination:

    - Single term:
      ``{"kind": "mse"}``
    - Weighted combination:
      ``{"kind": "combined", "terms": [...], "weights": [...]}``

    Args:
        objective_config (Mapping[str, Any]): Objective config dictionary.
        tensors (Optional[TensorMap], optional): Runtime tensor mapping used to
            resolve mask/weights/sigma keys. Defaults to ``None``.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        LossFn: Callable ``loss_fn(prediction, target)``.
    """
    if not isinstance(objective_config, Mapping):
        raise ValueError("objective_config must be a mapping")

    kind = objective_config.get("kind", "mse")
    if kind != "combined":
        return _build_single_loss_from_config(objective_config, tensors)

    terms = objective_config.get("terms")
    if not isinstance(terms, list) or len(terms) == 0:
        raise ValueError("combined objective requires non-empty 'terms' list")

    term_losses = [_build_single_loss_from_config(term, tensors) for term in terms]

    if "weights" in objective_config:
        weights_raw = objective_config["weights"]
        if not isinstance(weights_raw, list):
            raise ValueError("combined objective 'weights' must be a list")
        weights = [float(w) for w in weights_raw]
    else:
        weights = [float(term.get("weight", 1.0)) for term in terms]

    return combine_loss_fns(term_losses, weights=weights)


def build_loss_from_user_config(
    user_config: Mapping[str, Any],
    tensors: Optional[TensorMap] = None,
) -> Optional[LossFn]:
    """Build an objective loss from ``user_config['inference']['objective']``.

    Args:
        user_config (Mapping[str, Any]): Full runtime user configuration.
        tensors (Optional[TensorMap], optional): Runtime tensor mapping.
            Defaults to ``None``.

    Returns:
        Optional[LossFn]: Configured loss function, or ``None`` if the objective
            block is absent.
    """
    inference_cfg = user_config.get("inference")
    if not isinstance(inference_cfg, Mapping):
        return None

    objective_cfg = inference_cfg.get("objective")
    if objective_cfg is None:
        return None

    return build_loss_from_config(objective_cfg, tensors=tensors)
