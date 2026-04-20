from copy import deepcopy
from typing import Literal

from rubix.core.pipeline import RubixPipeline

InferenceMode = Literal["deterministic", "stochastic"]

_PIPELINE_BY_MODE = {
    "calc_gradient": {
        "deterministic": "calc_gradient_deterministic",
        "stochastic": "calc_gradient_stochastic",
    }
}


def get_pipeline_name_for_mode(
    pipeline_name: str,
    mode: InferenceMode,
) -> str:
    """Resolve a concrete pipeline name for the requested inference mode.

    Args:
        pipeline_name (str): Base pipeline name from the user configuration.
        mode (InferenceMode): Requested inference mode.

    Raises:
        ValueError: If ``mode`` is invalid.

    Returns:
        str: Concrete pipeline name for the selected mode.
    """
    if mode not in {"deterministic", "stochastic"}:
        raise ValueError("mode must be one of {'deterministic', 'stochastic'}")

    mapping = _PIPELINE_BY_MODE.get(pipeline_name, {})
    return mapping.get(mode, pipeline_name)


def make_inference_pipeline(user_config: dict, mode: InferenceMode) -> RubixPipeline:
    """Build a RubixPipeline configured for deterministic or stochastic inference.

    Args:
        user_config (dict): Standard Rubix configuration dictionary.
        mode (InferenceMode): Inference mode selector.

    Returns:
        RubixPipeline: Pipeline instance using mode-specific pipeline graph.

    Note:
        In stochastic mode, ``apply_noise`` is intentionally excluded from the
        sharded pipeline graph.  Instead, noise is applied once to the fully
        aggregated datacube after the cross-device reduction, which avoids
        incorrect noise statistics caused by summing independently-noised shards.
    """
    config_copy = deepcopy(user_config)
    base_name = config_copy["pipeline"]["name"]
    config_copy["pipeline"]["name"] = get_pipeline_name_for_mode(base_name, mode)
    return RubixPipeline(
        config_copy, apply_noise_post_aggregation=(mode == "stochastic")
    )
