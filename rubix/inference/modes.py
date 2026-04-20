from copy import deepcopy
from typing import Literal

from rubix.core.pipeline import RubixPipeline
from rubix.utils import get_pipeline_config

InferenceMode = Literal["deterministic", "stochastic"]

_PIPELINE_BY_MODE = {
    "calc_gradient": {
        "deterministic": "calc_gradient",
        "stochastic": "calc_gradient",
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


def get_post_aggregation_noise_for_mode(
    pipeline_name: str,
    mode: InferenceMode,
) -> bool:
    """Resolve post-aggregation noise policy for a pipeline and mode.

    Policy is read from pipeline config option
    ``options.post_aggregation_noise_by_mode`` when present.

    Args:
        pipeline_name (str): Selected pipeline name from the runtime config.
        mode (InferenceMode): Inference mode selector.

    Raises:
        ValueError: If the mode policy is malformed in pipeline config.

    Returns:
        bool: ``True`` if post-aggregation noise should be applied.
    """
    pipeline_config = get_pipeline_config(pipeline_name)
    options = pipeline_config.get("options", {})
    by_mode = options.get("post_aggregation_noise_by_mode", {})

    if not isinstance(by_mode, dict):
        raise ValueError("post_aggregation_noise_by_mode must be a mapping")

    value = by_mode.get(mode, False)
    if not isinstance(value, bool):
        raise ValueError(f"post_aggregation_noise_by_mode[{mode!r}] must be a boolean")

    return value


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
    selected_name = get_pipeline_name_for_mode(base_name, mode)
    config_copy["pipeline"]["name"] = selected_name
    return RubixPipeline(
        config_copy,
        apply_noise_post_aggregation=get_post_aggregation_noise_for_mode(
            selected_name, mode
        ),
    )
