from unittest.mock import MagicMock, patch

import pytest

from rubix.inference import get_pipeline_name_for_mode
from rubix.inference.modes import make_inference_pipeline


def test_get_pipeline_name_for_mode_gradient_modes():
    deterministic = get_pipeline_name_for_mode("calc_gradient", "deterministic")
    stochastic = get_pipeline_name_for_mode("calc_gradient", "stochastic")

    assert deterministic == "calc_gradient"
    assert stochastic == "calc_gradient"


def test_get_pipeline_name_for_mode_passthrough_for_unknown_pipeline():
    assert get_pipeline_name_for_mode("calc_ifu", "deterministic") == "calc_ifu"
    assert get_pipeline_name_for_mode("calc_ifu", "stochastic") == "calc_ifu"


def test_get_pipeline_name_for_mode_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        get_pipeline_name_for_mode("calc_gradient", "bad")  # type: ignore[arg-type]


def test_make_inference_pipeline_sets_post_aggregation_noise_for_stochastic():
    dummy_config = {"pipeline": {"name": "calc_gradient"}}

    with patch("rubix.inference.modes.RubixPipeline") as mock_rubix_pipeline:
        mock_rubix_pipeline.return_value = MagicMock()
        make_inference_pipeline(dummy_config, "stochastic")
        _, kwargs = mock_rubix_pipeline.call_args
        assert kwargs.get("apply_noise_post_aggregation") is True


def test_make_inference_pipeline_does_not_set_post_aggregation_noise_for_deterministic():
    dummy_config = {"pipeline": {"name": "calc_gradient"}}

    with patch("rubix.inference.modes.RubixPipeline") as mock_rubix_pipeline:
        mock_rubix_pipeline.return_value = MagicMock()
        make_inference_pipeline(dummy_config, "deterministic")
        _, kwargs = mock_rubix_pipeline.call_args
        assert kwargs.get("apply_noise_post_aggregation") is False


def test_make_inference_pipeline_defaults_to_no_post_aggregation_noise(monkeypatch):
    monkeypatch.setattr(
        "rubix.inference.modes.get_pipeline_config",
        lambda _: {"Transformers": {}},
    )

    dummy_config = {"pipeline": {"name": "calc_ifu"}}

    with patch("rubix.inference.modes.RubixPipeline") as mock_rubix_pipeline:
        mock_rubix_pipeline.return_value = MagicMock()
        make_inference_pipeline(dummy_config, "stochastic")
        _, kwargs = mock_rubix_pipeline.call_args
        assert kwargs.get("apply_noise_post_aggregation") is False


def test_make_inference_pipeline_rejects_invalid_noise_policy_type(monkeypatch):
    monkeypatch.setattr(
        "rubix.inference.modes.get_pipeline_config",
        lambda _: {
            "options": {
                "post_aggregation_noise_by_mode": {
                    "deterministic": "false",
                    "stochastic": True,
                }
            }
        },
    )

    dummy_config = {"pipeline": {"name": "calc_gradient"}}
    with pytest.raises(
        ValueError,
        match="post_aggregation_noise_by_mode\\['deterministic'\\] must be a boolean",
    ):
        make_inference_pipeline(dummy_config, mode="deterministic")
