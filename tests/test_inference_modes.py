import pytest
from unittest.mock import patch, MagicMock

from rubix.inference import get_pipeline_name_for_mode
from rubix.inference.modes import make_inference_pipeline


def test_get_pipeline_name_for_mode_gradient_modes():
    deterministic = get_pipeline_name_for_mode("calc_gradient", "deterministic")
    stochastic = get_pipeline_name_for_mode("calc_gradient", "stochastic")

    assert deterministic == "calc_gradient_deterministic"
    assert stochastic == "calc_gradient_stochastic"


def test_get_pipeline_name_for_mode_passthrough_for_unknown_pipeline():
    assert get_pipeline_name_for_mode("calc_ifu", "deterministic") == "calc_ifu"
    assert get_pipeline_name_for_mode("calc_ifu", "stochastic") == "calc_ifu"


def test_get_pipeline_name_for_mode_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        get_pipeline_name_for_mode("calc_gradient", "bad")  # type: ignore[arg-type]


def test_make_inference_pipeline_sets_post_aggregation_noise_for_stochastic():
    dummy_config = {"pipeline": {"name": "calc_gradient"}}
    mock_pipeline = MagicMock()
    mock_pipeline._apply_noise_post_aggregation = False

    with patch("rubix.inference.modes.RubixPipeline") as MockRubixPipeline:
        MockRubixPipeline.return_value = mock_pipeline
        make_inference_pipeline(dummy_config, "stochastic")
        _, kwargs = MockRubixPipeline.call_args
        assert kwargs.get("apply_noise_post_aggregation") is True


def test_make_inference_pipeline_does_not_set_post_aggregation_noise_for_deterministic():
    dummy_config = {"pipeline": {"name": "calc_gradient"}}
    mock_pipeline = MagicMock()
    mock_pipeline._apply_noise_post_aggregation = False

    with patch("rubix.inference.modes.RubixPipeline") as MockRubixPipeline:
        MockRubixPipeline.return_value = mock_pipeline
        make_inference_pipeline(dummy_config, "deterministic")
        _, kwargs = MockRubixPipeline.call_args
        assert kwargs.get("apply_noise_post_aggregation") is False
