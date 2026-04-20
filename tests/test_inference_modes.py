import pytest

from rubix.inference import get_pipeline_name_for_mode, make_inference_pipeline


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


def test_make_inference_pipeline_sets_post_aggregation_noise_for_stochastic(
    monkeypatch,
):
    captured = {}

    class DummyPipeline:
        def __init__(self, cfg, apply_noise_post_aggregation=False):
            captured["cfg"] = cfg
            captured["apply_noise_post_aggregation"] = apply_noise_post_aggregation

    monkeypatch.setattr("rubix.inference.modes.RubixPipeline", DummyPipeline)

    cfg = {"pipeline": {"name": "calc_gradient"}}
    _ = make_inference_pipeline(cfg, mode="stochastic")
    assert captured["cfg"]["pipeline"]["name"] == "calc_gradient_stochastic"
    assert captured["apply_noise_post_aggregation"] is True

    _ = make_inference_pipeline(cfg, mode="deterministic")
    assert captured["cfg"]["pipeline"]["name"] == "calc_gradient_deterministic"
    assert captured["apply_noise_post_aggregation"] is False
