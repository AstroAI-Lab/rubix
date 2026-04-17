import pytest

from rubix.inference import get_pipeline_name_for_mode


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
