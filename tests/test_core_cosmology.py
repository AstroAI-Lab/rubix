import pytest

from rubix.core.cosmology import get_cosmology
from rubix.cosmology import PLANCK15, RubixCosmology


def test_get_cosmology_planck15():
    config = {"cosmology": {"name": "PLANCK15"}}
    cosmology = get_cosmology(config)
    assert cosmology is PLANCK15


def test_get_cosmology_custom():
    config = {
        "cosmology": {
            "name": "CUSTOM",
            "args": {"Om0": 1.0, "w0": 2.0, "wa": 3.0, "h": 4.0},
        }
    }
    cosmology = get_cosmology(config)
    assert isinstance(cosmology, RubixCosmology)
    assert cosmology.Om0 == 1
    assert cosmology.w0 == 2
    assert cosmology.wa == 3
    assert cosmology.h == 4


def test_get_cosmology_invalid():
    config = {"cosmology": {"name": "INVALID"}}
    with pytest.raises(ValueError):
        get_cosmology(config)
