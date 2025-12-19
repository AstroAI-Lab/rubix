from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.core.ifu import get_calculate_dusty_datacube_particlewise


class DummyExtinctionModel:
    def __init__(self, Rv):
        self.Rv = Rv

    def extinguish(self, wavelengths, av):
        del wavelengths, av
        return jnp.array([0.5, 0.5])


def _patch_dusty_dependencies(monkeypatch):
    logger = MagicMock()

    telescope = SimpleNamespace(
        sbin=2,
        wave_seq=jnp.array([4000.0, 5000.0]),
    )

    monkeypatch.setattr("rubix.core.ifu.get_logger", lambda cfg=None: logger)
    monkeypatch.setattr("rubix.core.ifu.get_telescope", lambda cfg: telescope)
    monkeypatch.setattr(
        "rubix.core.ifu.get_lookup_interpolation",
        lambda cfg: lambda Z, age: jnp.array([1.0, 1.0]),
    )
    monkeypatch.setattr(
        "rubix.core.ifu.get_ssp",
        lambda cfg: SimpleNamespace(wavelength=jnp.array([1.0, 2.0])),
    )
    monkeypatch.setattr(
        "rubix.core.ifu.cosmological_doppler_shift",
        lambda z, wavelength: wavelength,
    )
    monkeypatch.setattr(
        "rubix.core.ifu._velocity_doppler_shift_single",
        lambda wavelength, velocity, direction: wavelength,
    )

    def fake_resample(initial_spectrum, initial_wavelength, target_wavelength):
        del initial_wavelength
        return initial_spectrum[: target_wavelength.shape[0]]

    monkeypatch.setattr(
        "rubix.core.ifu.resample_spectrum",
        fake_resample,
    )
    monkeypatch.setattr(
        "rubix.core.ifu.Rv_model_dict",
        {"Dummy": DummyExtinctionModel},
    )
    monkeypatch.setattr("rubix.core.ifu.RV_MODELS", ["Dummy"])

    return logger, telescope


def _build_rubixdata() -> RubixData:
    stars = StarsData()
    stars.age = jnp.array([1.0, 2.0])
    stars.metallicity = jnp.array([0.1, 0.2])
    stars.mass = jnp.array([1.0, 2.0])
    stars.velocity = jnp.array([0.0, 0.0])
    stars.pixel_assignment = jnp.array([0, 1], dtype=jnp.int32)
    stars.extinction = jnp.ones((2, 2), dtype=jnp.float32)
    return RubixData(galaxy=Galaxy(), stars=stars, gas=GasData())


def test_calculate_dusty_datacube_particlewise(monkeypatch):
    logger, telescope = _patch_dusty_dependencies(monkeypatch)

    config = {
        "pipeline": {"name": "calc_ifu"},
        "logger": {"log_level": "DEBUG", "log_file_path": None, "format": ""},
        "telescope": {"name": "Dummy"},
        "cosmology": {"name": "PLANCK15"},
        "galaxy": {"dist_z": 0.1},
        "ssp": {
            "template": {"name": "BruzualCharlot2003"},
            "dust": {"extinction_model": "Dummy", "Rv": 3.1},
        },
    }

    rubixdata = _build_rubixdata()

    calculate = get_calculate_dusty_datacube_particlewise(config)
    result = calculate(rubixdata)

    datacube = result.stars.datacube
    assert datacube.shape == (2, 2, telescope.wave_seq.shape[0])

    flattened = datacube.reshape(-1, telescope.wave_seq.shape[0])
    np.testing.assert_allclose(flattened[0], [0.5, 0.5])
    np.testing.assert_allclose(flattened[1], [1.0, 1.0])
    assert np.all(flattened[2:] == 0)

    logger.info.assert_called()
