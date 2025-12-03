import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from rubix.core.fits import load_fits, store_fits


def _make_config(rotation_type: str = "face-on") -> dict:
    rotation = {
        "type": rotation_type,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
    }
    if rotation_type not in ("face-on", "edge-on"):
        rotation.update(alpha=0.11, beta=-0.22, gamma=0.33)

    return {
        "pipeline": {"name": "test_pipeline"},
        "simulation": {"name": "TestSim"},
        "galaxy": {"dist_z": 0.2, "rotation": rotation},
        "data": {
            "subset": {"use_subset": True},
            "load_galaxy_args": {"id": 42},
            "args": {"snapshot": 7},
        },
        "ssp": {"template": {"name": "Template"}},
        "telescope": {
            "name": "DummyScope",
            "psf": {"name": "gaussian", "size": 3, "sigma": 0.6},
            "lsf": {"sigma": 0.8},
            "noise": {"signal_to_noise": 5, "noise_distribution": "gaussian"},
        },
        "cosmology": {"name": "TEST"},
    }


def _patch_logger_and_telescope(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr("rubix.core.fits.get_logger", lambda cfg=None: logger)
    telescope = SimpleNamespace(
        spatial_res=0.5,
        wave_res=1.5,
        wave_range=(3600, 7000),
    )
    monkeypatch.setattr("rubix.core.fits.get_telescope", lambda cfg: telescope)
    return logger, telescope


def _expected_filename(filepath: str, config: dict) -> str:
    base_filename = (
        f"{config['simulation']['name']}"
        f"_id{config['data']['load_galaxy_args']['id']}"
        f"_snap{config['data']['args']['snapshot']}"
        f"_{config['telescope']['name']}"
        f"_{config['pipeline']['name']}.fits"
    )
    return f"{filepath}{base_filename}"


def test_store_fits_face_on_rotation(tmp_path, monkeypatch):
    logger, telescope = _patch_logger_and_telescope(monkeypatch)
    config = _make_config(rotation_type="face-on")
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    filepath = os.path.join(str(tmp_path), "fits_output", "")

    store_fits(config, data, filepath)

    expected_file = _expected_filename(filepath, config)
    assert os.path.exists(expected_file)

    with fits.open(expected_file) as hdul:
        primary = hdul[0].header
        assert primary["PIPELINE"] == config["pipeline"]["name"]
        assert primary["DIST_z"] == config["galaxy"]["dist_z"]
        assert primary["ROTATION"] == config["galaxy"]["rotation"]["type"]
        assert primary["SIM"] == config["simulation"]["name"]
        assert primary["GALAXYID"] == config["data"]["load_galaxy_args"]["id"]
        assert primary["SNAPSHOT"] == config["data"]["args"]["snapshot"]
        assert primary["SUBSET"] == config["data"]["subset"]["use_subset"]
        assert primary["SSP"] == config["ssp"]["template"]["name"]
        assert primary["INSTR"] == config["telescope"]["name"]
        assert primary["PSF"] == config["telescope"]["psf"]["name"]
        assert primary["PSF_SIZE"] == config["telescope"]["psf"]["size"]
        assert primary["PSFSIGMA"] == config["telescope"]["psf"]["sigma"]
        assert primary["LSF"] == config["telescope"]["lsf"]["sigma"]
        assert primary["S_TO_N"] == config["telescope"]["noise"]["signal_to_noise"]
        assert primary["N_DISTR"] == config["telescope"]["noise"]["noise_distribution"]
        assert primary["COSMO"] == config["cosmology"]["name"]

        data_hdu = hdul[1].data
        np.testing.assert_array_equal(data_hdu, data.T)

    logger.info.assert_called_once_with(f"Datacube saved to {expected_file}")


def test_store_fits_custom_rotation_exposes_angles(tmp_path, monkeypatch):
    logger, telescope = _patch_logger_and_telescope(monkeypatch)
    config = _make_config(rotation_type="custom")
    data = np.zeros((1, 1, 1), dtype=np.float32)
    filepath = os.path.join(str(tmp_path), "fits_output", "")

    store_fits(config, data, filepath)
    expected_file = _expected_filename(filepath, config)
    assert os.path.exists(expected_file)

    with fits.open(expected_file) as hdul:
        primary = hdul[0].header
        assert "ROTATION" not in primary
        assert primary["ROT_A"] == pytest.approx(0.11)
        assert primary["ROT_B"] == pytest.approx(-0.22)
        assert primary["ROT_C"] == pytest.approx(0.33)

    logger.info.assert_called_with(f"Datacube saved to {expected_file}")


def test_load_fits_returns_cube_instance(monkeypatch):
    cube_instance = MagicMock()
    cube_factory = MagicMock(return_value=cube_instance)
    monkeypatch.setattr("rubix.core.fits.Cube", cube_factory)

    path = "/tmp/dummy.fits"
    result = load_fits(path)

    cube_factory.assert_called_once_with(filename=path)
    assert result is cube_instance
