import pytest

from rubix.core.psf import get_convolve_psf


def test_missing_psf_configuration():
    """Test if ValueError is raised when 'psf' key is missing."""
    config = {"telescope": {}}
    with pytest.raises(ValueError) as e:
        get_convolve_psf(config)
    assert str(e.value) == "PSF configuration not found in telescope configuration"


def test_missing_psf_name():
    """Test if ValueError is raised when 'name' key is missing."""
    config = {"telescope": {"psf": {}}}
    with pytest.raises(ValueError) as e:
        get_convolve_psf(config)
    assert str(e.value) == "PSF name not found in telescope configuration"


def test_missing_psf_size():
    """Test if ValueError is raised when 'size' key is missing."""
    config = {"telescope": {"psf": {"name": "gaussian"}}}
    with pytest.raises(ValueError) as e:
        get_convolve_psf(config)
    assert str(e.value) == "PSF size not found in telescope configuration"


def test_missing_psf_sigma():
    """Test if ValueError is raised when 'sigma' key is missing."""
    config = {"telescope": {"psf": {"name": "gaussian", "size": 5}}}
    with pytest.raises(ValueError) as e:
        get_convolve_psf(config)
    assert str(e.value) == "PSF sigma not found in telescope configuration"


def test_unknown_psf_name():
    """Test if ValueError is raised for an unknown PSF name."""
    config = {"telescope": {"psf": {"name": "unknown", "size": 5, "sigma": 1.5}}}
    with pytest.raises(ValueError) as e:
        get_convolve_psf(config)
    assert str(e.value) == "Unknown PSF kernel name: unknown"
