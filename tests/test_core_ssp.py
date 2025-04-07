import jax.numpy as jnp
import pytest

from rubix import config
from rubix.core.data import reshape_array
from rubix.core.ssp import (
    get_lookup_interpolation,
    get_lookup_interpolation_pmap,
    get_lookup_interpolation_vmap,
    get_ssp,
)

ssp_config = config["ssp"]
supported_templates = ssp_config["templates"]
TEMPLATE_NAME = list(supported_templates.keys())[0]
print("supported_templates:", supported_templates)
print("TEMPLATE_NAME:", TEMPLATE_NAME)


RTOL = 1e-5
ATOL = 1e-6
# Sample configuration
sample_config = {
    "pipeline": {"name": "calc_ifu"},
    "logger": {
        "log_level": "DEBUG",
        "log_file_path": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "telescope": {"name": "MUSE"},
    "cosmology": {"name": "PLANCK15"},
    "galaxy": {"dist_z": 0.1},
    "ssp": {
        "template": {"name": "BruzualCharlot2003"},
    },
}


def _get_sample_inputs(subset=None):
    ssp = get_ssp(sample_config)
    """metallicity = reshape_array(ssp.metallicity)
    age = reshape_array(ssp.age)
    spectra = reshape_array(ssp.flux)"""
    metallicity = ssp.metallicity
    age = ssp.age
    spectra = ssp.flux

    print("Metallicity shape: ", metallicity.shape)
    print("Age shape: ", age.shape)
    print("Spectra shape: ", spectra.shape)
    print(".............")

    import numpy as np

    # Create meshgrid for metallicity and age to cover all combinations
    metallicity_grid, age_grid = np.meshgrid(
        metallicity.flatten(), age.flatten(), indexing="ij"
    )
    metallicity_grid = jnp.array(metallicity_grid)
    age_grid = jnp.array(age_grid)
    metallicity_grid = reshape_array(metallicity_grid.flatten())  # type: ignore
    age_grid = reshape_array(age_grid.flatten())  # type: ignore
    print("Metallicity grid shape: ", metallicity_grid.shape)
    print("Age grid shape: ", age_grid.shape)

    spectra = spectra.reshape(-1, spectra.shape[-1])
    print("spectra after reshape: ", spectra.shape)
    spectra = reshape_array(spectra)

    print("spectra after reshape_array call: ", spectra.shape)

    # reshape spectra
    num_combinations = metallicity_grid.shape[1]
    spectra_reshaped = spectra.reshape(
        spectra.shape[0], num_combinations, spectra.shape[-1]
    )

    # Create Velocities for each combination

    velocities = jnp.ones((metallicity_grid.shape[0], num_combinations, 3))
    mass = jnp.ones_like(metallicity_grid)

    if subset is not None:
        metallicity_grid = metallicity_grid[:, :subset]
        age_grid = age_grid[:, :subset]
        velocities = velocities[:, :subset]
        mass = mass[:, :subset]
        spectra_reshaped = spectra_reshaped[:, :subset]
    inputs = dict(
        metallicity=metallicity_grid, age=age_grid, velocities=velocities, mass=mass
    )
    return inputs, spectra_reshaped


def test_get_lookup_interpolation_with_valid_config():
    config = {
        "ssp": {
            "template": {"name": TEMPLATE_NAME},
            "method": "cubic",
        },
    }
    lookup = get_lookup_interpolation(config)
    assert callable(lookup)
    ssp = get_ssp(config)
    metallicity = ssp.metallicity[0]
    age = ssp.age[0]

    flux = ssp.flux[0, 0]

    spectrum = lookup(metallicity, age)

    assert not jnp.isnan(spectrum).all()
    assert jnp.allclose(spectrum, flux)

    # Check what happens out of bounds
    metallicity_oob = ssp.metallicity[-1] + 10
    spectrum_zero = lookup(metallicity_oob, age)  # this should return zero
    print("out of bounds spectrum:", spectrum_zero)
    assert not jnp.isnan(spectrum_zero).all()
    assert (spectrum_zero == 0).all()

    age_oob = ssp.age[-1] + 10
    spectrum_zero_age = lookup(metallicity, age_oob)  # this should return zero
    print("out of bounds age spectrum:", spectrum_zero_age)
    assert not jnp.isnan(spectrum_zero_age).all()
    assert (spectrum_zero_age == 0).all()


def test_get_lookup_interpolation_with_missing_ssp_field():
    config = {}
    with pytest.raises(ValueError) as excinfo:
        get_lookup_interpolation(config)
    assert "Configuration does not contain 'ssp' field" in str(excinfo.value)


def test_get_lookup_interpolation_with_missing_template_field():
    config = {"ssp": {}}
    with pytest.raises(ValueError) as excinfo:
        get_lookup_interpolation(config)
    assert "Configuration does not contain 'template' field" in str(excinfo.value)


def test_get_lookup_interpolation_with_missing_name_field():
    config = {"ssp": {"template": {}}}
    with pytest.raises(ValueError) as excinfo:
        get_lookup_interpolation(config)
    assert "Configuration does not contain 'name' field" in str(excinfo.value)


def test_get_lookup_interpolation_with_missing_method_field():
    config = {
        "ssp": {"template": {"name": TEMPLATE_NAME}},
    }
    lookup = get_lookup_interpolation(config)
    assert callable(lookup)


def test_get_lookup_interpolation_vmap():
    config = {
        "ssp": {
            "template": {"name": TEMPLATE_NAME},
            "method": "cubic",
        },
    }
    lookup_vmap = get_lookup_interpolation_vmap(config)
    assert callable(lookup_vmap)
    inputs, ssp_spectra = _get_sample_inputs()
    spectrum = lookup_vmap(inputs["metallicity"], inputs["age"])

    print("SSP_Spectra shape:", ssp_spectra.shape)
    print("Spectrum shape:", spectrum.shape)
    print("Spectrum example", spectrum[0, 0])
    assert jnp.allclose(spectrum, ssp_spectra, rtol=RTOL, atol=ATOL)

    # check out of bounds
    inputs["metallicity"] = inputs["metallicity"] + 1e7
    inputs["age"] = inputs["age"] + 1e7
    spectrum = lookup_vmap(inputs["metallicity"], inputs["age"])

    assert not jnp.isnan(spectrum).all()
    assert (spectrum == 0).all()


def test_get_lookup_interpolation_pmap():
    config = {
        "ssp": {
            "template": {"name": TEMPLATE_NAME},
            "method": "cubic",
        },
    }
    lookup_pmap = get_lookup_interpolation_pmap(config)
    assert callable(lookup_pmap)
    inputs, ssp_spectra = _get_sample_inputs()
    spectrum = lookup_pmap(inputs["metallicity"], inputs["age"])

    print("SSP_Spectra shape:", ssp_spectra.shape)
    print("Spectrum shape:", spectrum.shape)
    print("Spectrum example", spectrum[0, 0])
    assert jnp.allclose(spectrum, ssp_spectra, rtol=RTOL, atol=ATOL)

    # check out of bounds
    inputs["metallicity"] = inputs["metallicity"] + 1e7
    inputs["age"] = inputs["age"] + 1e7
    spectrum = lookup_pmap(inputs["metallicity"], inputs["age"])

    assert not jnp.isnan(spectrum).all()
    assert (spectrum == 0).all()
