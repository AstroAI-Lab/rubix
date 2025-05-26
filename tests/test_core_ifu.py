import jax
import jax.numpy as jnp
import numpy as np

from rubix.core.data import Galaxy, GasData, RubixData, StarsData, reshape_array
from rubix.core.ifu import (
    get_calculate_spectra,
    get_doppler_shift_and_resampling,
    get_resample_spectrum_pmap,
    get_resample_spectrum_vmap,
    get_scale_spectrum_by_mass,
)
from rubix.core.ssp import get_ssp
from rubix.spectra.ifu import resample_spectrum

RTOL = 1e-4
ATOL = 1e-6
# Sample input data
sample_inputs = {
    "metallicity": jnp.array([0.1, 0.2]),
    "age": jnp.array([1.0, 2.0]),
    "mass": jnp.array([0.5, 1.0]),
    "velocities": jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    "spectra": jnp.zeros((2, 842)),
}


# reshape sample inputs

print("Sample_inputs:")
for key in sample_inputs:
    sample_inputs[key] = reshape_array(sample_inputs[key])
    print(f"Key: {key}, shape: {sample_inputs[key].shape}")


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


class MockRubixData:
    def __init__(self, stars, gas):
        self.stars = stars
        self.gas = gas


class MockStarsData:
    def __init__(self, velocity, metallicity, mass, age, spectra=None):
        # self.coords = coords
        self.velocity = velocity
        self.metallicity = metallicity
        self.mass = mass
        self.age = age
        # self.pixel_assignment = pixel_assignment
        self.spectra = spectra


class MockGasData:
    def __init__(self, spectra):
        self.spectra = None


# Sample inputs for testing
initial_spectra = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
initial_wavelengths = jnp.array([[4500.0, 5500.0, 6500.0], [4500.0, 5500.0, 6500.0]])
target_wavelength = jnp.array([4000.0, 5000.0, 6000.0])


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

    # Create meshgrid for metallicity and age to cover all combinations
    metallicity_grid, age_grid = np.meshgrid(
        metallicity.flatten(), age.flatten(), indexing="ij"
    )
    metallicity_grid = jnp.asarray(metallicity_grid.flatten())  # Convert to jax.Array
    age_grid = jnp.asarray(age_grid.flatten())  # Convert to jax.Array
    metallicity_grid = reshape_array(metallicity_grid)
    age_grid = reshape_array(age_grid)
    metallicity_grid = jnp.array(metallicity_grid)
    age_grid = jnp.array(age_grid)
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
    # inputs = dict(
    #    metallicity=metallicity_grid, age=age_grid, velocities=velocities, mass=mass
    # )
    inputs = MockRubixData(
        MockStarsData(
            velocity=velocities,
            metallicity=metallicity_grid,
            mass=mass,
            age=age_grid,
        ),
        MockGasData(spectra=None),
    )
    return inputs, spectra_reshaped


def test_resample_spectrum_vmap():
    print("initial_spectra shape", initial_spectra.shape)
    print("initial_wavelengths shape", initial_wavelengths.shape)
    print("target_wavelength shape", target_wavelength.shape)
    resample_spectrum_vmap = get_resample_spectrum_vmap(target_wavelength)
    result_vmap = resample_spectrum_vmap(initial_spectra, initial_wavelengths)

    expected_result = jnp.stack(
        [
            resample_spectrum(
                initial_spectra[0], initial_wavelengths[0], target_wavelength
            ),
            resample_spectrum(
                initial_spectra[1], initial_wavelengths[1], target_wavelength
            ),
        ]
    )
    assert jnp.allclose(result_vmap, expected_result)
    assert not jnp.any(jnp.isnan(result_vmap))


def test_resample_spectrum_pmap():
    # For pmap we need to reshape, such that first axis is the device axis
    initial_spectra = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    initial_wavelengths = jnp.array(
        [[4500.0, 5500.0, 6500.0], [4500.0, 5500.0, 6500.0]]
    )
    initial_spectra = reshape_array(initial_spectra)
    initial_wavelengths = reshape_array(initial_wavelengths)
    resample_spectrum_pmap = get_resample_spectrum_pmap(target_wavelength)
    result_pmap = resample_spectrum_pmap(initial_spectra, initial_wavelengths)

    # Check how many GPUs are available, since this defines the shape of the result
    if jax.device_count() > 1:
        expected_result = jnp.array(
            [
                resample_spectrum(
                    initial_spectra[0, 0], initial_wavelengths[0, 0], target_wavelength
                ),
                resample_spectrum(
                    initial_spectra[1, 0], initial_wavelengths[1, 0], target_wavelength
                ),
            ]
        )
        expected_result = reshape_array(expected_result)

    else:
        expected_result = jnp.stack(
            [
                resample_spectrum(
                    initial_spectra[0, 0], initial_wavelengths[0, 0], target_wavelength
                ),
                resample_spectrum(
                    initial_spectra[0, 1], initial_wavelengths[0, 1], target_wavelength
                ),
            ]
        )
    assert jnp.allclose(result_pmap, expected_result)
    assert not jnp.any(jnp.isnan(result_pmap))


def test_calculate_spectra():
    # Use an actual RubixData instance
    mock_rubixdata = RubixData(
        galaxy=Galaxy(),
        stars=StarsData(),
        gas=GasData(),
    )

    # Populate the RubixData object with mock data
    mock_rubixdata.stars.coords = jnp.array([[1, 2, 3]])
    mock_rubixdata.stars.velocity = jnp.array([[4.0, 5.0, 6.0]])
    mock_rubixdata.stars.metallicity = jnp.array(
        [[0.1]]
    )  # 2D array for vmap compatibility
    mock_rubixdata.stars.mass = jnp.array([[1000]])  # 2D array for vmap compatibility
    mock_rubixdata.stars.age = jnp.array([[4.5]])  # 2D array for vmap compatibility
    mock_rubixdata.galaxy.redshift = 0.1
    mock_rubixdata.galaxy.center = jnp.array([0, 0, 0])
    mock_rubixdata.galaxy.halfmassrad_stars = 1

    # Obtain the calculate_spectra function
    calculate_spectra = get_calculate_spectra(sample_config)

    # Mock expected spectra
    expected_spectra_shape = (1, 1, 842)  # Adjust shape as per your data
    expected_spectra = jnp.zeros(expected_spectra_shape)

    # Call the calculate_spectra function
    result = calculate_spectra(mock_rubixdata)

    # Validate the result
    calculated_spectra = result.stars.spectra

    assert calculated_spectra.shape == expected_spectra.shape, "Shape mismatch"
    assert jnp.allclose(
        calculated_spectra, expected_spectra, rtol=RTOL, atol=ATOL
    ), "Spectra values mismatch"
    assert not jnp.any(
        jnp.isnan(calculated_spectra)
    ), "NaN values in calculated spectra"


def test_scale_spectrum_by_mass():
    # Use an actual RubixData instance
    input = RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            velocity=sample_inputs["velocities"],
            metallicity=sample_inputs["metallicity"],
            mass=sample_inputs["mass"],
            age=sample_inputs["age"],
            spectra=sample_inputs["spectra"],
        ),
        gas=GasData(spectra=None),
    )

    # Calculate expected spectra
    expected_spectra = input.stars.spectra * jnp.expand_dims(input.stars.mass, axis=-1)

    # Call the function
    scale_spectrum_by_mass = get_scale_spectrum_by_mass(sample_config)
    result = scale_spectrum_by_mass(input)

    # Print for debugging
    print("Input Mass:", input.stars.mass)
    print("Input Spectra:", input.stars.spectra)
    print("Result Spectra:", result.stars.spectra)
    print("Expected Spectra:", expected_spectra)

    # Assertions
    assert jnp.array_equal(
        result.stars.spectra, expected_spectra
    ), "Spectra scaling mismatch"
    assert not jnp.any(
        jnp.isnan(result.stars.spectra)
    ), "NaN values found in result spectra"


def test_doppler_shift_and_resampling():
    # Obtain the function
    doppler_shift_and_resampling = get_doppler_shift_and_resampling(sample_config)

    # Create an actual RubixData object
    inputs = RubixData(
        galaxy=Galaxy(),  # Create a Galaxy instance as required
        stars=StarsData(
            velocity=sample_inputs["velocities"],
            metallicity=sample_inputs["metallicity"],
            mass=sample_inputs["mass"],
            age=sample_inputs["age"],
            spectra=sample_inputs["spectra"],  # Assign expected spectra
        ),
        gas=GasData(spectra=None),
    )

    # Mock expected spectra
    expected_spectra = sample_inputs["spectra"]

    # Call the function
    result = doppler_shift_and_resampling(inputs)

    # Assertions
    assert hasattr(result.stars, "spectra"), "Result does not have 'spectra'"
    assert not jnp.any(
        jnp.isnan(result.stars.spectra)
    ), "NaN values found in result spectra"
