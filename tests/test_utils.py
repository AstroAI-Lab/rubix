import h5py
import numpy as np
import pytest  # type: ignore # noqa
import yaml
from astropy.cosmology import Planck15 as cosmo

from rubix.utils import (
    SFTtoAge,
    convert_values_to_physical,
    get_config,
    load_galaxy_data,
    print_hdf5_file_structure,
    read_yaml,
)


def test_convert_values_to_physical():
    # Test with some arbitrary values
    value = 10
    a = 0.5
    a_scale_exponent = 2
    hubble_param = 0.7
    hubble_scale_exponent = 2
    CGS_conversion_factor = 1.0

    result = convert_values_to_physical(
        value,
        a,
        a_scale_exponent,
        hubble_param,
        hubble_scale_exponent,
        CGS_conversion_factor,
    )

    # Check if the result is as expected
    expected_result = (
        value
        * a**a_scale_exponent
        * hubble_param**hubble_scale_exponent
        * CGS_conversion_factor
    )
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_convert_values_to_physical_with_cgs_factor_zero():
    # Test with some arbitrary values
    value = 10
    a = 0.5
    a_scale_exponent = 2
    hubble_param = 0.7
    hubble_scale_exponent = 2
    CGS_conversion_factor = 0.0

    result = convert_values_to_physical(
        value,
        a,
        a_scale_exponent,
        hubble_param,
        hubble_scale_exponent,
        CGS_conversion_factor,
    )

    # Check if the result is as expected
    # When CGS_conversion_factor is 0, it should be treated as 1
    expected_result = (
        value * a**a_scale_exponent * hubble_param**hubble_scale_exponent * 1.0
    )
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_SFTtoAge():
    # Test with some arbitrary scale factor
    a = 0.5

    result = SFTtoAge(a)

    # Check if the result is as expected
    expected_result = cosmo.lookback_time((1 / a) - 1).value
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_hdf5_file_structure(tmp_path):
    # Create a temporary HDF5 file
    hdf5_file = tmp_path / "test_file.h5"
    with h5py.File(hdf5_file, "w") as f:
        group = f.create_group("group")
        dataset = group.create_dataset("dataset", (100,), dtype="i")
        dataset[:] = range(100)

    # Run the function to test
    output = print_hdf5_file_structure(str(hdf5_file))

    # Define expected output, adjusting for additional parentheses and newline
    expected_output = (
        f"File: {str(hdf5_file)}\nGroup: group\n    Dataset: dataset (int32) ((100,))\n"
    )

    # Check if the output matches the expected output
    assert output.strip() == expected_output.strip()


def test_empty_hdf5_file(tmp_path):
    # Test with an empty HDF5 file
    hdf5_file = tmp_path / "empty.h5"
    with h5py.File(hdf5_file, "w") as f:
        pass  # No groups or datasets created

    # Run the function to test
    output = print_hdf5_file_structure(str(hdf5_file))

    # Define expected output, adjusting for the extra newline
    expected_output = f"File: {str(hdf5_file)}\n"

    # Check if the output matches the expected output
    assert output.strip() == expected_output.strip()


def test_read_yaml_wrong_path():
    # Test with a non-existent YAML file
    with pytest.raises(Exception) as e:
        read_yaml("non_existent_file.yaml")


def create_test_hdf5_file(path):
    with h5py.File(path, "w") as f:
        # Create groups and datasets
        g = f.create_group("galaxy")
        g.create_dataset("center", data=np.array([0, 0, 0]))
        g.create_dataset("halfmassrad_stars", data=300)
        g.create_dataset("redshift", data=0.5)

        p = f.create_group("particles")
        stars = p.create_group("stars")
        stars.create_dataset("mass", data=np.array([1, 2, 3]))

        # Set attributes
        for key in g.keys():
            g[key].attrs["unit"] = "kpc"
        stars["mass"].attrs["unit"] = "Msun"


def test_load_galaxy_data_success(tmp_path):
    # Create a temporary HDF5 file
    file_path = tmp_path / "test_galaxy.hdf5"
    create_test_hdf5_file(str(file_path))

    # Test the function
    data, units = load_galaxy_data(str(file_path))
    assert data["subhalo_center"].tolist() == [0, 0, 0]
    assert data["subhalo_halfmassrad_stars"] == 300
    assert data["redshift"] == 0.5
    assert data["particle_data"]["stars"]["mass"].tolist() == [1, 2, 3]
    assert units["galaxy"]["center"] == "kpc"
    assert units["galaxy"]["halfmassrad_stars"] == "kpc"
    assert units["galaxy"]["redshift"] == "kpc"
    assert units["stars"]["mass"] == "Msun"


def test_load_galaxy_data_failure(tmp_path):
    # Create a temporary HDF5 file with missing datasets
    file_path = tmp_path / "test_faulty_galaxy.hdf5"
    with h5py.File(file_path, "w") as f:
        f.create_group("galaxy")

    # Test the function should raise an error
    with pytest.raises(KeyError):
        load_galaxy_data(str(file_path))


def test_load_galaxy_data_file_does_not_exist(tmp_path):
    # Create a temporary HDF5 file with missing datasets
    # Test the function should raise an error
    with pytest.raises(FileNotFoundError):
        load_galaxy_data("wrong path")


def test_read_yaml(tmp_path):
    # Create a temporary YAML file in the temporary directory
    test_file = tmp_path / "test.yaml"

    # Data to write to the YAML file
    data = {"key": "value", "numbers": [1, 2, 3]}

    # Write the YAML data to the file
    with open(test_file, "w") as file:
        yaml.dump(data, file)

    # Use the read_yaml function to read the data back from the file
    result = read_yaml(str(test_file))

    # Check that the data read from the file matches the original data
    assert (
        result == data
    ), "The data read from the YAML file does not match the expected data"


def test_read_yaml_error_handling(tmp_path):
    # Define a path to a non-existent file
    non_existent_file = tmp_path / "non_existent.yaml"

    # Expect a RuntimeError when trying to read a non-existent file
    with pytest.raises(RuntimeError) as excinfo:
        read_yaml(str(non_existent_file))

    assert "Something went wrong while reading yaml file" in str(
        excinfo.value
    ), "Expected RuntimeError not raised"


def test_get_config_with_dict():
    config = {"test_key": "test_value"}
    result = get_config(config)
    assert result == config


def test_get_config_with_str(mocker):
    config_file = "config.yaml"
    expected_result = {"key": "value"}
    mocker.patch("rubix.utils.read_yaml", return_value=expected_result)
    result = get_config(config_file)
    assert result == expected_result
