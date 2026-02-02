import os
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from rubix.galaxy import IllustrisAPI


@pytest.fixture
def api_key():
    return "mock_key"


@pytest.fixture
def api_instance(api_key, tmp_path):
    save_data_path = tmp_path / "test_data"
    save_data_path.mkdir()
    return IllustrisAPI(
        api_key=api_key,
        save_data_path=str(save_data_path),
        particle_type=["stars"],
        simulation="TNG50-1",
        snapshot=99,
    )


def test_get_subhalo_with_mock(api_instance, requests_mock):
    # Setup mock response
    mock_url = f"{api_instance.baseURL}/subhalos/11"
    mock_response = {
        "id": 11,
        "name": "Mock Subhalo",
        # Add more fields as expected by your test
    }
    # Include the 'content-type' header in the mock response
    headers = {"content-type": "application/json"}
    requests_mock.get(mock_url, json=mock_response, headers=headers)

    # Now, when api_instance.get_subhalo(11) is called, it will receive the mock_response
    result = api_instance.get_subhalo(11)
    assert result["id"] == 11  # Example assertion based on the mock response
    assert result["name"] == "Mock Subhalo"


@pytest.mark.parametrize(
    "input_id,particle_type,field_name,expected_exception",
    [
        (11, "stars", "Masses", None),
        ("invalid_id", "stars", "Masses", ValueError),
        (11, "invalid_type", "Masses", ValueError),
        (11, "invalid_type", "", ValueError),
    ],
)
def test_get_particle_data(
    api_instance,
    input_id,
    particle_type,
    field_name,
    expected_exception,
    requests_mock,
):
    # create mock response

    mock_url = f"{api_instance.baseURL}/subhalos/{input_id}/cutout.hdf5?{particle_type}={field_name}"
    mock_response = {
        "id": 11,
        "name": "Mock Subhalo",
        "PartType4": {field_name: ["mocked_data"]},
    }

    requests_mock.get(mock_url, json=mock_response)

    mock_dataset = np.array([1.0, 2.0, 3.0])

    data = {"PartType4": {"Masses": mock_dataset}}
    mock_hdf5 = MagicMock()

    mock_hdf5.__enter__.return_value = data
    print(mock_hdf5.keys())

    with (
        patch("os.path.exists", return_value=True),
        patch("h5py.File", return_value=mock_hdf5) as mock_hdf5,
    ):

        # Check the structure of mock_hd5 file
        #

        with h5py.File("mocked_file", "r") as f:
            print(f.keys())
        mock_response = {
            "id": 11,
            "name": "Mock Subhalo",
            "PartType4": {field_name: ["mocked_data"]},
        }
        # Include the 'content-type' header in the mock response
        headers = {"content-type": "application/json"}
        requests_mock.get(mock_url, json=mock_response, headers=headers)
        if expected_exception:
            with pytest.raises(expected_exception):
                api_instance.get_particle_data(input_id, particle_type, field_name)
        else:
            data_response = api_instance.get_particle_data(
                input_id, particle_type, field_name
            )
            print(data)
            assert isinstance(data, dict)
            assert field_name in data["PartType4"]
            np.testing.assert_array_equal(
                data["PartType4"][field_name], data_response["PartType4"][field_name]
            )


def test__init__():
    api = IllustrisAPI(api_key="test_key")
    assert api.headers == {"api-key": "test_key"}
    assert (
        api.particle_type == ["stars"]
        or api.particle_type == ["gas"]
        or api.particle_type == ["stars", "gas"]
        or api.particle_type == ["gas", "stars"]
    )
    assert api.snapshot == 99
    assert api.simulation == "TNG50-1"
    assert api.baseURL == "http://www.tng-project.org/api/TNG50-1/snapshots/99"
    assert "TNG50-1" in api.__str__()


def test_init_with_logger():
    import logging

    logger = logging.getLogger("test_logger")

    api = IllustrisAPI(api_key="test_key", logger=logger)
    assert api.logger == logger


def test_get_api_key(api_key):
    assert api_key is not None


def test_no_api_key():
    with pytest.raises(ValueError):
        api_instance = IllustrisAPI(api_key=None)
        assert api_instance is None


def test_get_http_error(api_instance, requests_mock):
    # Simulate an HTTP 404 error
    mock_url = f"{api_instance.baseURL}/nonexistent/path"
    requests_mock.get(mock_url, status_code=404)

    with pytest.raises(ValueError) as exc_info:
        api_instance._get(mock_url)

    assert "404 Client Error" in str(exc_info.value)


def test_get_missing_content_disposition_header(api_instance, requests_mock):
    # Simulate a response without the 'content-disposition' header
    mock_url = f"{api_instance.baseURL}/data_without_content_disposition"
    requests_mock.get(
        mock_url,
        content=b"Some binary data",
        headers={"content-type": "application/octet-stream"},
    )

    # Assert that a ValueError is raised due to the missing 'content-disposition' header
    with pytest.raises(ValueError) as exc_info:
        api_instance._get(mock_url)

    assert "No content-disposition header found. Cannot save file." in str(
        exc_info.value
    )


def test_subhalo_id_not_integers(api_instance):
    with pytest.raises(ValueError) as exc_info:
        api_instance.get_subhalo("invalid_id")
    assert "ID should be an integer." in str(exc_info.value)


def test_get_writes_file_correctly(api_instance, requests_mock, tmp_path):
    # Simulate a valid response with 'content-disposition' header
    mock_url = f"{api_instance.baseURL}/downloadable_content"
    mock_filename = "test_data"
    mock_content = b"Sample binary content"
    headers = {
        "content-type": "application/octet-stream",
        "content-disposition": f"attachment; filename={mock_filename}",
    }
    requests_mock.get(mock_url, content=mock_content, headers=headers)

    # Use a temporary directory provided by pytest to avoid writing to the disk
    api_instance.DATAPATH = str(tmp_path)

    # Call the _get method and assert it returns the correct filename
    returned_filename = api_instance._get(mock_url)
    assert (
        returned_filename == mock_filename
    ), "The returned filename does not match the expected value."

    # Check that the file was saved correctly
    saved_file_path = tmp_path / f"{mock_filename}.hdf5"
    with open(saved_file_path, "rb") as f:
        saved_content = f.read()
        assert (
            saved_content == mock_content
        ), "The saved file content does not match the mock response."


def test_get_with_provided_name(api_instance, requests_mock, tmp_path):
    # Adjusted to simulate a binary response
    mock_url = f"{api_instance.baseURL}/content_with_name"
    provided_name = "provided_test_data"
    mock_content = b"Provided content binary"
    headers = {
        "content-type": "application/octet-stream",
        "content-disposition": f"attachment; filename={provided_name}",
    }
    requests_mock.get(mock_url, content=mock_content, headers=headers)

    api_instance.DATAPATH = str(tmp_path)

    returned_filename = api_instance._get(mock_url, name=provided_name)

    assert returned_filename == provided_name

    saved_file_path = tmp_path / f"{provided_name}.hdf5"
    with open(saved_file_path, "rb") as f:
        saved_content = f.read()
        assert saved_content == mock_content


@pytest.fixture
def galaxy_data():
    return {"PartType0": {"Field1": [1, 2, 3]}, "PartType4": {"Field2": [4, 5, 6]}}


@pytest.fixture
def subhalo_data():
    return {"id": 0, "mass": 10e10, "other_property": "value"}


def test_load_galaxy(api_instance, galaxy_data, subhalo_data):
    id = 0
    with (
        patch.object(api_instance, "_get") as mock_get,
        patch.object(
            api_instance, "get_subhalo", return_value=subhalo_data
        ) as mock_get_subhalo,
        patch.object(api_instance, "_append_subhalo_data") as mock_append_subhalo_data,
        patch.object(
            api_instance, "_load_hdf5", return_value=galaxy_data
        ) as mock_load_hdf5,
    ):
        api_instance.DEFAULT_FIELDS = {"stars": "Masses"}
        result = api_instance.load_galaxy(id=id)

        # Verify the calls
        # the API loads only the stars data, so here we check for the correct fields
        # RBX-25 This may be changed if we  need to load gas particles as well
        # mock_get.assert_called_once_with(
        #     f"{api_instance.baseURL}/subhalos/{id}/cutout.hdf5?gas={','.join(api_instance.DEFAULT_FIELDS['PartType0'])}&stars={','.join(api_instance.DEFAULT_FIELDS['PartType4'])}",
        #     name=f"galaxy-id-{id}",
        # )
        mock_get.assert_called_once_with(
            f"{api_instance.baseURL}/subhalos/{id}/cutout.hdf5?stars=Masses",
            name=f"galaxy-id-{id}",
        )
        mock_get_subhalo.assert_called_once_with(id)
        mock_append_subhalo_data.assert_called_once_with(subhalo_data, id)
        mock_load_hdf5.assert_called_once_with(filename=f"galaxy-id-{id}")

        # Verify the result
        assert (
            result == galaxy_data
        ), "The returned data does not match the expected galaxy data."


def test_load_galaxy_unsupported_particle_type(api_instance, subhalo_data):
    id = 0
    with (
        patch.object(api_instance, "_get") as mock_get,
        patch.object(api_instance, "get_subhalo", return_value=subhalo_data),  # noqa
    ):
        api_instance.DEFAULT_FIELDS = {"stars": "Masses"}
        api_instance.particle_type = ["unsupported"]
        with pytest.raises(ValueError) as exc_info:
            api_instance.load_galaxy(id=id)
        assert "Got unsupported particle type" in str(exc_info.value)


def test_load_galaxy_multiple_fields(api_instance, galaxy_data, subhalo_data):
    id = 0
    with (
        patch.object(api_instance, "_get") as mock_get,
        patch.object(
            api_instance, "get_subhalo", return_value=subhalo_data
        ) as mock_get_subhalo,
        patch.object(api_instance, "_append_subhalo_data") as mock_append_subhalo_data,
        patch.object(
            api_instance, "_load_hdf5", return_value=galaxy_data
        ) as mock_load_hdf5,
    ):
        api_instance.DEFAULT_FIELDS = {"stars": ["Masses", "Coordinates"]}
        result = api_instance.load_galaxy(id=id)
        mock_get.assert_called_once_with(
            f"{api_instance.baseURL}/subhalos/{id}/cutout.hdf5?stars=Masses,Coordinates",
            name=f"galaxy-id-{id}",
        )
        mock_get_subhalo.assert_called_once_with(id)
        mock_append_subhalo_data.assert_called_once_with(subhalo_data, id)
        mock_load_hdf5.assert_called_once_with(filename=f"galaxy-id-{id}")

        # Verify the result
        assert (
            result == galaxy_data
        ), "The returned data does not match the expected galaxy data."


def test_load_galaxy_multiple_particle_types(api_instance, galaxy_data, subhalo_data):
    id = 0
    with (
        patch.object(api_instance, "_get") as mock_get,
        patch.object(
            api_instance, "get_subhalo", return_value=subhalo_data
        ) as mock_get_subhalo,
        patch.object(api_instance, "_append_subhalo_data") as mock_append_subhalo_data,
        patch.object(
            api_instance, "_load_hdf5", return_value=galaxy_data
        ) as mock_load_hdf5,
    ):
        api_instance.particle_type = ["stars", "gas"]
        api_instance.DEFAULT_FIELDS["stars"] = ["Masses", "Coordinates"]
        api_instance.DEFAULT_FIELDS["gas"] = ["Coordinates"]
        result = api_instance.load_galaxy(id=id)
        mock_get.assert_called_once_with(
            f"{api_instance.baseURL}/subhalos/{id}/cutout.hdf5?stars=Masses,Coordinates&gas=Coordinates",
            name=f"galaxy-id-{id}",
        )
        mock_get_subhalo.assert_called_once_with(id)
        mock_append_subhalo_data.assert_called_once_with(subhalo_data, id)
        mock_load_hdf5.assert_called_once_with(filename=f"galaxy-id-{id}")

        # Verify the result
        assert (
            result == galaxy_data
        ), "The returned data does not match the expected galaxy data."


def test_append_subhalo_data(api_instance, tmp_path):
    # Setup
    subhalo_data = {
        "redshift": 0,
        "center": np.array([0, 0, 0]),
        "halfmassrad": 0.5,
        "some-dict": {"key": "value"},
    }
    id = 0

    # Path setup
    file_path = tmp_path / f"galaxy-id-{id}.hdf5"
    api_instance.DATAPATH = str(tmp_path)

    # Execute
    api_instance._append_subhalo_data(subhalo_data, id)

    # Verify
    with h5py.File(file_path, "r") as f:
        assert "SubhaloData" in f, "SubhaloData group was not created."

        # Check for the presence of non-dictionary data
        for key, value in subhalo_data.items():
            if isinstance(value, dict):
                assert (
                    key not in f["SubhaloData"].keys()  # type: ignore
                ), f"Dictionary data '{key}' should not be in SubhaloData."
            else:
                assert (
                    key in f["SubhaloData"]
                ), f"Non-dictionary data '{key}' was not found in SubhaloData."  # type: ignore
                dataset = f["SubhaloData"][key]  # type: ignore
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(
                        dataset[:], value  # type: ignore
                    ), f"Data '{key}' does not match the expected value."  # type: ignore
                else:
                    assert (
                        dataset[()] == value  # type: ignore
                    ), f"Data '{key}' does not match the expected value."


def test_load_hdf5_success(api_instance, tmp_path):
    # Create a mock HDF5 file with test data
    filename = "test_data"
    file_path = tmp_path / f"{filename}.hdf5"
    test_data = {
        "PartType0": {"Density": np.array([1.0, 2.0, 3.0])},
        "PartType1": {"Velocity": np.array([4.0, 5.0, 6.0])},
    }

    with h5py.File(file_path, "w") as f:
        for type, fields in test_data.items():
            group = f.create_group(type)
            for field_name, values in fields.items():
                group.create_dataset(field_name, data=values)

    # Ensure the API's DATAPATH is set to the tmp_path
    api_instance.DATAPATH = str(tmp_path)

    # Load the file using the method to be tested
    loaded_data = api_instance._load_hdf5(filename)

    # Assert the loaded data matches the test data
    for type, fields in test_data.items():
        assert type in loaded_data, f"{type} not found in loaded data."
        for field_name, values in fields.items():
            np.testing.assert_array_equal(
                values,
                loaded_data[type][field_name],
                err_msg=f"Data for {type}/{field_name} does not match.",
            )


def test_load_hdf5_file_not_exist(api_instance, tmp_path):
    filename = "nonexistent_file"
    api_instance.DATAPATH = str(tmp_path)

    # Try loading a file that doesn't exist and assert a ValueError is raised
    with pytest.raises(ValueError) as exc_info:
        api_instance._load_hdf5(filename)
    assert "does not exist" in str(exc_info.value)


def test_load_hdf5_ignore_header(api_instance, tmp_path):
    # Create a mock HDF5 file including a 'Header' group
    filename = "test_with_header"
    file_path = tmp_path / f"{filename}.hdf5"

    with h5py.File(file_path, "w") as f:
        header = f.create_group("Header")
        header.create_dataset("SomeInfo", data=np.array([7.0, 8.0, 9.0]))
        parttype0 = f.create_group("PartType0")
        parttype0.create_dataset("Density", data=np.array([1.0, 2.0, 3.0]))

    # Set the DATAPATH and load the file
    api_instance.DATAPATH = str(tmp_path)
    loaded_data = api_instance._load_hdf5(filename)

    # Ensure 'Header' is ignored and other data is loaded correctly
    assert "Header" not in loaded_data, "'Header' group should be ignored."
    assert "PartType0" in loaded_data, "'PartType0' group not found in loaded data."
    np.testing.assert_array_equal(
        [1.0, 2.0, 3.0],
        loaded_data["PartType0"]["Density"],
        err_msg="'Density' data does not match.",
    )


def test_load_hdf5_with_extension(api_instance, tmp_path):
    # Test to ensure method handles filenames correctly with ".hdf5" extension
    filename = "test_data_with_extension.hdf5"
    expected_filename_without_extension = "test_data_with_extension"
    file_path = tmp_path / filename
    test_data = {"PartType0": {"Density": np.array([1.0, 2.0, 3.0])}}

    with h5py.File(file_path, "w") as f:
        for type, fields in test_data.items():
            group = f.create_group(type)
            for field_name, values in fields.items():
                group.create_dataset(field_name, data=values)

    # Ensure the API's DATAPATH is set to the tmp_path
    api_instance.DATAPATH = str(tmp_path)

    # Load the file using the method to be tested
    loaded_data = api_instance._load_hdf5(filename)

    # Assert the loaded data matches the test data
    for type, fields in test_data.items():
        assert type in loaded_data, f"{type} not found in loaded data."
        for field_name, values in fields.items():
            np.testing.assert_array_equal(
                values,
                loaded_data[type][field_name],
                err_msg=f"Data for {type}/{field_name} does not match.",
            )

    # Verify the correct handling of the filename with ".hdf5" extension
    full_path_constructed = os.path.join(
        api_instance.DATAPATH, f"{expected_filename_without_extension}.hdf5"
    )
    assert os.path.exists(
        full_path_constructed
    ), "File with expected filename doesn't exist."


def test_load_galaxy_overwrite_reuse(api_instance, tmp_path):
    id = 1
    filename = f"galaxy-id-{id}"
    filename_with_extension = f"{filename}.hdf5"
    file_path = tmp_path / filename_with_extension

    # Create a placeholder for the existing file
    file_path.touch()

    # Ensure the API's DATAPATH is set to the tmp_path
    api_instance.DATAPATH = str(tmp_path)

    with patch.object(
        api_instance, "_load_hdf5", return_value={"data": "loaded"}
    ) as mock_load_hdf5:
        # Test reuse without overwrite, file exists
        result = api_instance.load_galaxy(id=id, overwrite=False, reuse=True)
        mock_load_hdf5.assert_called_once_with(filename=filename)
        assert result == {
            "data": "loaded"
        }, "The data should be loaded from the existing file."

        # Reset mock
        mock_load_hdf5.reset_mock()

        # Test overwrite, file exists
        with (
            patch.object(api_instance, "_get") as mock_get,
            patch.object(
                api_instance, "get_subhalo", return_value={}
            ) as mock_get_subhalo,
            patch.object(api_instance, "_append_subhalo_data"),
        ):
            result = api_instance.load_galaxy(id=id, overwrite=True, reuse=False)
            mock_get.assert_called()
            mock_get_subhalo.assert_called()
            mock_load_hdf5.assert_called_once_with(filename=filename)
            assert "data" in result, "The data should be refreshed and loaded."

        # Test error raise when file exists, and neither overwrite nor reuse are true
        with pytest.raises(ValueError) as exc_info:
            api_instance.load_galaxy(id=id, overwrite=False, reuse=False)
        assert "already exists" in str(
            exc_info.value
        ), "An error should be raised when attempting to download without overwrite and reuse."
