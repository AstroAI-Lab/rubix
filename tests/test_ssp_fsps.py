import os
import sys
from importlib import reload
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from rubix.spectra.ssp.fsps_grid import write_fsps_data_to_disk
from rubix.spectra.ssp.grid import SSPGrid


# Mock the fsps.StellarPopulation class
class MockFSPS:
    class StellarPopulation:
        def __init__(self, zcontinuous=0, **kwargs):
            self.zlegend = np.array([0.001, 0.01, 0.1])
            self.log_age = np.array([9.0, 9.1, 9.2])

        def get_spectrum(self, peraa=True, **kwargs):
            return (
                np.array([4000, 4100, 4200]),
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            )


def test_no_fsps():
    # Test when python-fsps is not installed
    with patch.dict(sys.modules, {"fsps": None}):
        reload(sys.modules["rubix.spectra.ssp.fsps_grid"])
        import rubix.spectra.ssp.fsps_grid

        # result = retrieve_ssp_data_from_fsps()
        assert rubix.spectra.ssp.fsps_grid.HAS_FSPS is False


# in case FSPS is not installed, fake the FSPS package
sys.modules["fsps"] = MockFSPS()


def test_retrieve_ssp_data_from_fsps_no_fsps():
    # Test when python-fsps is not installed
    from rubix.spectra.ssp.fsps_grid import retrieve_ssp_data_from_fsps

    with patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", False):
        with pytest.raises(AssertionError):
            result = retrieve_ssp_data_from_fsps()
            assert result is None


def test_retrieve_ssp_data_from_fsps():
    with patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True):
        import rubix.spectra.ssp.fsps_grid

        # Call the function
        result = rubix.spectra.ssp.fsps_grid.retrieve_ssp_data_from_fsps()
        mock_sp_instance = MockFSPS.StellarPopulation()

        # Check the returned SSPGrid object
        assert isinstance(result, SSPGrid)
        assert np.allclose(result.metallicity, np.log10(mock_sp_instance.zlegend))
        assert np.allclose(result.age, mock_sp_instance.log_age - 9.0)
        assert np.allclose(
            result.wavelength, np.array([4000, 4100, 4200]) - 50
        )  # because wavelengths are shifted by the calculated offset in the mock to be centered
        assert np.allclose(
            result.flux,
            np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
        )


def test_retrieve_ssp_data_from_fsps_with_kwargs():
    with (
        patch.dict("sys.modules", {"fsps": MockFSPS()}),
        patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True),
    ):

        from rubix.spectra.ssp.fsps_grid import retrieve_ssp_data_from_fsps

        # Call the function with additional keyword arguments
        result = retrieve_ssp_data_from_fsps(add_neb_emission=False, imf_type=1)
        mock_sp_instance = MockFSPS.StellarPopulation()

        # Check the returned SSPGrid object
        assert isinstance(result, SSPGrid)
        assert np.allclose(result.metallicity, np.log10(mock_sp_instance.zlegend))
        assert np.allclose(result.age, mock_sp_instance.log_age - 9.0)
        assert np.allclose(
            result.wavelength, np.array([4000, 4100, 4200]) - 50
        )  # because wavelengths are shifted by 50 in the mock to be centered
        assert np.allclose(
            result.flux,
            np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
        )


# test write to disk of fsps template
def test_write_fsps_data_to_disk(tmpdir):
    # Create a temporary directory for testing
    temp_dir = tmpdir.mkdir("test_dir")

    # Define the output file name
    outname = "output.h5"

    with (
        patch.dict("sys.modules", {"fsps": MockFSPS()}),
        patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True),
    ):

        # Call the function
        write_fsps_data_to_disk(outname, file_location=temp_dir)

    # Check if the file is a valid HDF5 file
    file_path = os.path.join(temp_dir, outname)
    with h5py.File(file_path, "r") as hdf:
        assert "metallicity" in hdf
        assert "age" in hdf
        assert "wavelength" in hdf
        assert "flux" in hdf


def test_write_fsps_data_to_disk_invalid_file_location(tmpdir):
    # Define an invalid file location
    invalid_location = "invalid_path"

    # Define the output file name
    outname = "output.h5"

    with (
        patch.dict("sys.modules", {"fsps": MockFSPS()}),
        patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True),
    ):
        # Call the function with an invalid file location
        with pytest.raises(FileNotFoundError):
            write_fsps_data_to_disk(outname, file_location=invalid_location)
