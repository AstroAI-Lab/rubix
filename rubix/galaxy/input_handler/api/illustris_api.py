import logging
import os
from typing import Any, List, Optional, Union

import h5py
import requests

from rubix import config


class IllustrisAPI:
    """
    This class is used to load data from the Illustris API.

    It loads both subhalo data and particle data from a given simulation,
    snapshot, and subhalo ID.

    Check the source for the API documentation for more information:
    https://www.tng-project.org/data/docs/api/

    Attributes:
        URL (str): Base URL of the Illustris API.
        DEFAULT_FIELDS (dict[str, list[str]]): Default particle fields to
            download per particle type.

    Args:
        api_key (str): API key for authenticating with the Illustris API.
        particle_type (Optional[List[str]], optional): Particle categories to
            download (default: ["stars", "gas"]).
        simulation (str, optional): Simulation to connect to
            (default: "TNG50-1").
        snapshot (int, optional): Snapshot ID to query (default: 99).
        save_data_path (str, optional): Directory where downloaded data
            will be stored.
        logger (Optional[logging.Logger], optional): Logger instance for debug
            output.

    Raises:
        ValueError: If the API key is missing.
    """

    URL: str = "http://www.tng-project.org/api/"
    DEFAULT_FIELDS: dict[str, list[str]] = config["IllustrisAPI"]["DEFAULT_FIELDS"]

    def __init__(
        self,
        api_key: str,
        particle_type: Optional[List[str]] = None,
        simulation: str = "TNG50-1",
        snapshot: int = 99,
        save_data_path: str = "./api_data",
        logger: Optional[logging.Logger] = None,
    ):

        if api_key is None:
            raise ValueError("Please set the API key.")

        self.headers = {"api-key": api_key}
        self.particle_type = particle_type or ["stars", "gas"]
        self.snapshot = snapshot
        self.simulation = simulation
        self.baseURL = f"{self.URL}{self.simulation}/snapshots/{self.snapshot}"
        self.DATAPATH = save_data_path
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def _get(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> Union[dict[str, Any], str]:
        """
        Get data from the Illustris API.

        Args:
            path (str): Path to load data from.
            params (Optional[dict[str, Any]], optional):
                Query parameters for the request.
            name (Optional[str], optional):
                Filename to use when saving the response.

        Returns:
            dict[str, Any] | str: JSON data or the saved filename.

        Raises:
            ValueError: On HTTP failures or if a download cannot be saved.
        """

        os.makedirs(self.DATAPATH, exist_ok=True)
        try:
            self.logger.debug(
                f"Performing GET request from {path}, with parameters {params}"
            )
            r = requests.get(path, params=params, headers=self.headers)
            # raise exception if response code is not HTTP SUCCESS (200)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise ValueError(err)

        if r.headers["content-type"] == "application/json":
            return r.json()  # parse json responses automatically
        if "content-disposition" not in r.headers:
            raise ValueError("No content-disposition header found. Cannot save file.")
        filename = (
            r.headers["content-disposition"].split("filename=")[1]
            if name is None
            else name
        )
        file_path = os.path.join(self.DATAPATH, f"{filename}.hdf5")
        with open(file_path, "wb") as f:
            f.write(r.content)
        return filename  # return the filename string

    def get_subhalo(self, id: int) -> dict[str, Any]:
        """Get subhalo data for a given Illustris ID.

        Args:
            id (int): Subhalo ID to load.

        Returns:
            dict[str, Any]: Subhalo metadata.

        Raises:
            ValueError: If the provided ID is not an integer.
        """

        if not isinstance(id, int):
            raise ValueError("ID should be an integer.")
        return self._get(f"{self.baseURL}/subhalos/{id}")

    def _load_hdf5(self, filename: str) -> dict[str, Any]:
        """Load a previously downloaded HDF5 file.

        Args:
            filename (str): Base name of the file to load.

        Returns:
            dict[str, Any]: Loaded data grouped by particle type.

        Raises:
            ValueError: If the requested file cannot be found.
        """
        # Check if filename ends with .hdf5
        if filename.endswith(".hdf5"):
            filename = filename[:-5]
        returndict = dict()
        file_path = os.path.join(self.DATAPATH, f"{filename}.hdf5")
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist.")

        with h5py.File(file_path, "r") as f:
            for type in f.keys():
                if type == "Header":
                    continue
                # create new dictionary for each type
                returndict[type] = dict()
                for fields in f[type].keys():  # type: ignore
                    field_array = f[type][fields][()]  # type: ignore
                    returndict[type][fields] = field_array

        return returndict

    def get_particle_data(
        self,
        id: int,
        particle_type: str,
        fields: Union[str, List[str]],
    ) -> dict[str, Any]:
        """Download particle cutouts for a subhalo.

        Args:
            id (int): Subhalo ID to load.
            particle_type (str): Particle species to request.
            fields (Union[str, List[str]]): Data fields to include.

        Returns:
            dict[str, Any]: Downloaded particle data.

        Raises:
            ValueError: If the particle type, fields, or ID are invalid.
        """
        # Get fields in the right format
        if isinstance(fields, str):
            if fields == "":
                raise ValueError("Fields should not be empty.")
            fields = [fields]

        if not isinstance(id, int):
            raise ValueError("ID should be an integer.")
        fields = ",".join(fields)

        if particle_type not in ["stars", "gas", "dm"]:
            raise ValueError("Particle type should be 'stars', 'gas', or 'dm'.")
        url = f"{self.baseURL}/subhalos/{id}/cutout.hdf5?" f"{particle_type}={fields}"
        self._get(url, name="cutout")
        data = self._load_hdf5("cutout")
        return data

    def load_galaxy(
        self,
        id: int,
        overwrite: bool = False,
        reuse: bool = False,
    ) -> dict[str, Any]:
        """Download subhalo and particle data for a galaxy.

        The function fetches subhalo metadata and configured particle fields
        and stores everything in a local HDF5 file.

        Args:
            id (int): Subhalo ID to download.
            overwrite (bool, optional): Overwrite an existing file if True.
            reuse (bool, optional): Reuse an existing file instead of
                redownloading it.

        Returns:
            dict[str, Any]: Loaded galaxy data.

        Raises:
            ValueError: If the download is blocked by an existing file or an
                unsupported particle type is configured.

        Example:

                >>> illustris_api = IllustrisAPI(
                ...     api_key,
                ...     simulation="TNG50-1",
                ...     snapshot=99,
                ...     particle_type=["stars", "gas"],
                ... )
                >>> illustris_api.load_galaxy(id=0)
        """

        # Check if there is already a file with the same name
        if os.path.exists(os.path.join(self.DATAPATH, f"galaxy-id-{id}.hdf5")):
            # If file exists, check if we should overwrite it
            if not overwrite:
                # If we should not overwrite it, check if we should reuse it
                if reuse:
                    self.logger.info(
                        "Reusing existing file galaxy-id-%d.hdf5. "
                        "If you want to download the data again, "
                        "set reuse=False." % id
                    )
                    return self._load_hdf5(filename=f"galaxy-id-{id}")
                else:
                    # If we should not reuse it, raise an error
                    raise ValueError(
                        "File with name galaxy-id-%d.hdf5 already exists. "
                        "Please remove it before downloading the data, "
                        "or set overwrite=True, or reuse=True to load the data." % id
                    )
            else:
                self.logger.info(
                    (
                        f"Found existing file galaxy-id-{id}.hdf5, "
                        "but overwrite is set to True. "
                        "Overwriting the file."
                    )
                )

        # Check which particles we want to load
        self.logger.debug(f"Loading galaxy with ID {id}")
        url = f"{self.baseURL}/subhalos/{id}/cutout.hdf5?"

        for particle_type in self.particle_type:
            # Check if particle type is valid
            if particle_type not in self.DEFAULT_FIELDS.keys():
                raise ValueError(
                    (
                        "Got unsupported particle type. "
                        f"Supported types are "
                        f"{list(self.DEFAULT_FIELDS.keys())} "
                        f"and we got {particle_type}."
                    )
                )

            fields = self.DEFAULT_FIELDS[particle_type]
            # Check if fields is a list
            if isinstance(fields, list):
                fields = ",".join(fields)
            url += f"{particle_type}={fields}&"

        # Remove the last "&" from the url
        if url[-1] == "&":
            url = url[:-1]

        self._get(url, name=f"galaxy-id-{id}")
        subhalo_data = self.get_subhalo(id)
        self._append_subhalo_data(subhalo_data, id)
        data = self._load_hdf5(filename=f"galaxy-id-{id}")
        return data

    def _append_subhalo_data(self, subhalo_data, id):
        self.logger.debug(f"Appending subhalo data for subhalo {id}")
        # Append subhalo data to the HDF5 file
        file_path = os.path.join(self.DATAPATH, f"galaxy-id-{id}.hdf5")
        with h5py.File(file_path, "a") as f:
            f.create_group("SubhaloData")
            for key in subhalo_data.keys():
                if isinstance(subhalo_data[key], dict):
                    continue
                f["SubhaloData"].create_dataset(
                    key,
                    data=subhalo_data[key],
                )  # type: ignore

    def __str__(self) -> str:
        return (
            f"IllustrisAPI: Simulation {self.simulation}, "
            f"Snapshot {self.snapshot}, "
            f"Particle Type {self.particle_type}"
        )
