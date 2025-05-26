import os
from typing import List, Union

import h5py
import requests

from rubix import config


class IllustrisAPI:
    """
    This class is used to load data from the Illustris API.

    It loads both subhalo data and particle data from a given simulation, snapshot, and subhalo ID.

    Check the source for the API documentation for more information: https://www.tng-project.org/data/docs/api/
    """

    URL = "http://www.tng-project.org/api/"
    DEFAULT_FIELDS = config["IllustrisAPI"]["DEFAULT_FIELDS"]

    def __init__(
        self,
        api_key,
        particle_type: list = ["stars", "gas"],
        simulation="TNG50-1",
        snapshot=99,
        save_data_path="./api_data",
        logger=None,
    ):
        """Illustris API class.

        Class to load data from the Illustris API.

        Parameters
        ----------
        api_key : str
            API key for the Illustris API.
        particle_type : str
            Particle type to load. Default is "stars".
        simulation : str
            Simulation to load from. Default is "TNG100-1".
        snapshot : int
            Snapshot to load from. Default is 99.
        """

        if api_key is None:
            raise ValueError("Please set the API key.")

        self.headers = {"api-key": api_key}
        self.particle_type = particle_type
        self.snapshot = snapshot
        self.simulation = simulation
        self.baseURL = f"{self.URL}{self.simulation}/snapshots/{self.snapshot}"
        self.DATAPATH = save_data_path
        if logger is None:
            import logging

            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def _get(self, path, params=None, name=None):
        """Get data from the Illustris API.

        Parameters
        ----------
        path : str
            Path to load from.
        params : dict
            Parameters to pass to the API.
        name : str
            Name to save the file as. If None, the name will be taken from the content-disposition header.
        Returns
        -------
        r : requests object
            The requests object.

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

    def get_subhalo(self, id):
        """
        Get subhalo data from the Illustris API.

        Returns the subhalo data for the given subhalo ID.

        Args:
            id (int): Subhalo ID to load.

        Returns:
            The subhalo data as a dictionary (r).

        """

        if not isinstance(id, int):
            raise ValueError("ID should be an integer.")
        return self._get(f"{self.baseURL}/subhalos/{id}")

    def _load_hdf5(self, filename):
        """Load HDF5 file.

        Loads the HDF5 file with the given filename.

        Parameters
        ----------
        filename : str
            Filename to load.
        Returns
        -------
        returndict : dict
            Dictionary containing the data from the HDF5 file.
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
                    returndict[type][fields] = f[type][fields][()]  # type: ignore

        return returndict

    def get_particle_data(self, id: int, particle_type, fields: Union[str, List[str]]):
        """
        Get particle data from the Illustris API.

        Returns the particle data for the given subhalo ID.

        Args:
            id (int): Subhalo ID to load.
            fields (str or list): Fields to load. If a string, the fields should be comma-separated.

        Returns:
            Dictionary containing the particle data in the given fields (data).
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
        url = f"{self.baseURL}/subhalos/{id}/cutout.hdf5?{particle_type}={fields}"
        self._get(url, name="cutout")
        data = self._load_hdf5("cutout")
        return data

    def load_galaxy(self, id: int, overwrite: bool = False, reuse: bool = False):
        """
        Download Galaxy Data from the Illustris API.

        This function downloads both the subhalo data and the particle data for stars and gas particles, for the fields specified in DEFAULT_FIELDS.
        It saves the data in a HDF5 file.

        Args:
            id (int): The ID of the subhalo to download.
            overwrite (bool): Whether to overwrite the file if it already exists. Default is False.
            reuse (bool): Whether to reuse the file if it already exists. Default is False.

        Returns:
            The galaxy data as dictionary.

        Example
        --------
        >>> illustris_api = IllustrisAPI(api_key, simulation="TNG50-1", snapshot=99, particle_type=["stars", "gas"])
        >>> data = illustris_api.load_galaxy(id=0, verbose=True)
        """

        # Check if there is already a file with the same name
        if os.path.exists(os.path.join(self.DATAPATH, f"galaxy-id-{id}.hdf5")):
            # If file exists, check if we should overwrite it
            if not overwrite:
                # If we should not overwrite it, check if we should reuse it
                if reuse:
                    self.logger.info(
                        f"Reusing existing file galaxy-id-{id}.hdf5. If you want to download the data again, set reuse=False."
                    )
                    return self._load_hdf5(filename=f"galaxy-id-{id}")
                else:
                    # If we should not reuse it, raise an error
                    raise ValueError(
                        f"File with name galaxy-id-{id}.hdf5 already exists. Please remove it before downloading the data, or set overwrite=True, or reuse=True to load the data."
                    )
            else:
                self.logger.info(
                    f"Found existing file galaxy-id-{id}.hdf5, but overwrite is set to True. Overwriting the file."
                )

        # Check which particles we want to load

        self.logger.debug(f"Loading galaxy with ID {id}")
        url = f"{self.baseURL}/subhalos/{id}/cutout.hdf5?"

        for particle_type in self.particle_type:
            # Check if particle type is valid
            if particle_type not in self.DEFAULT_FIELDS.keys():
                raise ValueError(
                    f"Got unsupported particle type. Supported types are {self.DEFAULT_FIELDS.keys()} and we got {particle_type}."
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
                f["SubhaloData"].create_dataset(key, data=subhalo_data[key])  # type: ignore

    def __str__(self) -> str:
        return f"IllustrisAPI: Simulation {self.simulation}, Snapshot {self.snapshot}, Particle Type {self.particle_type}"
