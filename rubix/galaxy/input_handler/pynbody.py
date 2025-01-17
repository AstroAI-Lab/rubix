from .base import BaseHandler
import pynbody
import numpy as np
from rubix.utils import SFTtoAge
import logging
import astropy.units as u
import yaml
import os

Zsun = u.def_unit("Zsun", u.dimensionless_unscaled)
u.add_enabled_units(Zsun)


class PynbodyHandler(BaseHandler):
    def __init__(self, path, halo_path=None, logger=None, config=None, dist_z=None, halo_id=None):
        """Initialize handler with paths to snapshot and halo files."""
        self.path = path
        self.halo_path = halo_path
        self.halo_id = halo_id
        self.pynbody_config = config or self._load_config()
        self.logger = logger or self._default_logger()
        super().__init__()
        self.dist_z = dist_z
        self.logger.info(f"Galaxy redshift (dist_z) set to: {self.dist_z}")
        if "dm" not in self.config["particles"]:
            self.config["particles"]["dm"] = {}

        if "mass" not in self.config["particles"]["dm"]:
            self.config["particles"]["dm"]["mass"] = self.pynbody_config["units"][
                "stars"
            ]["mass"]
        self.load_data()

    def _load_config(self):
        """
        Load the PYNBODY YAML configuration.
        Check for an environment variable (RUBIX_PYNBODY_CONFIG) to specify the config path.
        If not set, fall back to the default relative path.
        """
        # Check for environment variable
        env_config_path = os.environ.get("RUBIX_PYNBODY_CONFIG", "")

        if env_config_path:
            self.logger.info(
                f"Using environment-specified config path: {env_config_path}"
            )
            config_path = env_config_path
        else:
            # Default to the relative path
            config_path = os.path.join(
                os.path.dirname(__file__), "../../config/pynbody_config.yml"
            )

        # Check if the config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"pynbody config file not found at: {config_path}. "
                "Ensure the file exists or set the RUBIX_PYNBODY_CONFIG environment variable."
            )

        # Load the YAML config
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _default_logger(self):
        """Create a default logger if none is provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def load_data(self):
        """Load data from snapshot and halo file (if available)."""
        self.sim = pynbody.load(self.path)
        self.sim.physical_units()

        halo = self.get_halo_data(halo_id=self.halo_id)
        if halo is not None:
            pynbody.analysis.angmom.faceon(halo)
            self.sim = halo

        fields = self.pynbody_config["fields"]
        load_classes = self.pynbody_config.get("load_classes", ["stars", "gas", "dm"])
        self.data = {}
        units = self.get_units()

        # Load data for stars, gas, and dark matter
        for cls in load_classes:
            if cls in ["stars", "gas", "dm"]:
                self.data[cls] = self.load_particle_data(
                    getattr(self.sim, cls), fields[cls], units[cls], cls
                )

        self.logger.info(
            f"Simulation snapshot and halo data loaded successfully for classes: {load_classes}."
        )

    def load_particle_data(self, sim_class, fields, units, particle_type):
        """
        Helper function to load particle data for a given particle class (stars/gas/dm).
        We check if each field is in the simulation's loadable keys.
        If it's missing, we log a warning and create a zero array (with correct shape & units).
        """
        data = {}
        loadable = sim_class.loadable_keys()

        for field, sim_field in fields.items():
            if sim_field in loadable:
                # For NIHAO, temperature is directly available as "temp" (if requested).
                data[field] = np.array(sim_class[sim_field]) * units.get(
                    field, u.dimensionless_unscaled
                )
            else:
                self.logger.warning(
                    f"Field '{field}' -> '{sim_field}' not found for {particle_type}. "
                    "Assigning zeros."
                )
                data[field] = np.zeros(len(sim_class)) * units.get(
                    field, u.dimensionless_unscaled
                )

        return data

    def get_halo_data(self, halo_id=None):
        """Load and return halo data if available."""
        if self.halo_path:
            halos = self.sim.halos(filename=self.halo_path)
            self.logger.info("Halo data loaded.")
            if halo_id:
                return halos[halo_id]
            else:
                return halos[0]
        else:
            self.logger.warning("No halo file provided or found.")
            return None

    def get_galaxy_data(self):
        """Return basic galaxy data."""
        if "stars" in self.data:
            positions = self.data["stars"]["coords"].value
            masses = self.data["stars"]["mass"].value
            halfmassrad_stars = self.calculate_halfmass_radius(positions, masses)
            self.logger.info(
                f"Half-mass radius calculated: {halfmassrad_stars:.2f} kpc"
            )
        else:
            halfmassrad_stars = None
            self.logger.warning(
                "No star data available to calculate the half-mass radius."
            )

        return {
            "redshift": self.dist_z,
            "center": [0, 0, 0],
            "halfmassrad_stars": halfmassrad_stars,
        }

    def get_particle_data(self):
        """Return particle data."""
        return self.data

    def get_simulation_metadata(self):
        """Return metadata for the simulation."""
        return {
            "path": self.path,
            "halo_path": self.halo_path,
            "logger": str(self.logger),
        }

    def calculate_halfmass_radius(self, positions, masses):
        """Calculates the half-mass radius based on the positions and masses of the stars."""

        if positions.ndim == 1:
            positions = positions[:, np.newaxis]
        distances = np.linalg.norm(positions, axis=1)
        sorted_indices = np.argsort(distances)
        cumulative_mass = np.cumsum(masses[sorted_indices])
        total_mass = cumulative_mass[-1]

        halfmass_index = np.searchsorted(cumulative_mass, total_mass / 2)
        halfmass_radius = distances[sorted_indices[halfmass_index]]
        return halfmass_radius

    def get_units(self):
        """
        Define and return units for all quantities based on the YAML config.
        We look up each unit string in our unit_map and store it.
        """
        unit_map = {
            "Msun": u.M_sun,
            "Gyr": u.Gyr,
            "Zsun": u.Unit("Zsun"),
            "kpc": u.kpc,
            "km/s": u.km / u.s,
            "Msun/kpc^3": u.M_sun / (u.kpc**3),
            "Msun/yr": u.M_sun / u.yr,
            "erg/g": u.erg / u.g,
            "K": u.K,
            "dimensionless": u.dimensionless_unscaled,
        }

        units_config = self.pynbody_config.get("units", {})
        converted_units = {}

        for category, fields in units_config.items():
            converted_units[category] = {}
            for field, unit_str in fields.items():
                if unit_str not in unit_map:
                    self.logger.warning(
                        f"Unit '{unit_str}' for '{category}.{field}' not recognized. "
                        "Using dimensionless."
                    )
                    converted_units[category][field] = u.dimensionless_unscaled
                else:
                    converted_units[category][field] = unit_map[unit_str]

        return converted_units
