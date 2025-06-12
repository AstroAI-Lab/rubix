import logging
import os

import astropy.units as u
import numpy as np
import pynbody
import yaml
import pickle

from rubix.cosmology import PLANCK15 as rubix_cosmo
from rubix.units import Zsun
from rubix.utils import SFTtoAge

from .base import BaseHandler


class PynbodyHandler(BaseHandler):
    def __init__(
        self, path, halo_path=None, component=None, component_file=None, logger=None, config=None, dist_z=None, halo_id=None
    ):
        """Initialize handler with paths to snapshot and halo files."""
        self.metallicity_unit = Zsun
        self.path = path
        self.halo_path = halo_path
        self.halo_id = halo_id
        self.pynbody_config = config or self._load_config()
        comp_cfg = self.pynbody_config.get("galaxy", {})
        self.component      = component
        self.component_file = component_file
        self.logger = logger or self._default_logger()
        super().__init__()
        self.dist_z = dist_z
        self.logger.info(f"Galaxy redshift (dist_z) set to: {self.dist_z}")
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

        self.logger.info(f"Simulation snapshot loaded from halo {self.halo_id}")
        halo = self.get_halo_data(halo_id=self.halo_id)
        if halo is not None:
            #pynbody.analysis.angmom.faceon(halo.s)
            pynbody.analysis.angmom.faceon(halo.s)
            ang_mom_vec = pynbody.analysis.angmom.ang_mom_vec(halo.s)
            rotation_matrix = pynbody.analysis.angmom.calc_sideon_matrix(ang_mom_vec)
            np.save("./data/rotation_matrix.npy", rotation_matrix)
            self.logger.info(
                "Rotation matrix calculated and saved to '/notebooks/data/rotation_matrix.npy'."
            )
            self.sim = halo

            self.logger.info(
                f"Loaded halo data for halo ID {self.halo_id} with {len(halo.s)} particles."
            )
            self.logger.info(
                f"Loading components from file: {self.component_file}."
            )
            self.logger.info(
                f"Filtering components by: {self.component}."
            )
            # If a component is specified, load it and store from the input handler only the particles belonging to that file.
            gsf_spliting = pickle.load(open(self.component_file, "rb"))
            tags = gsf_spliting["tags"]              # e.g. ["classicalBulge","ThinDisc",…]
            labels = gsf_spliting["label"]           # array of ints same length as iord
            gmm_iords = gsf_spliting["iord"]         # array of star iords

            # normalize config to a list (or empty list for no filter)
            if self.component is None:
                target_tags = []
            elif isinstance(self.component, str):
                target_tags = [self.component]
            else:
                target_tags = list(self.component)

            # validate
            valid = [t for t in target_tags if t in tags]
            invalid = set(target_tags) - set(valid)
            if invalid:
                self.logger.warning(
                    f"Ignoring unknown components: {sorted(invalid)}; valid tags are {tags}"
                )

            if valid:
                self.logger.info(
                    f"Filtering components by tags: {valid}."
                )
                # find all integer labels matching any requested tag
                target_idxs = [tags.index(t) for t in valid]
                # build mask: iord in any of those labels
                snap_iords = np.asarray(halo.s["iord"])
                sel_iords  = np.concatenate([gmm_iords[labels == idx] for idx in target_idxs])
                mask       = np.isin(snap_iords, sel_iords)
                halo.s     = halo.s[mask]
                self.logger.info(
                    f"Filtered to components {valid} "
                    f"({mask.sum()} particles out of {len(snap_iords)} total)."
                )
            else:
                self.logger.info("No component filtering; loading all stars.")

            self.sim = halo



        fields = self.pynbody_config["fields"]
        load_classes = self.pynbody_config.get("load_classes", ["stars", "gas"])
        self.data = {}
        units = self.get_units()

        # Load data for stars and gas
        for cls in load_classes:
            #if cls in ["stars", "gas"]:
            #    self.data[cls] = self.load_particle_data(
            #        getattr(self.sim, cls), fields[cls], units[cls], cls
            #    )
            if cls == "stars":
                # use the masked subhalo star‐Snap
                sim_class = self.sim.s
            elif cls == "gas":
                # gas remains unfiltered
                sim_class = self.sim.g   # or self.sim.gas
            else:
                continue

            self.data[cls] = self.load_particle_data(
                sim_class,
                fields[cls],
                units[cls],
                cls
            )

        # for cls in self.data:
        #    self.logger.info(f"Loaded {cls} data: {self.data[cls].keys()}")
        #    self.logger.info("Assigning metals to gas particles........")

        # Combine HI and OxMassFrac into a two-column metals field for gas
        #    self.data["gas"]["metals"] = np.column_stack((self.data["gas"]["HI"],
        #                                                self.data["gas"]["OxMassFrac"]))
        #    self.logger.info("Metals assigned to gas particles........")
        #    self.logger.info("Metals shape is: ", self.data["gas"]["metals"].shape)

        hi_data = self.load_particle_data(
            getattr(self.sim, "gas"),
            {"HI": "HI"},
            {"HI": u.dimensionless_unscaled},
            "gas",
        )
        ox_data = self.load_particle_data(
            getattr(self.sim, "gas"),
            {"OxMassFrac": "OxMassFrac"},
            {"OxMassFrac": u.dimensionless_unscaled},
            "gas",
        )
        # fe_data = self.load_particle_data(getattr(self.sim, "gas"), {"FeMassFrac": "FeMassFrac"}, {"FeMassFrac": u.dimensionless_unscaled}, "gas")
        # self.data["gas"]["metals"] = np.column_stack((hi_data["HI"], ox_data["OxMassFrac"]))
        # Create a metals array with 10 columns, filled with zeros initially
        n_particles = hi_data["HI"].shape[0]
        metals = np.zeros((n_particles, 10), dtype=hi_data["HI"].dtype)

        # Place HI values at column 0 and OxMassFrac (O) at column 4 (that it is storred in the same way as IllustrisTNG)
        metals[:, 0] = hi_data["HI"]
        metals[:, 4] = ox_data["OxMassFrac"]

        self.data["gas"]["metals"] = metals
        self.logger.info("Metals assigned to gas particles.")
        self.logger.info("Metals shape is: %s", self.data["gas"]["metals"].shape)

        age_at_z0 = rubix_cosmo.age_at_z0()
        self.data["stars"]["age"] = age_at_z0 * u.Gyr - self.data["stars"]["age"]

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
                self.logger.debug(
                    f"{len(data[field])} particles for {particle_type} loaded from '{sim_field}'"
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
