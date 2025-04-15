import jax
import jax.numpy as jnp

from rubix import config
from rubix.galaxy.input_handler.base import create_rubix_galaxy
from rubix.logger import get_logger
from rubix.spectra.ssp.factory import get_ssp_template


def random_data(n_particles, min_val, max_val, dimension, key=42):
    key = jax.random.PRNGKey(key)
    if dimension == 1:
        return jax.random.uniform(key, (n_particles,), minval=min_val, maxval=max_val)
    else:
        return jax.random.uniform(
            key, (n_particles, dimension), minval=min_val, maxval=max_val
        )


def create_dummy_rubix(n_particles, output_path):
    ssp = get_ssp_template("BruzualCharlot2003")

    # get bounds of the metallicity and age bins
    metallicity_bins = ssp.metallicity
    age_bins = ssp.age

    metallicity_min, metallicity_max = metallicity_bins.min(), metallicity_bins.max()
    age_min, age_max = age_bins.min(), age_bins.max()

    particle_data = {}

    particle_data["stars"] = {}
    particle_data["stars"]["coords"] = random_data(n_particles, -100, 100, 3)
    particle_data["stars"]["velocity"] = random_data(n_particles, -100, 100, 3)
    particle_data["stars"]["metallicity"] = random_data(
        n_particles, metallicity_min, metallicity_max, 1
    )
    particle_data["stars"]["mass"] = random_data(n_particles, 1, 1, 1)
    particle_data["stars"]["age"] = random_data(n_particles, age_min, age_max, 1)

    galaxy_data = {}
    galaxy_data["center"] = jnp.array([0, 0, 0])
    galaxy_data["redshift"] = 0
    galaxy_data["halfmassrad_stars"] = 10

    simulation_metadata = {}
    simulation_metadata["Time"] = 0
    simulation_metadata["HubbleParam"] = 0.7

    # Dummy units
    units = {
        "stars": {
            "coords": "kpc",
            "mass": "Msun",
            "metallicity": "",
            "velocity": "kpc/s",
            "age": "Gyr",
        },
        "galaxy": {"center": "kpc", "halfmassrad_stars": "kpc", "redshift": ""},
    }
    logger = get_logger()
    logger.debug(f"UNITS: {units}")
    create_rubix_galaxy(
        file_path=output_path,
        particle_data=particle_data,
        galaxy_data=galaxy_data,
        simulation_metadata=simulation_metadata,
        units=units,
        config=config["BaseHandler"],
        logger=logger,
    )
