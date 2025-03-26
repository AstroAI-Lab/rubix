import time
from typing import Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import block_until_ready
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.pipeline import linear_pipeline as pipeline
from rubix.utils import get_config, get_pipeline_config

from .data import get_reshape_data, get_rubix_data
from .dust import get_extinction
from .ifu import (
    get_calculate_datacube,
    get_calculate_spectra,
    get_doppler_shift_and_resampling,
    get_scale_spectrum_by_mass,
)
from .lsf import get_convolve_lsf
from .noise import get_apply_noise
from .psf import get_convolve_psf
from .rotation import get_galaxy_rotation
from .ssp import get_ssp
from .telescope import get_filter_particles, get_spaxel_assignment, get_telescope


class RubixPipeline:
    """
    RubixPipeline is responsible for setting up and running the data processing pipeline.

    Args:
        user_config (dict or str): Parsed user configuration for the pipeline.
        pipeline_config (dict): Configuration for the pipeline.
        logger(Logger) : Logger instance for logging messages.
        ssp(object) : Stellar population synthesis model.
        telescope(object) : Telescope configuration.
        #data (dict): Dictionary containing particle data.
        func (callable): Compiled pipeline function to process data.

    Example
    --------
    >>> from rubix.core.pipeline import RubixPipeline
    >>> config = "path/to/config.yml"
    >>> inputdata = pipe.prepare_data()
    >>> rubixdata = pipe.run(inputdata)
    >>> ssp_model = pipeline.ssp
    >>> telescope = pipeline.telescope
    """

    def __init__(self, user_config: Union[dict, str]):
        self.user_config = get_config(user_config)
        self.pipeline_config = get_pipeline_config(self.user_config["pipeline"]["name"])
        self.logger = get_logger(self.user_config["logger"])
        self.ssp = get_ssp(self.user_config)
        self.telescope = get_telescope(self.user_config)
        # self.data = self._prepare_data()
        self.func = None

    def prepare_data(self):
        """
        Prepares and loads the data for the pipeline.

        Returns:
            Dictionary containing particle data with keys:
            'n_particles', 'coords', 'velocities', 'metallicity', 'mass', and 'age'.
        """
        # Get the data
        self.logger.info("Getting rubix data...")
        rubixdata = get_rubix_data(self.user_config)
        star_count = (
            len(rubixdata.stars.coords) if rubixdata.stars.coords is not None else 0
        )
        gas_count = len(rubixdata.gas.coords) if rubixdata.gas.coords is not None else 0
        self.logger.info(
            f"Data loaded with {star_count} star particles and {gas_count} gas particles."
        )
        # Setup the data dictionary
        # TODO: This is a temporary solution, we need to figure out a better way to handle the data
        # This works, because JAX can trace through the data dictionary
        # Other option may be named tuples or data classes to have fixed keys

        # self.logger.debug("Data: %s", rubixdata)
        # self.logger.debug(
        #    "Data Shape: %s",
        #    {k: v.shape for k, v in rubixdata.items() if hasattr(v, "shape")},
        # )

        return rubixdata

    @jaxtyped(typechecker=typechecker)
    def _get_pipeline_functions(self) -> list:
        """
        Sets up the pipeline functions.

        Returns:
            List of functions to be used in the pipeline.
        """
        self.logger.info("Setting up the pipeline...")
        self.logger.debug("Pipeline Configuration: %s", self.pipeline_config)

        # TODO: maybe there is a nicer way to load the functions from the yaml config?
        rotate_galaxy = get_galaxy_rotation(self.user_config)
        filter_particles = get_filter_particles(self.user_config)
        spaxel_assignment = get_spaxel_assignment(self.user_config)
        calculate_spectra = get_calculate_spectra(self.user_config)
        reshape_data = get_reshape_data(self.user_config)
        scale_spectrum_by_mass = get_scale_spectrum_by_mass(self.user_config)
        doppler_shift_and_resampling = get_doppler_shift_and_resampling(
            self.user_config
        )
        apply_extinction = get_extinction(self.user_config)
        calculate_datacube = get_calculate_datacube(self.user_config)
        convolve_psf = get_convolve_psf(self.user_config)
        convolve_lsf = get_convolve_lsf(self.user_config)
        apply_noise = get_apply_noise(self.user_config)

        functions = [
            rotate_galaxy,
            filter_particles,
            spaxel_assignment,
            calculate_spectra,
            reshape_data,
            scale_spectrum_by_mass,
            doppler_shift_and_resampling,
            apply_extinction,
            calculate_datacube,
            convolve_psf,
            convolve_lsf,
            apply_noise,
        ]

        return functions

    # TODO: currently returns dict, but later should return only the IFU cube
    def run(self, inputdata):
        """
        Runs the data processing pipeline.

        Parameters
        ----------
        input_data : dict
            Data prepared from the `prepare_data` method.

        Returns
        -------
        dict
            Output of the pipeline after processing the input data.
        """
        # Create the pipeline
        time_start = time.time()
        functions = self._get_pipeline_functions()
        self._pipeline = pipeline.LinearTransformerPipeline(
            self.pipeline_config, functions
        )

        # Assembling the pipeline
        self.logger.info("Assembling the pipeline...")
        self._pipeline.assemble()

        # Compiling the expressions
        self.logger.info("Compiling the expressions...")
        self.func = self._pipeline.compile_expression()

        # Running the pipeline
        self.logger.info("Running the pipeline on the input data...")
        output = self.func(inputdata)

        block_until_ready(output)
        time_end = time.time()
        self.logger.info(
            "Pipeline run completed in %.2f seconds.", time_end - time_start
        )

        output.galaxy.redshift_unit = inputdata.galaxy.redshift_unit
        output.galaxy.center_unit = inputdata.galaxy.center_unit
        output.galaxy.halfmassrad_stars_unit = inputdata.galaxy.halfmassrad_stars_unit

        if output.stars.coords != None:
            output.stars.coords_unit = inputdata.stars.coords_unit
            output.stars.velocity_unit = inputdata.stars.velocity_unit
            output.stars.mass_unit = inputdata.stars.mass_unit
            # output.stars.metallictiy_unit = self.data.stars.metallictiy_unit
            output.stars.age_unit = inputdata.stars.age_unit
            output.stars.spatial_bin_edges_unit = "kpc"
            # output.stars.wavelength_unit = rubix_config["ssp"]["units"]["wavelength"]
            # output.stars.spectra_unit = rubix_config["ssp"]["units"]["flux"]
            # output.stars.datacube_unit = rubix_config["ssp"]["units"]["flux"]

        if output.gas.coords != None:
            output.gas.coords_unit = inputdata.gas.coords_unit
            output.gas.velocity_unit = inputdata.gas.velocity_unit
            output.gas.mass_unit = inputdata.gas.mass_unit
            output.gas.density_unit = inputdata.gas.density_unit
            output.gas.internal_energy_unit = inputdata.gas.internal_energy_unit
            # output.gas.metallicity_unit = self.data.gas.metallicity_unit
            output.gas.sfr_unit = inputdata.gas.sfr_unit
            output.gas.electron_abundance_unit = inputdata.gas.electron_abundance_unit
            output.gas.spatial_bin_edges_unit = "kpc"
            # output.gas.wavelength_unit = rubix_config["ssp"]["units"]["wavelength"]
            # output.gas.spectra_unit = rubix_config["ssp"]["units"]["flux"]
            # output.gas.datacube_unit = rubix_config["ssp"]["units"]["flux"]

        return output

    # TODO: implement gradient calculation
    def gradient(self):
        """
        This function will calculate the gradient of the pipeline, but is yet not implemented.
        """
        raise NotImplementedError("Gradient calculation is not implemented yet")
