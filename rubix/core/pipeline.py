import time
from types import SimpleNamespace
from typing import Union

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import dataclasses

# For shard_map and device mesh.
import numpy as np
from beartype import beartype as typechecker
from jax import block_until_ready
from jax.experimental import shard_map
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import jaxtyped
from functools import partial
from jax import lax
from jax.experimental.pjit import pjit

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
from .data import RubixData, Galaxy, StarsData, GasData


class RubixPipeline:
    """
    RubixPipeline is responsible for setting up and running the data processing pipeline.

    Usage
    -----
    >>> pipe = RubixPipeline(config)
    >>> inputdata = pipe.prepare_data()
    >>> # To run without sharding:
    >>> output = pipe.run(inputdata)
    >>> # To run with sharding using jax.shard_map:
    >>> final_datacube = pipe.run_sharded(inputdata, shard_size=100000)
    """

    def __init__(self, user_config: Union[dict, str]):
        self.user_config = get_config(user_config)
        self.pipeline_config = get_pipeline_config(self.user_config["pipeline"]["name"])
        self.logger = get_logger(self.user_config["logger"])
        self.ssp = get_ssp(self.user_config)
        self.telescope = get_telescope(self.user_config)
        self.func = None

    def prepare_data(self):
        """
        Prepares and loads the data for the pipeline.

        Returns:
            Object containing particle data with attributes such as:
            'coords', 'velocities', 'mass', 'age', and 'metallicity' under stars and gas.
        """
        self.logger.info("Getting rubix data...")
        rubixdata = get_rubix_data(self.user_config)
        star_count = (
            len(rubixdata.stars.coords) if rubixdata.stars.coords is not None else 0
        )
        gas_count = len(rubixdata.gas.coords) if rubixdata.gas.coords is not None else 0
        self.logger.info(
            f"Data loaded with {star_count} star particles and {gas_count} gas particles."
        )
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

        rotate_galaxy = get_galaxy_rotation(self.user_config)
        filter_particles = get_filter_particles(self.user_config)
        spaxel_assignment = get_spaxel_assignment(self.user_config)
        calculate_spectra = get_calculate_spectra(self.user_config)
        #reshape_data = get_reshape_data(self.user_config)
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
            #reshape_data,
            scale_spectrum_by_mass,
            doppler_shift_and_resampling,
            apply_extinction,
            calculate_datacube,
            convolve_psf,
            convolve_lsf,
            apply_noise,
        ]
        return functions

    def run(self, inputdata):
        """
        Runs the data processing pipeline on the complete input data.

        Parameters
        ----------
        inputdata : object
            Data prepared from the `prepare_data` method.

        Returns
        -------
        object
            Pipeline output (which includes the datacube and unit attributes).
        """
        time_start = time.time()
        functions = self._get_pipeline_functions()
        self._pipeline = pipeline.LinearTransformerPipeline(
            self.pipeline_config, functions
        )
        self.logger.info("Assembling the pipeline...")
        self._pipeline.assemble()
        self.logger.info("Compiling the expressions...")
        self.func = self._pipeline.compile_expression()
        self.logger.info("Running the pipeline on the input data...")
        output = self.func(inputdata)
        block_until_ready(output)
        time_end = time.time()
        self.logger.info(
            "Pipeline run completed in %.2f seconds.", time_end - time_start
        )

        # Propagate unit attributes from input to output.
        output.galaxy.redshift_unit = inputdata.galaxy.redshift_unit
        output.galaxy.center_unit = inputdata.galaxy.center_unit
        output.galaxy.halfmassrad_stars_unit = inputdata.galaxy.halfmassrad_stars_unit

        if output.stars.coords is not None:
            output.stars.coords_unit = inputdata.stars.coords_unit
            output.stars.velocity_unit = inputdata.stars.velocity_unit
            output.stars.mass_unit = inputdata.stars.mass_unit
            output.stars.age_unit = inputdata.stars.age_unit
            output.stars.spatial_bin_edges_unit = "kpc"

        if output.gas.coords is not None:
            output.gas.coords_unit = inputdata.gas.coords_unit
            output.gas.velocity_unit = inputdata.gas.velocity_unit
            output.gas.mass_unit = inputdata.gas.mass_unit
            output.gas.density_unit = inputdata.gas.density_unit
            output.gas.internal_energy_unit = inputdata.gas.internal_energy_unit
            output.gas.sfr_unit = inputdata.gas.sfr_unit
            output.gas.electron_abundance_unit = inputdata.gas.electron_abundance_unit
            output.gas.spatial_bin_edges_unit = "kpc"

        return output

    def run_sharded(self, inputdata):
        """
        Runs the pipeline on sharded input data in parallel using jax.shard_map.
        It splits the particle arrays (e.g. under stars and gas) into shards, runs
        the compiled pipeline on each shard, and then combines the resulting datacubes.

        Parameters
        ----------
        inputdata : object
            Data prepared from the `prepare_data` method.
        shard_size : int
            Number of particles per shard.

        Returns
        -------
        jax.numpy.ndarray
            The final datacube combined from all shards.
        """
        #time_start = time.time()
        # Assemble and compile the pipeline as before.
        functions = self._get_pipeline_functions()
        self._pipeline = pipeline.LinearTransformerPipeline(
            self.pipeline_config, functions
        )
        self.logger.info("Assembling the pipeline...")
        self._pipeline.assemble()
        self.logger.info("Compiling the expressions...")
        self.func = self._pipeline.compile_expression()

        devices = jax.devices()
        num_devices = len(devices)
        self.logger.info("Number of devices: %d", num_devices)

        mesh = Mesh(devices, ("data",))

        # — sharding specs by rank —
        replicate_0d   = NamedSharding(mesh, P())               # for scalars
        replicate_1d   = NamedSharding(mesh, P(None))           # for 1-D arrays
        shard_2d       = NamedSharding(mesh, P("data", None))   # for (N, D)
        replicate_3d   = NamedSharding(mesh, P(None, None, None)) # for full cube

        # — 1) allocate empty instances —
        galaxy_spec = object.__new__(Galaxy)
        stars_spec  = object.__new__(StarsData)
        gas_spec    = object.__new__(GasData)
        rubix_spec  = object.__new__(RubixData)

        # — 2) assign NamedSharding to each field —
        # galaxy
        galaxy_spec.redshift          = replicate_0d
        galaxy_spec.center            = replicate_1d
        galaxy_spec.halfmassrad_stars = replicate_0d

        # stars
        stars_spec.coords             = shard_2d
        stars_spec.velocity           = shard_2d
        stars_spec.mass               = replicate_1d
        stars_spec.age                = replicate_1d
        stars_spec.metallicity        = replicate_1d
        stars_spec.pixel_assignment   = replicate_1d
        stars_spec.spatial_bin_edges  = NamedSharding(mesh, P(None, None))
        stars_spec.mask               = replicate_1d
        stars_spec.spectra            = shard_2d
        stars_spec.datacube           = replicate_3d

        # gas  (same idea)
        gas_spec.coords               = shard_2d
        gas_spec.velocity             = shard_2d
        gas_spec.mass                 = replicate_1d
        gas_spec.density              = replicate_1d
        gas_spec.internal_energy      = replicate_1d
        gas_spec.metallicity          = replicate_1d
        gas_spec.metals               = replicate_1d
        gas_spec.sfr                  = replicate_1d
        gas_spec.electron_abundance   = replicate_1d
        gas_spec.pixel_assignment     = replicate_1d
        gas_spec.spatial_bin_edges    = NamedSharding(mesh, P(None, None))
        gas_spec.mask                 = replicate_1d
        gas_spec.spectra              = shard_2d
        gas_spec.datacube             = replicate_3d

        # — link them up —
        rubix_spec.galaxy = galaxy_spec
        rubix_spec.stars  = stars_spec
        rubix_spec.gas    = gas_spec

        
        @partial(jax.jit,
        #how inputs ARE sharded when the function is called
        in_shardings  = (rubix_spec,),
        out_shardings = replicate_3d,
        )
        def shard_pipeline(sharded_rubixdata):
            out_local = self.func(sharded_rubixdata)
            # locally computed partial cube
            local_cube = out_local.stars.datacube  
            # reduce across devices
            return local_cube

        with mesh:
            # `shard_pipeline` returns a GDA with shape (num_devices, 25,25,5994)
            partial_cubes = shard_pipeline(inputdata)
        # now in host‐land you can simply
        #full_cube = jnp.sum(partial_cubes, axis=0)

        return partial_cubes
        
        """
        def _shard_pipeline(sharded_rubixdata):
            out_local  = self.func(sharded_rubixdata)
            local_cube = out_local.stars.datacube
            # this requires that you actually be in a mesh context with an axis_name="data"
            full_cube  = lax.psum(local_cube, axis_name="data")
            return full_cube

        # compile it
        shard_pipeline = pjit(
            _shard_pipeline,                   # <— the function
            in_shardings  = (rubix_spec,),
            out_shardings = (replicate_3d,),
        )

        # then inside your mesh:
        with mesh:
            final_datacube = shard_pipeline(inputdata)
        
        return final_datacube
        """

    def gradient(self):
        """
        This function will calculate the gradient of the pipeline, but is not implemented.
        """
        raise NotImplementedError("Gradient calculation is not implemented yet")
