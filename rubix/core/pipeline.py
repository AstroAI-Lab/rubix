import dataclasses
import time
from functools import partial
from types import SimpleNamespace
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# For shard_map and device mesh.
import numpy as np
from beartype import beartype as typechecker
from jax import block_until_ready, lax
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.pipeline import linear_pipeline as pipeline
from rubix.utils import get_config, get_pipeline_config

from .data import (
    Galaxy,
    GasData,
    RubixData,
    StarsData,
    _pad_particles_equinox,
    get_rubix_data,
)
from .dust import get_extinction
from .ifu import get_calculate_datacube_optimized, get_calculate_datacube_vectorized
from .lsf import get_convolve_lsf
from .noise import get_apply_noise
from .psf import get_convolve_psf
from .rotation import get_galaxy_rotation
from .ssp import get_ssp
from .telescope import get_filter_particles, get_spaxel_assignment, get_telescope


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
        t1 = time.time()
        self.logger.info("Getting rubix data...")
        rubixdata = get_rubix_data(self.user_config)

        # Safely get star count
        star_count = (
            len(rubixdata.stars.coords) if rubixdata.stars.coords is not None else 0
        )

        # Safely get gas count - check if gas exists first
        gas_count = 0
        if rubixdata.gas is not None and rubixdata.gas.coords is not None:
            gas_count = len(rubixdata.gas.coords)

        self.logger.info(
            f"Data loaded with {star_count} star particles and {gas_count} gas particles."
        )
        t2 = time.time()
        self.logger.info("Data preparation completed in %.2f seconds.", t2 - t1)
        return rubixdata

    @jaxtyped(typechecker=typechecker)
    def _get_pipeline_functions(self) -> list:
        """
        Sets up the pipeline functions that all work with immutable Equinox data.

        Returns:
            List of functions to be used in the pipeline.
        """
        self.logger.info("Setting up the pipeline...")
        self.logger.debug("Pipeline Configuration: %s", self.pipeline_config)

        rotate_galaxy = get_galaxy_rotation(self.user_config)
        filter_particles = get_filter_particles(self.user_config)
        spaxel_assignment = get_spaxel_assignment(self.user_config)
        # reshape_data = get_reshape_data(self.user_config)
        apply_extinction = get_extinction(self.user_config)

        # Use vectorized datacube calculation for better GPU performance
        calculate_datacube = get_calculate_datacube_vectorized(self.user_config)
        # calculate_datacube = get_calculate_datacube_optimized(self.user_config)

        convolve_psf = get_convolve_psf(self.user_config)
        convolve_lsf = get_convolve_lsf(self.user_config)
        apply_noise = get_apply_noise(self.user_config)

        functions = [
            rotate_galaxy,
            filter_particles,
            spaxel_assignment,
            # reshape_data,
            apply_extinction,
            calculate_datacube,  # Now using vectorized version
            convolve_psf,
            convolve_lsf,
            apply_noise,
        ]
        return functions

    def run_sharded(self, inputdata):
        """
        Runs the pipeline on sharded input data using Equinox modules.
        This method uses JAX's shard_map to distribute the computation across multiple devices.
        Args:
            inputdata (RubixData): The input data containing particle information.
        Returns:
            Sharded datacube result after processing through the pipeline.
        Raises:
            ValueError: If the input data is not of type RubixData.
        """
        time_start = time.time()

        # Check if we have enough particles for efficient sharding
        if inputdata.stars.coords is not None:
            n_particles = inputdata.stars.coords.shape[0]
            devices = jax.devices()
            num_devices = len(devices)

            # Rule of thumb: need at least 10 particles per device for efficient sharding
            min_particles_per_device = 10
            if n_particles < num_devices * min_particles_per_device:
                self.logger.warning(
                    f"Only {n_particles} particles for {num_devices} devices "
                    f"(minimum recommended: {num_devices * min_particles_per_device}). "
                    f"Falling back to non-sharded execution for better efficiency."
                )
                return self.run(inputdata)

        # Continue with sharded execution
        # Assemble and compile the pipeline
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

        mesh = Mesh(devices, axis_names=("data",))

        # Create sharding specs using JAX tree_map
        def create_sharding_for_array(arr):
            """Create appropriate sharding based on array shape"""
            if arr is None:
                return None

            if arr.ndim == 0:  # scalar
                return NamedSharding(mesh, P())
            elif arr.ndim == 1:  # 1D array
                return NamedSharding(mesh, P("data"))
            elif arr.ndim == 2:  # 2D array
                return NamedSharding(mesh, P("data", None))
            elif arr.ndim == 3:  # 3D array (datacube)
                return NamedSharding(mesh, P(None, None, None))
            else:
                return NamedSharding(mesh, P("data"))

        # Create sharding specification using JAX tree_map
        sharding_spec = jtu.tree_map(
            create_sharding_for_array, inputdata, is_leaf=lambda x: x is None
        )

        # Extract PartitionSpec for shard_map
        partition_spec_tree = jtu.tree_map(
            lambda s: s.spec if isinstance(s, NamedSharding) else None,
            sharding_spec,
            is_leaf=lambda x: x is None,
        )

        # Pad particles if needed - check for star coords existence
        if inputdata.stars.coords is not None:
            n = inputdata.stars.coords.shape[0]
            pad = (num_devices - (n % num_devices)) % num_devices
            if pad:
                self.logger.info("Padding particles for %d devices...", num_devices)
                inputdata = _pad_particles_equinox(inputdata, pad)  # Use new function

        # Place data on devices
        inputdata = jax.device_put(inputdata, sharding_spec)

        # Create sharded pipeline
        def _shard_pipeline(sharded_rubixdata):
            out_local = self.func(sharded_rubixdata)
            local_cube = out_local.stars.datacube
            # All-reduce across the "data" axis
            summed_cube = lax.psum(local_cube, axis_name="data")
            return summed_cube

        sharded_pipeline = shard_map(
            _shard_pipeline,
            mesh=mesh,
            in_specs=(partition_spec_tree,),
            out_specs=NamedSharding(mesh, P(None, None, None)).spec,
            check_rep=False,
        )

        time_mid = time.time()
        sharded_result = sharded_pipeline(inputdata)

        time_end = time.time()
        self.logger.info("Sharding completed in %.2f seconds.", time_mid - time_start)
        self.logger.info(
            "Sharded pipeline run completed in %.2f seconds.", time_end - time_mid
        )

        return sharded_result

    @jaxtyped(typechecker=typechecker)
    def run(self, inputdata):
        """
        Runs the pipeline on input data without sharding (single device execution).
        This is more efficient for small datasets or when you want to use a single device.

        Args:
            inputdata (RubixData): The input data containing particle information.
        Returns:
            RubixData: Complete processed data after running through the pipeline.
        """
        time_start = time.time()

        # Check if we have data
        if inputdata.stars.coords is not None:
            n_particles = inputdata.stars.coords.shape[0]
            self.logger.info(f"Running non-sharded pipeline on {n_particles} particles")
        else:
            self.logger.warning("No star particles found in input data")
            return inputdata

        # Assemble and compile the pipeline (same as sharded version)
        functions = self._get_pipeline_functions()
        self._pipeline = pipeline.LinearTransformerPipeline(
            self.pipeline_config, functions
        )
        self.logger.info("Assembling the pipeline...")
        self._pipeline.assemble()
        self.logger.info("Compiling the expressions...")
        self.func = self._pipeline.compile_expression()

        # Move data to device (GPU if available)
        inputdata = jax.device_put(inputdata)

        time_mid = time.time()

        # Run the pipeline directly (no sharding)
        result = self.func(inputdata)

        time_end = time.time()
        self.logger.info(
            "Pipeline setup completed in %.2f seconds.", time_mid - time_start
        )
        self.logger.info(
            "Pipeline execution completed in %.2f seconds.", time_end - time_mid
        )

        return result
