import dataclasses
import time
from functools import partial
from types import SimpleNamespace
from typing import Union

import jax
import jax.numpy as jnp

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
    get_reshape_data,
    get_rubix_data,
)
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
        # reshape_data = get_reshape_data(self.user_config)
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
            # reshape_data,
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

        This is the recomended method to run the pipeline in parallel at the moment!!!

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
        time_start = time.time()
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

        mesh = Mesh(devices, axis_names=("data",))

        # — sharding specs by rank —
        replicate_0d = NamedSharding(mesh, P())  # for scalars
        replicate_1d = NamedSharding(mesh, P(None))  # for 1-D arrays
        shard_2d = NamedSharding(mesh, P("data", None))  # for (N, D)
        shard_1d = NamedSharding(mesh, P("data"))  # for (N,)
        replicate_3d = NamedSharding(mesh, P(None, None, None))  # for full cube

        # — 1) allocate empty instances —
        galaxy_spec = object.__new__(Galaxy)
        stars_spec = object.__new__(StarsData)
        gas_spec = object.__new__(GasData)
        rubix_spec = object.__new__(RubixData)

        # — 2) assign NamedSharding to each field —
        # galaxy
        galaxy_spec.redshift = replicate_0d
        galaxy_spec.center = replicate_1d
        galaxy_spec.halfmassrad_stars = replicate_0d

        # stars
        stars_spec.coords = shard_2d
        stars_spec.velocity = shard_2d
        stars_spec.mass = shard_1d
        stars_spec.age = shard_1d
        stars_spec.metallicity = shard_1d
        stars_spec.pixel_assignment = shard_1d
        stars_spec.spatial_bin_edges = NamedSharding(mesh, P(None, None))
        stars_spec.mask = shard_1d
        stars_spec.spectra = shard_2d
        stars_spec.datacube = replicate_3d

        # gas  (same idea)
        gas_spec.coords = shard_2d
        gas_spec.velocity = shard_2d
        gas_spec.mass = shard_1d
        gas_spec.density = shard_1d
        gas_spec.internal_energy = shard_1d
        gas_spec.metallicity = shard_1d
        gas_spec.metals = shard_1d
        gas_spec.sfr = shard_1d
        gas_spec.electron_abundance = shard_1d
        gas_spec.pixel_assignment = shard_1d
        gas_spec.spatial_bin_edges = NamedSharding(mesh, P(None, None))
        gas_spec.mask = shard_1d
        gas_spec.spectra = shard_2d
        gas_spec.datacube = replicate_3d

        # — link them up —
        rubix_spec.galaxy = galaxy_spec
        rubix_spec.stars = stars_spec
        rubix_spec.gas = gas_spec

        # 1) Make a pytree of PartitionSpec
        partition_spec_tree = tree_map(
            lambda s: s.spec if isinstance(s, NamedSharding) else None, rubix_spec
        )

        # if the particle number is not modulo the device number, we have to padd a few empty particles
        # to make it work
        # this is a bit of a hack, but it works
        n = inputdata.stars.coords.shape[0]
        pad = (num_devices - (n % num_devices)) % num_devices

        if pad:
            # pad along the first axis
            inputdata.stars.coords = jnp.pad(inputdata.stars.coords, ((0, pad), (0, 0)))
            inputdata.stars.velocity = jnp.pad(
                inputdata.stars.velocity, ((0, pad), (0, 0))
            )
            inputdata.stars.mass = jnp.pad(inputdata.stars.mass, ((0, pad)))
            inputdata.stars.age = jnp.pad(inputdata.stars.age, ((0, pad)))
            inputdata.stars.metallicity = jnp.pad(
                inputdata.stars.metallicity, ((0, pad))
            )

        inputdata = jax.device_put(inputdata, rubix_spec)

        # create the sharded data
        def _shard_pipeline(sharded_rubixdata):
            out_local = self.func(sharded_rubixdata)
            local_cube = out_local.stars.datacube  # shape (25,25,5994)
            # in‐XLA all‐reduce across the "data" axis:
            summed_cube = lax.psum(local_cube, axis_name="data")
            return summed_cube  # replicated on each device

        sharded_pipeline = shard_map(
            _shard_pipeline,  # the function to compile
            mesh=mesh,  # the mesh to use
            in_specs=(partition_spec_tree,),
            out_specs=replicate_3d.spec,
            check_rep=False,
        )

        # with mesh:
        #    inputdata = jax.device_put(inputdata, rubix_spec)
        # partial_cubes = shard_pipeline(inputdata)
        # full_cube = lax.psum(partial_cubes, axis_name="data")
        # partial_cubes = jax.block_until_ready(partial_cubes)
        # full_cube = jax.block_until_ready(full_cube)

        # full_cube = partial_cubes.sum(axis=0)

        sharded_result = sharded_pipeline(inputdata)
        
        jax.block_until_ready(sharded_result)
        time_end = time.time()
        self.logger.info(
            "Pipeline run completed in %.2f seconds.", time_end - time_start
        )
        # final_cube = jnp.sum(partial_cubes, axis=0)

        return sharded_result

    def run_sharded_chunked(self, inputdata):
        """
        Runs the pipeline on sharded input data in parallel using jax.shard_map.
        It splits the particle arrays (e.g. under stars and gas) into shards, runs
        the compiled pipeline on each shard, and then combines the resulting datacubes.

        This is an experimental function and is not recommended to use at the moment!!!

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
        time_start = time.time()
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
        replicate_0d = NamedSharding(mesh, P())  # for scalars
        replicate_1d = NamedSharding(mesh, P(None))  # for 1-D arrays
        shard_2d = NamedSharding(mesh, P("data", None))  # for (N, D)
        shard_1d = NamedSharding(mesh, P("data"))  # for (N,)
        replicate_3d = NamedSharding(mesh, P(None, None, None))  # for full cube

        # — 1) allocate empty instances —
        galaxy_spec = object.__new__(Galaxy)
        stars_spec = object.__new__(StarsData)
        gas_spec = object.__new__(GasData)
        rubix_spec = object.__new__(RubixData)

        # — 2) assign NamedSharding to each field —
        # galaxy
        galaxy_spec.redshift = replicate_0d
        galaxy_spec.center = replicate_1d
        galaxy_spec.halfmassrad_stars = replicate_0d

        # stars
        stars_spec.coords = shard_2d
        stars_spec.velocity = shard_2d
        stars_spec.mass = shard_1d
        stars_spec.age = shard_1d
        stars_spec.metallicity = shard_1d
        stars_spec.pixel_assignment = shard_1d
        stars_spec.spatial_bin_edges = NamedSharding(mesh, P(None, None))
        stars_spec.mask = shard_1d
        stars_spec.spectra = shard_2d
        stars_spec.datacube = replicate_3d

        # gas  (same idea)
        gas_spec.coords = shard_2d
        gas_spec.velocity = shard_2d
        gas_spec.mass = shard_1d
        gas_spec.density = shard_1d
        gas_spec.internal_energy = shard_1d
        gas_spec.metallicity = shard_1d
        gas_spec.metals = shard_1d
        gas_spec.sfr = shard_1d
        gas_spec.electron_abundance = shard_1d
        gas_spec.pixel_assignment = shard_1d
        gas_spec.spatial_bin_edges = NamedSharding(mesh, P(None, None))
        gas_spec.mask = shard_1d
        gas_spec.spectra = shard_2d
        gas_spec.datacube = replicate_3d

        # — link them up —
        rubix_spec.galaxy = galaxy_spec
        rubix_spec.stars = stars_spec
        rubix_spec.gas = gas_spec

        # 1) Make a pytree of PartitionSpec
        partition_spec_tree = tree_map(
            lambda s: s.spec if isinstance(s, NamedSharding) else None, rubix_spec
        )

        # if the particle number is not modulo the device number, we have to padd a few empty particles
        # to make it work
        # this is a bit of a hack, but it works
        telescope = get_telescope(self.user_config)
        num_spaxels = int(telescope.sbin)
        n_wave = int(telescope.wave_seq.shape[0])
        n_stars = int(inputdata.stars.coords.shape[0])
        chunk_size = 1000 * num_devices
        n_chunks = (n_stars + chunk_size - 1) // chunk_size
        total_len = n_chunks * chunk_size

        pad_amt = total_len - n_stars

        n = inputdata.stars.coords.shape[0]
        pad = (num_devices - (n % num_devices)) % num_devices + pad_amt

        if pad:
            # pad along the first axis
            inputdata.stars.coords = jnp.pad(inputdata.stars.coords, ((0, pad), (0, 0)))
            inputdata.stars.velocity = jnp.pad(
                inputdata.stars.velocity, ((0, pad), (0, 0))
            )
            inputdata.stars.mass = jnp.pad(inputdata.stars.mass, ((0, pad)))
            inputdata.stars.age = jnp.pad(inputdata.stars.age, ((0, pad)))
            inputdata.stars.metallicity = jnp.pad(
                inputdata.stars.metallicity, ((0, pad))
            )

        """
        # Precompute all static sizes on the host
        telescope = get_telescope(self.user_config)
        num_spaxels = int(telescope.sbin)
        n_wave = int(telescope.wave_seq.shape[0])
        n_stars = int(inputdata.stars.coords.shape[0])
        chunk_size = 1000 * num_devices
        n_chunks = (n_stars + chunk_size - 1) // chunk_size
        total_len = n_chunks * chunk_size

        pad_amt = total_len - n_stars
        if pad_amt:
            pad_width_2d = ((0, pad_amt), (0, 0))
            pad_width_1d = ((0, pad_amt),)
            inputdata.stars.coords = jnp.pad(inputdata.stars.coords, pad_width_2d)
            inputdata.stars.velocity = jnp.pad(inputdata.stars.velocity, pad_width_2d)
            inputdata.stars.mass = jnp.pad(inputdata.stars.mass, pad_width_1d)
            inputdata.stars.age = jnp.pad(inputdata.stars.age, pad_width_1d)
            inputdata.stars.metallicity = jnp.pad(inputdata.stars.metallicity, pad_width_1d)
        """

        # Helper to slice RubixData along axis 0
        def slice_data(rubixdata, start):
            def slicer(x):
                if isinstance(x, jax.Array) and x.shape and x.shape[0] == total_len:
                    return lax.dynamic_slice_in_dim(x, start, chunk_size, axis=0)
                else:
                    return x

            return jax.tree_util.tree_map(slicer, rubixdata)

        inputdata = jax.device_put(inputdata, rubix_spec)

        # create the sharded data
        def _shard_pipeline(sharded_rubixdata):
            out_local = self.func(sharded_rubixdata)
            local_cube = out_local.stars.datacube  # shape (25,25,5994)
            # in‐XLA all‐reduce across the "data" axis:
            summed_cube = lax.psum(local_cube, axis_name="data")
            return summed_cube  # replicated on each device

        sharded_pipeline = shard_map(
            _shard_pipeline,  # the function to compile
            mesh=mesh,  # the mesh to use
            in_specs=(partition_spec_tree,),
            out_specs=replicate_3d.spec,
            check_rep=False,
        )

        full_cube = jnp.zeros((num_spaxels, num_spaxels, n_wave), jnp.float32)
        for i in range(n_chunks):  # Process 4 chunks
            # print(f"Processing chunk {i + 1}/{n_chunks}...")
            start = i * (n_stars // n_chunks)
            chunk_data = slice_data(inputdata, start)
            partial_cube = sharded_pipeline(chunk_data)
            full_cube += partial_cube

        full_cube = jax.block_until_ready(full_cube)

        time_end = time.time()
        self.logger.info(
            "Pipeline run completed in %.2f seconds.", time_end - time_start
        )

        return full_cube

    def gradient(self):
        """
        This function will calculate the gradient of the pipeline, but is not implemented.
        """
        raise NotImplementedError("Gradient calculation is not implemented yet")
