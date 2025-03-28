import time
from types import SimpleNamespace
from typing import Union

import jax
import jax.numpy as jnp

# For shard_map and device mesh.
import numpy as np
from beartype import beartype as typechecker
from jax import block_until_ready
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec as P
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

    def run_sharded(self, inputdata, shard_size=100000):
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

        # --- Helper: Shard the particle data ---
        def shard_subdata(subdata):
            # subdata is expected to be a SimpleNamespace with attributes that are arrays.
            new_subdata = {}
            for attr, value in vars(subdata).items():
                if hasattr(value, "shape"):
                    n_particles = value.shape[0]
                    n_shards = n_particles // shard_size
                    # Truncate if needed.
                    new_value = value[: n_shards * shard_size]
                    # Reshape so that the first dimension indexes shards.
                    new_subdata[attr] = new_value.reshape(
                        (n_shards, shard_size) + value.shape[1:]
                    )
                else:
                    new_subdata[attr] = value
            return SimpleNamespace(**new_subdata)

        # Create a new sharded input object.
        sharded_input = {}
        # Assume that 'stars' and 'gas' contain particle data.
        for key in ["stars", "gas"]:
            if hasattr(inputdata, key):
                sharded_input[key] = shard_subdata(getattr(inputdata, key))
        # Preserve other parts (e.g. galaxy and units) as-is.
        for key in vars(inputdata):
            if key not in sharded_input:
                sharded_input[key] = getattr(inputdata, key)
        sharded_input = SimpleNamespace(**sharded_input)
        # -----------------------------------------

        # Determine the number of shards from one batched array (e.g. stars.coords).
        n_shards = sharded_input.stars.coords.shape[0]
        devices = np.array(jax.devices())
        if n_shards != devices.shape[0]:
            raise ValueError(
                f"Number of shards ({n_shards}) must equal number of devices ({devices.shape[0]})."
            )
        mesh = Mesh(devices, ("x",))

        # Define a function that will process one shard.
        def pipeline_shard_fn(shard_input):
            # Here, shard_input is a dict (or nested namespace) for one shard.
            output = self.func(shard_input)
            # Assume output has a 'datacube' attribute.
            return output.datacube

        # Convert the sharded input namespace to a dict.
        def to_dict(ns):
            if isinstance(ns, SimpleNamespace):
                return {k: to_dict(v) for k, v in vars(ns).items()}
            else:
                return ns

        sharded_input_dict = to_dict(sharded_input)

        # Create partitioning specifications for all array leaves.
        def create_sharding_spec(val):
            if hasattr(val, "shape") and isinstance(val, jnp.ndarray):
                return P("x")
            elif isinstance(val, dict):
                return {k: create_sharding_spec(v) for k, v in val.items()}
            else:
                return None

        in_shardings = jax.tree_util.tree_map(create_sharding_spec, sharded_input_dict)
        # Assume output datacube is sharded along 'x'.
        out_shardings = P("x")

        # Use jax.shard_map to parallelize across shards.
        sharded_pipeline_fn = shard_map.shard_map(
            pipeline_shard_fn,
            in_shardings,
            out_shardings,
            mesh,
        )

        with mesh:
            sharded_datacubes = sharded_pipeline_fn(sharded_input_dict)

        # Combine the datacubes (here, by summing over the shard axis).
        final_datacube = jnp.sum(sharded_datacubes, axis=0)
        time_end = time.time()
        self.logger.info(
            "Sharded pipeline run completed in %.2f seconds.", time_end - time_start
        )
        return final_datacube

    def gradient(self):
        """
        This function will calculate the gradient of the pipeline, but is not implemented.
        """
        raise NotImplementedError("Gradient calculation is not implemented yet")
