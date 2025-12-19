import time
import warnings

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Any, Optional, Sequence, Union
from jax import lax

try:
    from jax.shard_map import shard_map  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - older JAX compatibility
    warnings.filterwarnings(
        "ignore",
        message="jax.experimental.shard_map is deprecated in v0.8.0.*",
        category=DeprecationWarning,
        module=__name__,
    )
    from jax.experimental.shard_map import shard_map

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.tree_util import tree_map
from jaxtyping import jaxtyped

from rubix.logger import get_logger
from rubix.pipeline import linear_pipeline as pipeline
from rubix.utils import _pad_particles, get_config, get_pipeline_config

from .data import Galaxy, GasData, RubixData, StarsData, get_rubix_data
from .dust import get_extinction
from .ifu import (
    get_calculate_datacube_particlewise,
    get_calculate_dusty_datacube_particlewise,
)
from .lsf import get_convolve_lsf
from .noise import get_apply_noise
from .psf import get_convolve_psf
from .rotation import get_galaxy_rotation
from .ssp import get_ssp
from .telescope import get_filter_particles, get_spaxel_assignment, get_telescope


class RubixPipeline:
    """Builds and executes the Rubix data processing pipeline with the provided configuration.

    Args:
        user_config (Union[dict, str]):
            Parsed configuration dictionary or path to a configuration file.

    Example:

            >>> from rubix.core.pipeline import RubixPipeline
            >>> config = "path/to/config.yml"
            >>> target_datacube = ...  # Load or define your target datacube here
            >>> pipe = RubixPipeline(config)
            >>> inputdata = pipe.prepare_data()
            >>> final_datacube = pipe.run_sharded(inputdata)
            >>> ssp_model = pipe.ssp
            >>> telescope = pipe.telescope
            >>> loss_value = pipe.loss(inputdata, target_datacube)
            >>> gradient_data = pipe.gradient(inputdata, target_datacube)
    """

    def __init__(self, user_config: Union[dict, str]):
        self.user_config = get_config(user_config)
        pipeline_name = self.user_config["pipeline"]["name"]
        self.pipeline_config = get_pipeline_config(pipeline_name)
        self.logger = get_logger(self.user_config["logger"])
        self.ssp = get_ssp(self.user_config)
        self.telescope = get_telescope(self.user_config)
        self.func = None

    def prepare_data(self):
        """
        Prepares and loads the data for the pipeline.

        Returns:
            Object containing particle data with attributes such as:
                'coords', 'velocities', 'mass', 'age', and 'metallicity'
                under stars and gas.
        """
        t1 = time.time()
        self.logger.info("Getting rubix data...")
        rubixdata = get_rubix_data(self.user_config)
        star_count = (
            len(rubixdata.stars.coords) if rubixdata.stars.coords is not None else 0
        )
        gas_count = len(rubixdata.gas.coords) if rubixdata.gas.coords is not None else 0
        self.logger.info(
            f"Data loaded with {star_count} star particles and {gas_count} gas particles."
        )
        t2 = time.time()
        self.logger.info(f"Data preparation completed in {t2 - t1:.2f} seconds.")
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
        calculate_extinction = get_extinction(self.user_config)
        calculate_datacube_particlewise = get_calculate_datacube_particlewise(
            self.user_config
        )
        calculate_dusty_datacube_particlewise = (
            get_calculate_dusty_datacube_particlewise(self.user_config)
        )
        convolve_psf = get_convolve_psf(self.user_config)
        convolve_lsf = get_convolve_lsf(self.user_config)
        apply_noise = get_apply_noise(self.user_config)

        functions = [
            rotate_galaxy,
            filter_particles,
            spaxel_assignment,
            calculate_extinction,
            calculate_datacube_particlewise,
            calculate_dusty_datacube_particlewise,
            convolve_psf,
            convolve_lsf,
            apply_noise,
        ]
        return functions

    def run_sharded(
        self,
        inputdata: RubixData,
        devices: Optional[Sequence[Any]] = None,
    ) -> jnp.ndarray:
        """
            Run the compiled pipeline across devices by sharding the particle data.

        It splits the particle arrays under stars and gas into shards,
            runs the compiled pipeline on each shard, and then combines the
            resulting datacubes.

            Note:
                This is the recommended method to run the pipeline in parallel at
                the moment.

            Args:
                inputdata (RubixData):
                    Output of :py:meth:`prepare_data`.
                    Contains star and gas particles.
                devices (Optional[Sequence[Any]], optional):
                    Devices to use for :func:`shard_map`. These should be
                    :class:`jax.Device` instances.
                    Defaults to ``jax.devices()``.

            Returns:
                jnp.ndarray: Sharded pipeline output aggregated across devices.
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

        if devices is None:  # pragma: no cover
            devices = jax.devices()
            num_devices = len(devices)
        else:
            num_devices = len(devices)
        self.logger.info("Number of devices: %d", num_devices)

        mesh = Mesh(devices, axis_names=("data",))

        # — sharding specs by rank —
        replicate_0d = NamedSharding(mesh, P())  # for scalars
        replicate_1d = NamedSharding(mesh, P(None))  # for 1-D arrays
        shard_2d = NamedSharding(mesh, P("data", None))  # for (N, D)
        shard_1d = NamedSharding(mesh, P("data"))  # for (N,)
        shard_bins = NamedSharding(mesh, P(None, None))
        replicate_3d = NamedSharding(
            mesh,
            P(None, None, None),
        )  # for full cube

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
        stars_spec.spatial_bin_edges = shard_bins
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
        gas_spec.spatial_bin_edges = shard_bins
        gas_spec.mask = shard_1d
        gas_spec.spectra = shard_2d
        gas_spec.datacube = replicate_3d

        # — link them up —
        rubix_spec.galaxy = galaxy_spec
        rubix_spec.stars = stars_spec
        rubix_spec.gas = gas_spec

        # 1) Make a pytree of PartitionSpec
        partition_spec_tree = tree_map(
            lambda s: s.spec if isinstance(s, NamedSharding) else None,
            rubix_spec,
        )

        # If the particle number is not divisible by the device count,
        # pad a few empty particles so the numbers line up.
        n = inputdata.stars.coords.shape[0]
        pad = (num_devices - (n % num_devices)) % num_devices
        if pad:
            self.logger.info(
                "Padding particles to make the number of particles divisible "
                "by the number of devices (%d).",
                num_devices,
            )
            inputdata = _pad_particles(inputdata, pad)

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

        sharded_result = sharded_pipeline(inputdata)

        time_end = time.time()
        self.logger.info(
            "Total time for sharded pipeline run: %.2f seconds.",
            time_end - time_start,
        )

        return sharded_result

    def gradient(
        self,
        rubixdata: RubixData,
        targetdata: jnp.ndarray,
    ) -> RubixData:
        """Compute the gradient of the loss with respect to ``rubixdata``.

        Args:
            rubixdata (RubixData):
                Pytree describing the current pipeline input data.
            targetdata (jnp.ndarray):
                Target datacube used to compute the loss.

        Returns:
            RubixData:
                Gradient pytree that matches the structure of ``rubixdata``.
        """
        return jax.grad(self.loss, argnums=0)(rubixdata, targetdata)

    def loss(
        self,
        rubixdata: RubixData,
        targetdata: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the mean squared error between pipeline output and target.

        Args:
            rubixdata (RubixData):
                Input data passed to :py:meth:`run`.
            targetdata (jnp.ndarray):
                Target datacube used for comparison.

        Returns:
            jnp.ndarray:
                Scalar mean squared error value.
        """
        output = self.run_sharded(rubixdata)
        loss_value = jnp.sum((output - targetdata) ** 2)
        return loss_value
