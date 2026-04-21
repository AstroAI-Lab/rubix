import jax
import jax.numpy as jnp

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import forward, optimize_params, optimize_variational_ifu_cube


class SyntheticIFUPipeline:
    """Tiny synthetic pipeline supporting deterministic and stochastic runs."""

    def __init__(self, template: jnp.ndarray):
        self.template = template

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        # Simple scale parameter from stars.age, plus optional keyed jitter
        # to emulate stochastic forward passes.
        scale = rubixdata.stars.age[0]
        cube = scale * self.template

        if rubixdata.noise_key is None:
            return cube

        noise = jax.random.normal(
            rubixdata.noise_key, shape=cube.shape, dtype=cube.dtype
        )
        return cube + 0.01 * noise


def _make_rubix_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
            age=jnp.array([0.0]),
            metallicity=jnp.array([0.01]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def test_e2e_deterministic_stochastic_and_vi_smoke():
    cube_shape = (2, 2, 8)
    template = jnp.ones(cube_shape, dtype=jnp.float32)
    pipeline = SyntheticIFUPipeline(template)

    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.2])}}
    target = 1.5 * template

    # Deterministic gradient-based fit
    det_result = optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=0.2,
        max_steps=80,
        tol=1e-8,
    )
    assert det_result.loss_history[0] > det_result.loss_history[-1]

    # Verify noise_key plumbing: predictions must differ between None and a key.
    noise_key = jax.random.PRNGKey(7)
    pred_det = forward(pipeline, params_init, static_data, noise_key=None)
    pred_sto = forward(pipeline, params_init, static_data, noise_key=noise_key)
    assert not jnp.allclose(pred_det, pred_sto), (
        "noise_key had no effect on forward output; plumbing may be broken"
    )

    # Two distinct keys must also produce different noisy predictions.
    pred_sto2 = forward(
        pipeline, params_init, static_data, noise_key=jax.random.PRNGKey(42)
    )
    assert not jnp.allclose(pred_sto, pred_sto2), (
        "different noise_keys produced identical outputs; PRNG splitting may be broken"
    )

    # Stochastic run with explicit noise key should remain numerically stable.
    sto_result = optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=0.1,
        max_steps=20,
        tol=1e-8,
        noise_key=noise_key,
    )
    assert jnp.isfinite(sto_result.final_loss)
    # The stochastic loss trace must also differ from the deterministic one at
    # step 0, confirming that noise_key was threaded through optimize_params.
    assert not jnp.isclose(
        jnp.array(sto_result.loss_history[0]),
        jnp.array(det_result.loss_history[0]),
    ), (
        "stochastic and deterministic first-step losses are identical; "
        "noise_key may not be reaching the pipeline inside optimize_params"
    )

    # VI smoke run on the same synthetic cube.
    vi_result = optimize_variational_ifu_cube(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        sigma=jnp.ones_like(target),
        learning_rate=5e-2,
        max_steps=60,
        tol=1e-8,
        num_samples=3,
        beta_kl=1e-4,
        seed=3,
    )
    assert vi_result.objective_history[0] > vi_result.objective_history[-1]
    assert jnp.isfinite(vi_result.final_objective)
