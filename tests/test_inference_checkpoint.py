import jax.numpy as jnp

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import (
    load_checkpoint,
    make_optimization_checkpoint,
    make_variational_checkpoint,
    optimize_params,
    optimize_variational_posterior,
    resume_optimization_from_checkpoint,
    resume_variational_from_checkpoint,
    save_checkpoint,
)


class DummyPipeline:
    """Simple differentiable pipeline for checkpoint tests."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        value = jnp.sum(rubixdata.stars.age) + 2.0 * jnp.sum(
            rubixdata.stars.metallicity
        )
        return jnp.reshape(value, (1, 1, 1))


def _make_rubix_data() -> RubixData:
    return RubixData(
        galaxy=Galaxy(),
        stars=StarsData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
            age=jnp.array([1.0]),
            metallicity=jnp.array([0.01]),
        ),
        gas=GasData(
            coords=jnp.zeros((1, 3)),
            velocity=jnp.zeros((1, 3)),
            mass=jnp.ones(1),
        ),
    )


def test_optimize_state_resume_matches_single_run(tmp_path):
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.0]), "metallicity": jnp.array([0.0])}}
    target = jnp.array([[[5.0]]])

    full = optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=0.1,
        max_steps=40,
        tol=1e-8,
    )

    first, state = optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=0.1,
        max_steps=20,
        tol=1e-8,
        return_state=True,
    )

    ckpt_path = tmp_path / "opt.pkl"
    save_checkpoint(
        ckpt_path,
        make_optimization_checkpoint(first, state, learning_rate=0.1, tol=1e-8),
    )
    ckpt = load_checkpoint(ckpt_path)

    resumed, _ = resume_optimization_from_checkpoint(
        ckpt,
        pipeline=pipeline,
        static_data=static_data,
        target=target,
        max_steps=20,
    )

    assert jnp.allclose(resumed.final_loss, full.final_loss)


def test_variational_state_resume_matches_single_run(tmp_path):
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {"age": jnp.array([0.5]), "metallicity": jnp.array([0.001])}
    }
    target = jnp.array([[[5.0]]])

    full = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=5e-2,
        max_steps=40,
        tol=1e-8,
        num_samples=4,
        beta_kl=1e-4,
        seed=11,
    )

    first, state = optimize_variational_posterior(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=5e-2,
        max_steps=20,
        tol=1e-8,
        num_samples=4,
        beta_kl=1e-4,
        seed=11,
        return_state=True,
    )

    ckpt_path = tmp_path / "vi.pkl"
    save_checkpoint(
        ckpt_path,
        make_variational_checkpoint(
            first,
            state,
            learning_rate=5e-2,
            tol=1e-8,
            num_samples=4,
            beta_kl=1e-4,
            seed=11,
        ),
    )
    ckpt = load_checkpoint(ckpt_path)

    resumed, _ = resume_variational_from_checkpoint(
        ckpt,
        pipeline=pipeline,
        static_data=static_data,
        target=target,
        max_steps=20,
    )

    assert jnp.allclose(resumed.final_objective, full.final_objective)
