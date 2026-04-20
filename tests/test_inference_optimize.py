import jax.numpy as jnp

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.inference import build_age_metallicity_transforms, optimize_params


class DummyPipeline:
    """Simple differentiable pipeline used for optimizer tests."""

    def run_sharded(self, rubixdata: RubixData) -> jnp.ndarray:
        age_sum = jnp.sum(rubixdata.stars.age)
        metallicity_sum = jnp.sum(rubixdata.stars.metallicity)
        value = age_sum + 2.0 * metallicity_sum
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


def test_optimize_params_reduces_loss_without_transforms():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {"stars": {"age": jnp.array([0.0]), "metallicity": jnp.array([0.0])}}
    target = jnp.array([[[5.0]]])

    result = optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=0.1,
        max_steps=250,
        tol=1e-6,
    )

    assert result.loss_history[0] > result.loss_history[-1]
    assert result.steps_run <= 250
    assert result.best_loss <= result.final_loss

    # Verify best_params and best_loss are internally consistent: recomputing
    # the loss at best_params should match result.best_loss.
    best_prediction = pipeline.run_sharded(
        RubixData(
            galaxy=static_data.galaxy,
            stars=StarsData(
                coords=static_data.stars.coords,
                velocity=static_data.stars.velocity,
                mass=static_data.stars.mass,
                age=result.best_params["stars"]["age"],
                metallicity=result.best_params["stars"]["metallicity"],
            ),
            gas=static_data.gas,
        )
    )
    best_loss_recomputed = float(jnp.sum((best_prediction - target) ** 2))
    assert abs(best_loss_recomputed - result.best_loss) < 1e-5

    prediction = pipeline.run_sharded(
        RubixData(
            galaxy=static_data.galaxy,
            stars=StarsData(
                coords=static_data.stars.coords,
                velocity=static_data.stars.velocity,
                mass=static_data.stars.mass,
                age=result.params["stars"]["age"],
                metallicity=result.params["stars"]["metallicity"],
            ),
            gas=static_data.gas,
        )
    )
    assert jnp.allclose(prediction, target, atol=2e-2, rtol=0.0)


def test_optimize_params_with_transforms_respects_bounds_and_converges():
    pipeline = DummyPipeline()
    static_data = _make_rubix_data()
    params_init = {
        "stars": {
            "age": jnp.array([0.5]),
            "metallicity": jnp.array([0.001]),
        }
    }
    transforms = build_age_metallicity_transforms(
        age_lower=0.0,
        age_upper=20.0,
        metallicity_lower=0.0,
        metallicity_upper=0.05,
    )
    target = jnp.array([[[7.04]]])  # 7.0 + 2 * 0.02

    result = optimize_params(
        pipeline=pipeline,
        params_init=params_init,
        static_data=static_data,
        target=target,
        learning_rate=0.15,
        max_steps=300,
        tol=1e-6,
        transforms=transforms,
    )

    final_age = result.params["stars"]["age"]
    final_metallicity = result.params["stars"]["metallicity"]

    assert jnp.all(final_age > 0.0)
    assert jnp.all(final_age < 20.0)
    assert jnp.all(final_metallicity > 0.0)
    assert jnp.all(final_metallicity < 0.05)
    assert result.loss_history[0] > result.loss_history[-1]
    assert len(result.grad_norm_history) == len(result.loss_history)
    assert result.best_loss <= result.final_loss

    # Verify best_params and best_loss are internally consistent: recomputing
    # the loss at best_params should match result.best_loss.
    best_prediction = pipeline.run_sharded(
        RubixData(
            galaxy=static_data.galaxy,
            stars=StarsData(
                coords=static_data.stars.coords,
                velocity=static_data.stars.velocity,
                mass=static_data.stars.mass,
                age=result.best_params["stars"]["age"],
                metallicity=result.best_params["stars"]["metallicity"],
            ),
            gas=static_data.gas,
        )
    )
    best_loss_recomputed = float(jnp.sum((best_prediction - target) ** 2))
    assert abs(best_loss_recomputed - result.best_loss) < 1e-5

    prediction = pipeline.run_sharded(
        RubixData(
            galaxy=static_data.galaxy,
            stars=StarsData(
                coords=static_data.stars.coords,
                velocity=static_data.stars.velocity,
                mass=static_data.stars.mass,
                age=final_age,
                metallicity=final_metallicity,
            ),
            gas=static_data.gas,
        )
    )
    assert jnp.allclose(prediction, target, atol=2e-2, rtol=0.0)
