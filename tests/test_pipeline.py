from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest
from jax import jit, make_jaxpr
from jax.tree_util import Partial

from rubix.core.data import Galaxy, GasData, RubixData, StarsData
from rubix.core.pipeline import RubixPipeline
from rubix.pipeline import linear_pipeline as lp
from rubix.utils import _pad_particles, read_yaml


# helper stuff that we need
def add(x, s: float = 0.0):
    return x + s


def mult(x, m: float = 1):
    return x * m


def div(x, d: float = 1):
    return x / d


def sub(x, s: float = 0.0):
    return x - s


def manual_func(x):
    return div(mult(add(sub(add(x, s=4), s=2), s=3), m=3), d=4)


@pytest.fixture
def pipeline_fixture():
    cfg = read_yaml(Path(__file__).parent / "demo.yml")

    return cfg


@pytest.fixture
def pipeline_fixture_full():
    cfg = read_yaml(Path(__file__).parent / "demo.yml")

    ppl = lp.LinearTransformerPipeline(cfg, [add, sub, mult, div])

    x = jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32)

    return ppl, x


def test_pipeline_construction(pipeline_fixture):
    cfg = pipeline_fixture

    pipeline = lp.LinearTransformerPipeline(cfg, [add, sub, mult, div])

    assert pipeline.config == cfg
    assert len(pipeline._pipeline) == len(cfg["Transformers"])
    assert len(pipeline._names) == 5
    assert len(pipeline.transformers) == 4
    assert list(pipeline.transformers.keys()) == ["add", "sub", "mult", "div"]
    assert pipeline.expression is not None
    assert pipeline.compiled_expression is None


def test_register_transformer(pipeline_fixture):
    cfg = pipeline_fixture

    pipeline = lp.LinearTransformerPipeline(cfg, [add, sub, mult, div])

    pipeline.transformers = {}

    pipeline.register_transformer(add)
    pipeline.register_transformer(sub)
    pipeline.register_transformer(mult)
    pipeline.register_transformer(div)

    assert pipeline.transformers == {
        "add": add,
        "sub": sub,
        "mult": mult,
        "div": div,
    }

    with pytest.raises(
        ValueError, match="A transformer of this name is already present"
    ):
        pipeline.register_transformer(add)


def test_update_pipeline(pipeline_fixture_full):
    pipeline, _ = pipeline_fixture_full

    pipeline.config["Transformers"]["D"] = {
        "name": "add",
        "depends_on": "Z",
        "args": [],
        "kwargs": {"s": 1.4},
    }

    pipeline.update_pipeline("Z")

    assert pipeline._names == ["C", "B", "A", "X", "Z", "D"]

    with pytest.raises(
        RuntimeError,
        match="Node 'Not there' not found in the config",
    ):
        pipeline.update_pipeline("Not there")

    assert pipeline._names == ["C", "B", "A", "X", "Z", "D"]


def test_build_pipeline(pipeline_fixture_full):
    pipeline, _ = pipeline_fixture_full

    pipeline.build_pipeline()

    assert pipeline._names == ["C", "B", "A", "X", "Z"]

    assert list(pipeline.pipeline.keys()) == ["C", "B", "A", "X", "Z"]


def test_build_pipeline_broken(pipeline_fixture_full):
    pipeline, _ = pipeline_fixture_full
    cfg = deepcopy(pipeline.config)

    del pipeline.config["Transformers"]["C"]["name"]

    with pytest.raises(
        ValueError,
        match=("Each node of a pipeline must have a config node containing " "'name'"),
    ):
        pipeline.build_pipeline()

    pipeline.config = deepcopy(cfg)

    pipeline.config["Transformers"]["X"]["depends_on"] = None

    with pytest.raises(
        ValueError,
        match="There can only be one starting point",
    ):
        pipeline.build_pipeline()

    pipeline.config = deepcopy(cfg)

    pipeline.config["Transformers"]["D"] = {
        "name": add,
        "depends_on": "X",
        "kwargs": {"s": 1.4},
    }

    with pytest.raises(
        ValueError,
        match="Config node must have a possibly empty args element",
    ):
        pipeline.build_pipeline()

    pipeline.config["Transformers"]["D"] = {
        "name": add,
        "depends_on": "X",
        "args": [],
    }

    with pytest.raises(
        ValueError,
        match="Config node must have a possible empty kwargs element",
    ):
        pipeline.build_pipeline()

    pipeline.config["Transformers"]["D"] = {
        "name": add,
        "depends_on": "X",
        "args": [],
        "kwargs": {"s": 1.4},
    }

    with pytest.raises(
        ValueError,
        match=(
            "Dependencies must be unique in a linear pipeline as branching is "
            "not allowed. Found X at least twice"
        ),
    ):
        pipeline.build_pipeline()

    pipeline.config["Transformers"]["D"] = {
        "name": add,
        "args": [],
        "kwargs": {"s": 1.4},
    }

    with pytest.raises(
        ValueError,
        match="Config node must have a possibly 'null' valued node depends_on",
    ):
        pipeline.build_pipeline()

    pipeline.transformers = []

    with pytest.raises(
        RuntimeError,
        match="No registered transformers present",
    ):
        pipeline.build_pipeline()


def test_build_expression(pipeline_fixture_full):
    pipeline, x = pipeline_fixture_full

    pipeline.build_pipeline()

    pipeline.build_expression()

    assert jnp.allclose(manual_func(x), pipeline.expression(x))


def test_assemble(pipeline_fixture_full):
    pipeline, x = pipeline_fixture_full

    pipeline.assemble()

    assert jnp.allclose(manual_func(x), pipeline.expression(x))

    assert pipeline._names == ["C", "B", "A", "X", "Z"]


def test_compile_expression(pipeline_fixture_full):
    pipeline, x = pipeline_fixture_full

    pipeline.assemble()

    f = pipeline.compile_expression()

    assert jnp.allclose(manual_func(x), f(x))

    pipeline.expression = "this won't work"

    with pytest.raises(RuntimeError, match="Expression compilation failed"):
        pipeline.compile_expression()


def test_get_jaxpr(pipeline_fixture_full):
    from functools import partial

    pipeline, x = pipeline_fixture_full

    # README: this does just fix the s parameter to 1.0.
    # This does not solve the underlying problem that get_jaxpr does not have
    # a `static_argnames` facility.
    pipeline.expression = partial(pipeline.expression, s=1.0)
    pipeline.assemble()

    expr = pipeline.get_jaxpr(
        x,
    )

    manual_expr = make_jaxpr(manual_func)(x)

    # check that the expressions are equivalent
    assert len(expr.eqns) == len(manual_expr.eqns)

    for op1, op2 in zip(expr.eqns, manual_expr.eqns):
        assert op1.primitive == op2.primitive
        assert op1.params == op2.params
        assert op1.effects == op2.effects

    assert str(expr) == str(manual_expr)

    assert len(expr.eqns) == len(manual_expr.eqns)

    for op1, op2 in zip(expr.eqns, manual_expr.eqns):
        assert op1.primitive == op2.primitive
        assert op1.params == op2.params
        assert op1.effects == op2.effects

    assert str(expr) == str(manual_expr)


def test_apply(pipeline_fixture_full):
    pipeline, x = pipeline_fixture_full

    with pytest.raises(
        ValueError,
        match="Cannot apply the pipeline to an empty list of arguments",
    ):
        pipeline.apply()

    res = pipeline.apply(x)

    assert jnp.allclose(res, manual_func(x))

    assert pipeline.compiled_expression is not None


def test_get_jaxpr_for_element(pipeline_fixture_full):
    pipeline, x = pipeline_fixture_full

    expr = make_jaxpr(Partial(mult, m=3.0))(x)  # that is what we do internally

    manual_expr = pipeline.get_jaxpr_for_element("X", x)

    assert len(expr.eqns) == len(manual_expr.eqns)

    for op1, op2 in zip(expr.eqns, manual_expr.eqns):
        assert op1.primitive == op2.primitive
        assert op1.params == op2.params
        assert op1.effects == op2.effects

    assert str(expr) == str(manual_expr)

    assert len(expr.eqns) == len(manual_expr.eqns)

    for op1, op2 in zip(expr.eqns, manual_expr.eqns):
        assert op1.primitive == op2.primitive
        assert op1.params == op2.params
        assert op1.effects == op2.effects

    assert str(expr) == str(manual_expr)

    with pytest.raises(
        RuntimeError,
        match="Cannot create intermediate expression for 'Not there'",
    ):
        pipeline.get_jaxpr_for_element(
            "Not there",
            x,
            static_args=[
                1,
            ],
        )


def test_compile_element(pipeline_fixture_full):
    pipeline, x = pipeline_fixture_full

    manual = jit(
        Partial(mult, m=3.0),
        static_argnames=[
            "m",
        ],
    )

    fp = pipeline.compile_element(
        "X",
        static_kwargs=[
            "m",
        ],
    )

    assert jnp.allclose(manual(x), fp(x))

    with pytest.raises(
        RuntimeError,
        match="Compilation of element 'Not there' failed",
    ):
        pipeline.compile_element(
            "Not there",
            static_kwargs=[
                "m",
            ],
        )


@pytest.fixture
def simple_pipeline(monkeypatch):
    user_config = {
        "pipeline": {"name": "test_pipeline"},
        "logger": {"log_level": "INFO"},
    }
    pipeline_config = {"Transformers": {}}
    logger = MagicMock()

    monkeypatch.setattr("rubix.core.pipeline.get_config", lambda cfg: cfg)
    monkeypatch.setattr(
        "rubix.core.pipeline.get_pipeline_config",
        lambda name: pipeline_config,
    )
    monkeypatch.setattr("rubix.core.pipeline.get_logger", lambda cfg: logger)
    monkeypatch.setattr("rubix.core.pipeline.get_ssp", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "rubix.core.pipeline.get_telescope",
        lambda cfg: MagicMock(),
    )

    pipeline = RubixPipeline(user_config)
    return pipeline, logger


def _make_rubix_data(star_count=3, gas_count=2):
    stars = StarsData(
        coords=jnp.zeros((star_count, 3)),
        velocity=jnp.zeros((star_count, 3)),
        mass=jnp.arange(star_count, dtype=jnp.float32),
        age=jnp.arange(star_count, dtype=jnp.float32),
        metallicity=jnp.arange(star_count, dtype=jnp.float32),
    )
    gas = GasData(
        coords=jnp.zeros((gas_count, 3)),
        velocity=jnp.zeros((gas_count, 3)),
        mass=jnp.ones(gas_count, dtype=jnp.float32),
        density=jnp.ones(gas_count, dtype=jnp.float32),
        internal_energy=jnp.ones(gas_count, dtype=jnp.float32),
        metallicity=jnp.ones(gas_count, dtype=jnp.float32),
    )
    data = RubixData(galaxy=Galaxy(), stars=stars, gas=gas)
    return data


def test_prepare_data_logs_counts(simple_pipeline, monkeypatch):
    pipeline, logger = simple_pipeline
    rubixdata = _make_rubix_data(star_count=4, gas_count=3)

    monkeypatch.setattr(
        "rubix.core.pipeline.get_rubix_data",
        lambda cfg: rubixdata,
    )

    result = pipeline.prepare_data()

    assert result is rubixdata
    assert any("Data loaded" in call.args[0] for call in logger.info.call_args_list)


def test_pad_particles_extends_arrays():
    data = _make_rubix_data(star_count=2)
    padded = _pad_particles(data, pad=3)

    assert padded.stars.coords.shape[0] == 5
    assert jnp.count_nonzero(padded.stars.coords[-3:]) == 0
    assert padded.stars.mass[-3:].sum() == 0


def test_run_sharded_triggers_padding(simple_pipeline, monkeypatch):
    pipeline, _ = simple_pipeline
    data = _make_rubix_data(star_count=3, gas_count=1)

    mock_pad = MagicMock(side_effect=lambda inp, pad: inp)
    monkeypatch.setattr("rubix.core.pipeline._pad_particles", mock_pad)

    monkeypatch.setattr(RubixPipeline, "_get_pipeline_functions", lambda self: [])

    class DummyLinearPipeline:
        def __init__(self, cfg, functions):
            self.config = cfg

        def assemble(self):
            pass

        def compile_expression(self):
            class DummyOutput:
                def __init__(self):
                    self.stars = MagicMock(datacube=jnp.zeros((1, 1, 1)))

            return lambda *_: DummyOutput()

    monkeypatch.setattr(
        "rubix.core.pipeline.pipeline.LinearTransformerPipeline",
        DummyLinearPipeline,
    )
    monkeypatch.setattr(
        "rubix.core.pipeline.Mesh",
        lambda devices, axis_names: None,
    )

    class DummyNamedSharding:
        def __init__(self, mesh, spec):
            self.spec = spec

    monkeypatch.setattr(
        "rubix.core.pipeline.NamedSharding",
        DummyNamedSharding,
    )
    monkeypatch.setattr(
        "rubix.core.pipeline.P",
        lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(
        "rubix.core.pipeline.jax.device_put",
        lambda data, spec: data,
    )
    monkeypatch.setattr(
        "rubix.core.pipeline.lax.psum",
        lambda value, axis_name: value,
    )
    monkeypatch.setattr(
        "rubix.core.pipeline.shard_map",
        lambda func, mesh, in_specs, out_specs, check_rep: (
            lambda inputdata: func(inputdata)
        ),
    )

    result = pipeline.run_sharded(data, devices=[object(), object()])

    assert mock_pad.call_count == 1
    _, pad_arg = mock_pad.call_args[0]
    assert pad_arg == 1
    assert isinstance(result, jnp.ndarray)


def test_gradient_calls_jax_grad(simple_pipeline, monkeypatch):
    pipeline, _ = simple_pipeline
    expected = MagicMock()
    captured = {}

    def fake_grad(fn, argnums=0):
        captured["fn"] = fn
        captured["argnums"] = argnums

        def gradient_fn(rubixdata, targetdata):
            captured["rubixdata"] = rubixdata
            captured["targetdata"] = targetdata
            return expected

        return gradient_fn

    monkeypatch.setattr("rubix.core.pipeline.jax.grad", fake_grad)
    rubixdata = MagicMock()
    target = MagicMock()

    result = pipeline.gradient(rubixdata, target)

    assert captured["fn"].__func__ is pipeline.loss.__func__
    assert captured["fn"].__self__ is pipeline
    assert captured["argnums"] == 0
    assert captured["rubixdata"] is rubixdata
    assert captured["targetdata"] is target
    assert result is expected


def test_loss_uses_run(simple_pipeline):
    pipeline, _ = simple_pipeline
    rubixdata = MagicMock()
    target = jnp.array([1.0, 2.0])
    output = jnp.array([3.0, 4.0])

    pipeline.run_sharded = MagicMock(return_value=output)

    loss_value = pipeline.loss(rubixdata, target)

    pipeline.run_sharded.assert_called_once_with(rubixdata)
    expected = jnp.sum((output - target) ** 2)
    assert jnp.allclose(loss_value, expected)
