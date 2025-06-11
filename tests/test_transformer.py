import jax.numpy as jnp
import pytest
from jax import jit, make_jaxpr, random
from jax.errors import TracerBoolConversionError

from rubix.pipeline import transformer as pt


def func(
    param: float, matrix: jnp.array, vector: jnp.array, kparam: float = 3.14
) -> jnp.array:
    if kparam > 5.0:
        return jnp.dot(matrix, vector.T) * param + kparam
    else:
        return jnp.dot(matrix, vector.T) * param - kparam


@pytest.fixture()
def data():
    key = random.key(0)
    m = random.normal(key, (2, 2), dtype=jnp.float32)
    v = random.normal(key, (2,), dtype=jnp.float32)
    p = 2.71
    k_l = 6.28
    k_s = 3.14

    return m, v, p, k_l, k_s


def test_bound_transformer(data):
    m, v, p, k_l, k_s = data

    f_l = pt.bound_transformer(p, kparam=k_l)(func)

    assert f_l.args == (2.71,)
    assert f_l.keywords == {
        "kparam": k_l,
    }

    f_s = pt.bound_transformer(p, kparam=k_s)(func)

    assert jnp.allclose(
        f_l(
            m,
            v,
        ).block_until_ready(),
        func(p, m, v, kparam=k_l).block_until_ready(),
    )

    assert jnp.allclose(f_s(m, v), func(p, m, v, kparam=k_s))

    jit_f_l = jit(
        pt.bound_transformer(p, kparam=k_l)(func),
        static_argnames=[
            "kparam",
        ],
    )

    jit_f_s = jit(
        pt.bound_transformer(p, kparam=k_s)(func),
        static_argnames=[
            "kparam",
        ],
    )

    assert jnp.allclose(
        jit_f_l(
            m,
            v,
        ),
        func(p, m, v, kparam=k_l),
    )

    assert jnp.allclose(jit_f_s(m, v), func(p, m, v, kparam=k_s))


def test_compiled_transformer(data):
    m, v, p, k_l, k_s = data

    jfl = pt.compiled_transformer(
        p,
        static_kwargs=[
            "kparam",
        ],
        kparam=k_l,
    )(func)

    jfs = pt.compiled_transformer(
        p,
        static_kwargs=[
            "kparam",
        ],
        kparam=k_s,
    )(func)

    assert jnp.allclose(
        jfl(m, v).block_until_ready(),
        func(p, m, v, kparam=k_l).block_until_ready(),
    )

    assert jnp.allclose(jfs(m, v), func(p, m, v, kparam=k_s))

    # non static kwarg results in tracing error
    jfl = pt.compiled_transformer(static_kwargs=[])(func)

    with pytest.raises(
        TracerBoolConversionError,
        match="Attempted boolean conversion of traced array with shape bool",
    ):
        jfl(p, m, v, kparam=k_l).block_until_ready()


def test_expression_transformer(data):
    m, v, p, k_l, k_s = data

    f_l_expr = pt.expression_transformer(
        p,
        m,
        v,
        k_l,
        static_args=[
            3,
        ],
    )(func)

    func_l_expr = make_jaxpr(
        func,
        static_argnums=[
            3,
        ],
    )(p, m, v, k_l)

    # check that computational structure is the same.
    # Input and output variables and source info differ.
    assert len(f_l_expr.eqns) == len(func_l_expr.eqns)

    for op1, op2 in zip(f_l_expr.eqns, func_l_expr.eqns):
        assert op1.primitive == op2.primitive
        assert op1.params == op2.params
        assert op1.effects == op2.effects

    assert str(f_l_expr) == str(func_l_expr)

    assert len(f_l_expr.eqns) == len(func_l_expr.eqns)

    f_l_exprfunc = pt.expression_transformer(
        static_args=[
            3,
        ],
    )(func)

    f_func = f_l_exprfunc(
        p,
        m,
        v,
        k_l,
    )

    assert len(f_func.eqns) == len(func_l_expr.eqns)

    for op1, op2 in zip(f_func.eqns, func_l_expr.eqns):
        assert op1.primitive == op2.primitive
        assert op1.params == op2.params
        assert op1.effects == op2.effects

    assert str(f_func) == str(func_l_expr)

    assert len(f_func.eqns) == len(func_l_expr.eqns)
