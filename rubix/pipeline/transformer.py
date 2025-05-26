from copy import deepcopy

from jax import jit, make_jaxpr
from jax.tree_util import Partial


def bound_transformer(*args, **kwargs):
    """
    bound_transformer  creates a jax.Partial function from an input
    function and given arguments and keyword arguments.
    The user must take care that arguments are bound starting from the first,
    i.e., leftmost. If specific arguments should be bound using keyword
    arguments may be advisable.
    """

    def transformer_wrap(kernel):
        return Partial(
            deepcopy(kernel), *deepcopy(args), **deepcopy(kwargs)
        )  # deepcopy to avoid context dependency

    return transformer_wrap


def compiled_transformer(
    *args,
    static_args: list = [],
    static_kwargs: list = [],
    **kwargs,
):
    """
    compiled_transformer creates a precompiled function with jax with given arguments and keyword arguments that will be bound to the function, similar
    to using functools.partial with *args and **kwargs.

    Note that any array args/kwargs will behave as dynamic arguments in the jax jit, while any non-array args/kwargs will behave as static.
    static_args and static_kwargs refer to the remaining arguments.

    *args count from the first positional argument of the decorated function in order. *args and **kwargs are bound to the decorated function

    Args:
        static_args (list, optional): Indices of static, i.e., untraced arguments of the bound function, by default [].
        static_kwargs (list, optional): Names of static, i.e., untraced, keyword arguments of the bound function, by default {}.
    """

    def transformer_wrap(kernel):

        return jit(
            Partial(deepcopy(kernel), *deepcopy(args), **deepcopy(kwargs)),
            static_argnums=deepcopy(static_args),
            static_argnames=deepcopy(static_kwargs),
        )

    return transformer_wrap


def expression_transformer(
    *args,
    static_args: list = [],
):
    """
    expression_transformer creates a jax intermediate expression with given
    untraced arguments from a function. Please note that this only works with
    static positional arguments: JAX does currently not provide a way to have
    static keyword arguments when creating a jaxpr and not a jited function.

    Args:
        static_args (list, optional): Indices of static, i.e., untraced arguments to the function, by default [].
    """

    def transformer_wrap(kernel):
        if len(args) > 0:
            return make_jaxpr(deepcopy(kernel), static_argnums=static_args)(
                *deepcopy(args)
            )
        else:
            return make_jaxpr(deepcopy(kernel), static_argnums=static_args)

    return transformer_wrap
