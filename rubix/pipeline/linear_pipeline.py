from copy import deepcopy
from typing import Any, Callable, Sequence

from jax.tree_util import Partial

from . import abstract_pipeline as apl
from .transformer import bound_transformer


class LinearTransformerPipeline(apl.AbstractPipeline):
    """Minimal linear pipeline that chains transformer functions.

    An implementation of a data transformation pipeline in the form
    of a simple, 1-D chain of composed functions in which each function
    uses the output of the function before it as arguments.

    Note:
        This does only set up all necessary things to build the pipeline.
        The pipeline itself has to be created after registering transformers
        to use by calling `assemble()`.

    Args:
        cfg (dict): Configuration describing the pipeline nodes and their
            dependencies.
        transformers (list[Callable[..., Any]]): Transformer functions that
            will be registered before building the pipeline.
    """

    def __init__(
        self,
        cfg: dict,
        transformers: list[Callable[..., Any]],
    ) -> None:
        super().__init__(cfg, transformers)

    def update_pipeline(self, current_name: str) -> None:
        """Adds a new pipeline node with name 'current_name' to
        the pipeline, taking into account internal linear dependencies.
        Mostly used internally for adding nodes one by one.

        Args:
            current_name (str): Name of the node to append.

        Raises:
            RuntimeError: When the requested node is missing from the config.
        """

        if current_name not in self.config["Transformers"]:
            raise RuntimeError(f"Node '{current_name}' not found in the config")

        for key, node in self.config["Transformers"].items():
            if current_name == node["depends_on"]:
                func = bound_transformer(*node["args"], **node["kwargs"])(
                    self.transformers[node["name"]]
                )
                self._pipeline.append(func)
                self._names.append(key)
                self.update_pipeline(key)

    def build_pipeline(self) -> None:
        """Construct the pipeline from the stored configuration.

        Note:
            This only works when all transformers the pipeline is composed of
            have been registered with it. Multiple different versions
            (configurations) of the same transformer can be used in a pipeline.

        Raises:
            RuntimeError: When no transformers are registered.
            ValueError: When the configuration contains invalid or branching
                dependencies or when there are multiple starting points to the
                pipeline.

        """

        if len(self.transformers) == 0:
            raise RuntimeError("No registered transformers present")

        # sanity check: and make sure that dependencies are not there multiple
        # times, as branching is not allowed either
        # use a set for quick lookup. This check also captures
        # multiple end points.
        dependencies: set[str] = set()
        for key, node in self.config["Transformers"].items():

            if "name" not in node:
                raise ValueError(
                    (
                        "Each node of a pipeline must have a config node "
                        "containing 'name'"
                    )
                )

            if "args" not in node:
                raise ValueError("Config node must have a possibly empty args element")

            if "kwargs" not in node:
                raise ValueError(
                    "Config node must have a possible empty kwargs element"
                )

            if "depends_on" not in node:
                raise ValueError(
                    (
                        "Config node must have a possibly 'null' valued node "
                        "depends_on"
                    )
                )

            dep = node["depends_on"]
            if dep is None:
                continue

            if dep in dependencies:
                raise ValueError(
                    (
                        "Dependencies must be unique in a linear pipeline as "
                        "branching is not allowed. "
                        f"Found {dep} at least twice"
                    )
                )
            else:
                dependencies.add(dep)

        # find the starting point
        start_func = None
        start_name = None
        for key, node in self.config["Transformers"].items():

            if node["depends_on"] is None and start_name is None:
                start_func = bound_transformer(*node["args"], **node["kwargs"])(
                    self.transformers[node["name"]]
                )
                start_name = key
            elif node["depends_on"] is None and start_name is not None:
                raise ValueError("There can only be one starting point.")
            else:
                continue

        self._pipeline = [start_func]
        self._names = [start_name]
        self.update_pipeline(start_name)

    def build_expression(self) -> None:
        """Compose the assembled pipeline into a single callable that has
        the same signature as the first element of the pipeline."""

        def expr(input, pipeline=[]):
            res = input
            for f in pipeline:
                res = f(res)
            return res

        # deepcopy is needed to isolate the expr-function instance from the
        # class, since in principle it's a closure that pulls in the
        # surrounding scope, which includes `self`
        self.expression = Partial(expr, pipeline=deepcopy(self._pipeline))

    def apply(
        self,
        *args: Any,
        static_args: Sequence[int] | None = None,
        static_kwargs: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the compiled pipeline to the provided positional arguments
        `*args` and keyword arguments `**kwargs` that match the signature of
        the first method in the pipeline with static keyword arguments that are
        not traced by JAX. First applies the jax jit to the pre-assembled
        pipeline, then applies the result to the provided arguments.

        Args:
            *args (Any): Positional arguments to feed to the pipeline's first
                element.
            static_args (Sequence[int] | None, optional): Positional indices
                that should not be traced by JAX. Defaults to ``None``.
            static_kwargs (Sequence[str] | None, optional): Keyword names
                that should not be traced by JAX. Defaults to ``None``.
            **kwargs (Any): Keyword arguments to forward to the pipeline.

        Returns:
            Any: The result of running the pipeline.

        Raises:
            ValueError: When no positional arguments are provided.
        """

        if len(args) == 0:
            raise ValueError("Cannot apply the pipeline to an empty list of arguments")

        if self.compiled_expression is None:
            self.compile_expression(
                static_args=static_args,
                static_kwargs=static_kwargs,
            )

        return self.compiled_expression(*args, **kwargs)  # type: ignore
