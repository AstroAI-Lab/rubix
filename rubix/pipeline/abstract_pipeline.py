from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from jax import jit

try:
    from jax._src.core import ClosedJaxpr
except ImportError:  # pragma: no cover - fallback for older JAX
    from jax.core import ClosedJaxpr

from .transformer import compiled_transformer, expression_transformer


class AbstractPipeline(ABC):
    """Abstract base class for data transformation pipelines.

    Derived classes must implement `build_pipeline`, `build_expression`, and
    `apply`. These helpers build the pipeline, assemble it into a pure
    function, and apply it to input data.

    Args:
        cfg (dict): Configuration dictionary defining the pipeline.
        transformers (list[Callable[..., Any]]): Transformers that will be
            registered with the pipeline.
    """

    def __init__(
        self,
        cfg: dict,
        transformers: list[Callable[..., Any]],
    ) -> None:
        self.config = cfg
        self._pipeline: list[Callable[..., Any]] = []
        self._names: list[str] = []
        self.transformers: dict[str, Callable[..., Any]] = {}
        self.expression: Callable[..., Any] | None = None
        self.compiled_expression: Callable[..., Any] | None = None

        for transformer in transformers:
            self.register_transformer(transformer)

        self.assemble()

    def assemble(self) -> None:
        """Assemble the pipeline into a self-contained function."""
        self.build_pipeline()
        self.build_expression()

    @property
    def pipeline(self) -> dict[str, Callable[..., Any]]:
        """Return the registered pipeline elements as a
            dictionary of name: function pairs.

        Returns:
            dict[str, Callable[..., Any]]: Mapping from name to function.
        """
        return dict(zip(self._names, self._pipeline))

    def register_transformer(self, cls: Callable[..., Any]) -> None:
        """Register a transformer function for later use to
        make it available to the calling pipeline object.

            Note:
            The registered function must be a pure functional
            function in order to be transformable with jax.
            The registered transformers are used to build a pipeline.

        Args:
            cls (Callable[..., Any]): Function to register.

        Raises:
            ValueError:
                When a transformer with the same name is already present.
        """
        if cls.__name__ in self.transformers:
            raise ValueError("A transformer of this name is already present")
        self.transformers[cls.__name__] = cls

    def get_jaxpr(
        self,
        *args: Any,
        static_args: Optional[Sequence[int]] = None,
    ) -> Callable[..., Any] | ClosedJaxpr:
        """Return a JAX intermediate expression for the pipeline.

            Note:
            Please note that this only works with static positional arguments:
            JAX does currently not provide a way to have static keyword
            arguments when creating a jaxpr and not a jitted function.
            You can use `partial` to fix keyword arguments before calling this
            method.

        Args:
            *args (Any): Positional arguments forwarded to the expression
                whose intermediate representation should be produced.
            static_args (Optional[Sequence[int]], optional): Static positional
                indices forwarded to ``jax.make_jaxpr`` via
                ``static_argnums``. Defaults to ``None``.

        Returns:
            ClosedJaxpr: When ``*args`` is provided.
            Callable[..., Any]: When ``*args`` is empty.
        """
        static_args = static_args or []
        return expression_transformer(
            *args,
            static_args=static_args,
        )(self.expression)

    def compile_expression(
        self,
        static_args: Optional[Sequence[int]] = None,
        static_kwargs: Optional[Sequence[str]] = None,
    ) -> Callable[..., Any]:
        """Compile the pipeline expression using ``jax.jit``.

        Args:
            static_args (Optional[Sequence[int]], optional): Positional indices
                forwarded to ``jit`` as ``static_argnums``. Defaults to
                ``None``.
            static_kwargs (Optional[Sequence[str]], optional): Keyword names
                forwarded to ``jit`` as ``static_argnames``. Defaults to
                ``None``.

        Raises:
            RuntimeError: When compilation fails.

        Returns:
            Callable[..., Any]: Compiled pipeline function.
        """
        static_args = static_args or ()
        static_kwargs = static_kwargs or ()
        try:
            compiled = jit(
                self.expression,
                static_argnums=static_args,
                static_argnames=static_kwargs,
            )
        except Exception as e:
            raise RuntimeError("Expression compilation failed") from e

        self.compiled_expression = compiled
        return compiled

    def compile_element(
        self,
        name: str,
        static_args: Optional[Sequence[int]] = None,
        static_kwargs: Optional[Sequence[str]] = None,
    ) -> Callable[..., Any]:
        """Compile a specific pipeline element using ``jax.jit``.

        Args:
            name (str): Name of the element to compile.
            static_args (Optional[Sequence[int]], optional): Positional indices
                forwarded to ``jit`` as ``static_argnums``. Defaults to
                ``None``.
            static_kwargs (Optional[Sequence[str]], optional): Keyword names
                forwarded to ``jit`` as ``static_argnames``. Defaults to
                ``None``.

        Raises:
            RuntimeError: When compilation of the element fails.

        Returns:
            Callable[..., Any]: The compiled transformer.
        """
        static_args = static_args or ()
        static_kwargs = static_kwargs or ()
        try:
            compiled = compiled_transformer(
                static_args=static_args, static_kwargs=static_kwargs
            )(self.pipeline[name])
        except Exception as e:
            raise RuntimeError(f"Compilation of element '{name}' failed") from e
        return compiled

    def get_jaxpr_for_element(
        self,
        name: str,
        *args: Any,
        static_args: Optional[Sequence[int]] = None,
    ) -> Callable[..., Any] | ClosedJaxpr:
        """
        get_jaxpr_for_element Create a jax intermediate expression for a given
        element of the pipeline named 'name' with static arguments 'static_args
        and arguments *args. If no arguments are provided, a function is
        returned which will return the intermediate representation once it is
        called with arguments.

        Args:
            name (str): Name of the element to inspect.
            *args (Any): Positional arguments forwarded to the element.
            static_args (Optional[Sequence[int]], optional): Static positional
                indices forwarded to ``expression_transformer``. Defaults to
                ``None``.

        Raises:
            RuntimeError: When the expression cannot be created.

        Returns:
            ClosedJaxpr: When ``*args`` is provided.
            Callable[..., Any]: When ``*args`` is empty.
        """
        static_args = static_args or []
        try:
            expr = expression_transformer(*args, static_args=static_args)(
                self.pipeline[name]
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot create intermediate expression for '{name}'"
            ) from e
        return expr

    @abstractmethod
    def build_pipeline(self):
        pass

    @abstractmethod
    def build_expression(self):
        pass

    @abstractmethod
    def apply(self):
        pass
