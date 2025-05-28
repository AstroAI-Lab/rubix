from abc import ABC, abstractmethod

from jax import jit

from .transformer import compiled_transformer, expression_transformer


class AbstractPipeline(ABC):
    """
    AbstractPipeline Abstract baseclass for data transformation pipelines.
    Provides methods `build_pipeline`, `build_expression` and `apply` which
    must be implemented by every derived class and
    which are responsible for building up the pipeline, for assembling it into
    a self-contained pure functional function and for applyin the latter to
    data, respectively.
    """

    def __init__(self, cfg: dict, transformers: list):
        """
        __init__ _Create a new pipeline. This should only be called in derived
                classes' __init__ methods.

        Args:
            cfg (dict): Read config file defining the pipeline
            transformers (list): Transformer functions to use
        """
        self.config = cfg
        self._pipeline = []
        self._names = []
        self.transformers = {}
        self.expression = None
        self.compiled_expression = None

        for t in transformers:
            self.register_transformer(t)

        self.assemble()

    def assemble(self):
        """
        assemble Assemble the pipeline into a self-contained function with the
        same signature as the pipeline's first element. Can only run if all
        functions that make up the pipeline are registered with it by calling
        `register_transformer`.

        Raises
        ------
        RuntimeError
            When no transformers are registered to build the pipeline out of.
        """
        self.build_pipeline()
        self.build_expression()

    @property
    def pipeline(self) -> dict:
        """
        pipeline Get the sequence of functions that make up the pipeline as a
        dictionary of name: function pairs.

        Returns:
            Description of the pipeline as name: function pairs as dict.
        """
        return dict(zip(self._names, self._pipeline))

    def register_transformer(self, cls):
        """
        register_transformer Make a functtion available to the calling
        pipeline object. The registered function must be a pure functional
        function in order to be transformable with jax. The registered transformers
        are used to build a pipeline.

        Args:
            cls: function object to register.

        Raises
        ------
        ValueError
            When the function is already registered  with the pipeline
        """
        if cls.__name__ in self.transformers:
            raise ValueError("A transformer of this name is already present")
        self.transformers[cls.__name__] = cls

    def get_jaxpr(self, *args, static_args: list = []):
        """
        get_jaxpr Get a jax intermediate expression for the function that
        represents an application of this pipeline to input data.
        Please note that this only works with tatic positional arguments: JAX does currently not provide a way to have static keyword arguments when creating a jaxpr and not a jited function.
        You can use `partial` to fix keyword arguments before calling this method.

        Parameters
        ----------
        static_args : list, optional
            Static argument indices. Will be forwarded to the static_argnums
            argument of jax.make_jaxpr, by default []

        Returns
        -------
        jax.ClosedJaxpr
            If *args is not empty: A jax intermediate representation that
            results from applying the calling pipeline to the provided arguments.
        Callable
            if *args is empty. A function that will result in a jax
            intermediate expression if called with desired arguments.
        """
        return expression_transformer(*args, static_args=static_args)(self.expression)

    def compile_expression(self, static_args=[], static_kwargs=[]):
        """
        compile_expression Compile the function that represents an application
                            of this pipeline to input data using jax jit.

        Parameters
        ----------
        static_args : list, optional
            static poisitional arguments that should not be traced by jit, by default []
        static_kwargs : list, optional
            statiuc keyword arguments that should not be traced by jit, by default []

        Returns:
            Compiled pipeline function as PjitFunction
        """
        f = None

        try:
            f = jit(
                self.expression,
                static_argnums=static_args,
                static_argnames=static_kwargs,
            )
        except Exception as e:
            raise RuntimeError("Expression compilation failed") from e

        self.compiled_expression = f

        return f

    def compile_element(self, name: str, static_args=[], static_kwargs=[]):
        """
        compile_element Compile an element of the pipeline named 'name' with
        the jax jit with the provided static_args and static kwargs.

        Parameters
        ----------
        name : str
            Name of the element to be compiled
        static_args : list, optional
            static positional argument indices. Will be forwarded to the jit
            static_argnums argument., by default []
        static_kwargs : list, optional
            Names of the static keyword arguments. Will be forwarded to the
            jit static_argnames argument, by default []

        Returns
        -------
        _type_
            _description_
        """
        f = None
        try:
            f = compiled_transformer(
                static_args=static_args, static_kwargs=static_kwargs
            )(self.pipeline[name])
        except Exception as e:
            raise RuntimeError(f"Compilation of element '{name}' failed") from e
        return f

    def get_jaxpr_for_element(self, name: str, *args, static_args: list = []):
        """
        get_jaxpr_for_element Create a jax intermediate expression for a given
        element of the pipeline named 'name' with static arguments 'static_args
        and arguments *args. If no arguments are provided, a function is
        returned which will return the intermediate representation once it is
        called with arguments.

        Parameters
        ----------
        name : str
            Name of the element to be retrieved
        static_args : list, optional
            static positional argument indices, by default []

        Returns
        -------
        jax.ClosedJaxpr
            If *args is not empty: Intermediate expression respresenting the
            computation that is carried out when calling the element with the
            given arguments.
        Callable
            If *args is empty: Function that returns a jax.ClosedJaxpr once
            called with appropriate arguments.
        """
        expr = None
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
