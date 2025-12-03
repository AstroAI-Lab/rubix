Installation
============

`RUBIX` can be installed via `pip`

Clone the repository and navigate to the root directory of the repository. Then run

```
pip install .
```

If you want to contribute to the development of `RUBIX`, we recommend the following editable installation from this repository:

```
git clone https://github.com/AstroAI-Lab/rubix.git
cd rubix
pip install -e .
```
Having done so, the test suit can be run unsing `pytest`:

```
python -m pytest
```

Note that if `JAX` is not yet installed, only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

Note that if you want to use the FSPS stellar template, please follow the installation instructions
provided in the [FSPS documentation](https://python-fsps.readthedocs.io/en/latest/installation/.html).

Get started with this simple example notebooks/rubix_pipeline_single_function.ipynb.
