Installation
============

`RUBIX` can be installed via `pip`

Clone the repository and navigate to the root directory of the repository. Then run

```
pip install .
```

If you want to contribute to the development of `RUBIX`, we recommend the following editable installation from this repository:

```
git clone https://github.com/ufuk-cakir/rubix
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

To habe support of the FSPS stellar population library, you need to install the
`fsps` package. This can be done following the instruction [here](https://python-fsps.readthedocs.io/en/latest/installation/)
If you do not have `fsps` installed, you can still use the `RUBIX` pipeline, but you will not be able to use the FSPS template.
In this case you have to set your SPS_HOME variable to a dummy path, e.g. `os.environ['SPS_HOME'] = /dev/null` before you load your config for the rubix pipeline.

Get started with this simple example notebooks/rubix_pipeline_single_function_shard_map_fits.ipynb.
