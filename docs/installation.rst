Installation
============

`RUBIX` can be installed via `pip`

Clone the repository and navigate to the root directory of the repository. Then run

```
pip install .[cpu]
```

If you want to contribute to the development of `RUBIX`, we recommend the following editable installation from this repository:

```
git clone https://github.com/AstroAI-Lab/rubix
cd rubix
pip install -e .[cpu,tests,dev]
```
Having done so, the test suit can be run unsing `pytest`:

```
python -m pytest
```

Note that if `JAX` is not yet installed, with the `cpu` option only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) or use the option `cuda`.

Get started with this simple example notebooks/rubix_pipeline_single_function_shard_map.ipynb.

Configuration
=============

When you run the pipeline you provide a configuration dict that references the files in `rubix/config/`. The following sections are required for the default pipelines:

- `pipeline.name`: Choose one of `calc_ifu`, `calc_dusty_ifu`, or another entry from `pipeline_config.yml`.
- `galaxy`: Must include `dist_z` and a `rotation` block (`type` or explicit `alpha`, `beta`, `gamma`).
- `telescope`: Needs `name`, a `psf` block (Gaussian kernel with both `size` and `sigma`), an `lsf` block with `sigma`, and `noise` containing `signal_to_noise` plus a `noise_distribution` (`normal` or `uniform`).
- `ssp.dust`: Declares `extinction_model` and `Rv` before the dusty pipeline can produce an extincted datacube.
- `data.args.particle_type`: Must include `"stars"` (add `"gas"` if you rely on the optional gas branch) so the filtering/rotation steps know which components to process.

The telescopes in `rubix/telescope` currently only support square pixels, so every config should set `pixel_type: square` in the relevant telescope definition.
