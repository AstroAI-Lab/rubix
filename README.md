# Welcome to rubix

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ufuk-cakir/rubix/ci.yml?branch=main)](https://github.com/ufuk-cakir/rubix/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/rubix/badge/)](https://rubix.readthedocs.io/)
[![codecov](https://codecov.io/gh/ufuk-cakir/rubix/branch/main/graph/badge.svg)](https://codecov.io/gh/ufuk-cakir/rubix)
[![All Contributors](https://img.shields.io/github/all-contributors/ufuk-cakir/rubix?color=ee8449&style=flat-square)](#contributors)

## Installation

The Python package `rubix` can be downloades from git and can be installed:

```
git clone https://github.com/ufuk-cakir/rubix
cd rubix
pip install .
```

## Development installation

If you want to contribute to the development of `rubix`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/ufuk-cakir/rubix
cd rubix
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

This project depends on [jax](https://github.com/google/jax). It only installed for cpu computations with the testing dependencies. For installation instructions with gpu support,
please refer to [here](https://github.com/google/jax?tab=readme-ov-file#installation).


## Documentation
Sphinx Documentation of all the functions is currently available under [this link](https://astro-rubix.web.app/).

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).

