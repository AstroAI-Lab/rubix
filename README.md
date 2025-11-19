<p align="center">
  <img src="./logo_rubix.png" alt="Rubix Logo" width="30%">
</p>

# Welcome to RUBIX

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/AstroAI-Lab/rubix/blob/main/docs/CONTRIBUTING.md)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/AstroAI-Lab/rubix/ci.yml?branch=main)](https://github.com/AstroAI-Lab/rubix/actions/workflows/ci.yml)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/AstroAI-Lab/rubix/CI?label=build)](https://github.com/AstroAI-Lab/rubix/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/rubix/badge/)](https://astro-rubix.web.app)
[![codecov](https://codecov.io/gh/AstroAI-Lab/rubix/branch/main/graph/badge.svg)](https://codecov.io/gh/AstroAI-Lab/rubix)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![All Contributors](https://img.shields.io/github/all-contributors/AstroAI-Lab/rubix?color=ee8449&style=flat-square)](#contributors)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checking](https://github.com/beartype/beartype-assets/blob/main/badge/bear-ified.svg)](https://github.com/beartype/beartype)

RUBIX is a versatile Integral Field Unit (IFU) tool designed for astrophysical simulations. It transforms any particle based galaxy model (e.g. cosmological hydrodynamical simulation outputs) into realistic mock IFU cubes, enabling both forward and inverse modeling. Built on JAX, RUBIX leverages GPU acceleration and automatic differentiation, allowing users to perform gradient-based optimization for inverse modeling alongside traditional forward modeling.

Key features include:
- **Mock IFU Cube Generation:** Convert simulation data into realistic IFU cubes.
- **GPU-Accelerated Computations:** Built on JAX for high-performance GPU support.
- **Gradient-Based Inverse Modeling:** Utilize gradients for efficient inverse modeling techniques.
- **Flexible and Extensible:** Designed to easily integrate with existing pipelines and astrophysical analysis tools.

## Installation

The Python package `rubix` can be downloades from git and can be installed:

```
git clone https://github.com/AstroAI-Lab/rubix.git
cd rubix
pip install .
```

## Development installation

If you want to contribute to the development of `rubix`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/AstroAI-Lab/rubix.git
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

## Contribution

Contributions to `rubix` are welcome and greatly appreciated!
Whether you're fixing bugs, improving documentation, or suggesting new features, your help is valuable to us.

Please see [here](docs/source/CONTRIBUTING.md) for contribution guidelines.

Thank you for helping improve `rubix`!

## Citation & Acknowledgement

Please cite **both** of the following papers ([Cakir et al. 2024](https://arxiv.org/abs/2412.08265), [Schaible et al. 2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250615811R/abstract)) if you use Rubix in your research:

    @ARTICLE{2024arXiv241208265C,
       author = {{{\c{C}}ak{\i}r}, Ufuk and {Schaible}, Anna Lena and {Buck}, Tobias},
        title = "{Fast GPU-Powered and Auto-Differentiable Forward Modeling of IFU Data Cubes}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies, Physics - Computational Physics, Physics - Data Analysis, Statistics and Probability},
         year = 2024,
        month = dec,
          eid = {arXiv:2412.08265},
        pages = {arXiv:2412.08265},
          doi = {10.48550/arXiv.2412.08265},
archivePrefix = {arXiv},
       eprint = {2412.08265},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241208265C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}





## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://cakir-ufuk.de/"><img src="https://avatars.githubusercontent.com/u/92611643?v=4?s=100" width="100px;" alt="Ufuk Çakır"/><br /><sub><b>Ufuk Çakır</b></sub></a><br /><a href="#code-ufuk-cakir" title="Code">💻</a> <a href="#content-ufuk-cakir" title="Content">🖋</a> <a href="#data-ufuk-cakir" title="Data">🔣</a> <a href="#doc-ufuk-cakir" title="Documentation">📖</a> <a href="#design-ufuk-cakir" title="Design">🎨</a> <a href="#example-ufuk-cakir" title="Examples">💡</a> <a href="#ideas-ufuk-cakir" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-ufuk-cakir" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-ufuk-cakir" title="Maintenance">🚧</a> <a href="#plugin-ufuk-cakir" title="Plugin/utility libraries">🔌</a> <a href="#projectManagement-ufuk-cakir" title="Project Management">📆</a> <a href="#question-ufuk-cakir" title="Answering Questions">💬</a> <a href="#research-ufuk-cakir" title="Research">🔬</a> <a href="#review-ufuk-cakir" title="Reviewed Pull Requests">👀</a> <a href="#tool-ufuk-cakir" title="Tools">🔧</a> <a href="#test-ufuk-cakir" title="Tests">⚠️</a> <a href="#talk-ufuk-cakir" title="Talks">📢</a> <a href="#userTesting-ufuk-cakir" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/anschaible"><img src="https://avatars.githubusercontent.com/u/131476730?v=4?s=100" width="100px;" alt="anschaible"/><br /><sub><b>anschaible</b></sub></a><br /><a href="#code-anschaible" title="Code">💻</a> <a href="#content-anschaible" title="Content">🖋</a> <a href="#data-anschaible" title="Data">🔣</a> <a href="#doc-anschaible" title="Documentation">📖</a> <a href="#design-anschaible" title="Design">🎨</a> <a href="#example-anschaible" title="Examples">💡</a> <a href="#ideas-anschaible" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-anschaible" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-anschaible" title="Maintenance">🚧</a> <a href="#plugin-anschaible" title="Plugin/utility libraries">🔌</a> <a href="#projectManagement-anschaible" title="Project Management">📆</a> <a href="#question-anschaible" title="Answering Questions">💬</a> <a href="#research-anschaible" title="Research">🔬</a> <a href="#review-anschaible" title="Reviewed Pull Requests">👀</a> <a href="#tool-anschaible" title="Tools">🔧</a> <a href="#test-anschaible" title="Tests">⚠️</a> <a href="#talk-anschaible" title="Talks">📢</a> <a href="#userTesting-anschaible" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://tobibu.github.io"><img src="https://avatars.githubusercontent.com/u/7574273?v=4?s=100" width="100px;" alt="Tobias Buck"/><br /><sub><b>Tobias Buck</b></sub></a><br /><a href="#code-TobiBu" title="Code">💻</a> <a href="#content-TobiBu" title="Content">🖋</a> <a href="#data-TobiBu" title="Data">🔣</a> <a href="#doc-TobiBu" title="Documentation">📖</a> <a href="#design-TobiBu" title="Design">🎨</a> <a href="#example-TobiBu" title="Examples">💡</a> <a href="#ideas-TobiBu" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-TobiBu" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-TobiBu" title="Maintenance">🚧</a> <a href="#plugin-TobiBu" title="Plugin/utility libraries">🔌</a> <a href="#projectManagement-TobiBu" title="Project Management">📆</a> <a href="#question-TobiBu" title="Answering Questions">💬</a> <a href="#research-TobiBu" title="Research">🔬</a> <a href="#review-TobiBu" title="Reviewed Pull Requests">👀</a> <a href="#tool-TobiBu" title="Tools">🔧</a> <a href="#test-TobiBu" title="Tests">⚠️</a> <a href="#talk-TobiBu" title="Talks">📢</a> <a href="#userTesting-TobiBu" title="User Testing">📓</a> <a href="#mentoring-TobiBu" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/robin-janssen"><img src="https://avatars.githubusercontent.com/u/82322346?v=4?s=100" width="100px;" alt="Robin Janssen"/><br /><sub><b>Robin Janssen</b></sub></a><br /><a href="#code-robin-janssen" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nihatog"><img src="https://avatars.githubusercontent.com/u/185299085?v=4?s=100" width="100px;" alt="nihatog"/><br /><sub><b>nihatog</b></sub></a><br /><a href="#code-nihatog" title="Code">💻</a> <a href="#doc-nihatog" title="Documentation">📖</a> <a href="#example-nihatog" title="Examples">💡</a> <a href="#research-nihatog" title="Research">🔬</a> <a href="#review-nihatog" title="Reviewed Pull Requests">👀</a> <a href="#test-nihatog" title="Tests">⚠️</a> <a href="#userTesting-nihatog" title="User Testing">📓</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Licence

[GNU General Public License v3.0](https://github.com/synthesizer-project/synthesizer/blob/main/LICENSE.md)

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
