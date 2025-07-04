![Rubix Logo](./logo_rubix.png)

# Welcome to RUBIX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ufuk-cakir/rubix/ci.yml?branch=main)](https://github.com/ufuk-cakir/rubix/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/rubix/badge/)](https://rubix.readthedocs.io/)
[![codecov](https://codecov.io/gh/ufuk-cakir/rubix/branch/main/graph/badge.svg)](https://codecov.io/gh/ufuk-cakir/rubix)
[![All Contributors](https://img.shields.io/github/all-contributors/ufuk-cakir/rubix?color=ee8449&style=flat-square)](#contributors)

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


### 1. File your issue

If you find a bug or think of an enhancement, please open an issue on GitHub. For example, you might write an issue like:

- **Title:** Fix incorrect galaxy rotation calculation  
- **Description:**  
  The galaxy rotation function (rotate_galaxy) does not properly convert angle inputs, causing unexpected behavior when non-scalar JAX arrays are passed. Please investigate and fix this conversion so that it accepts a Python float.

### 2. Create a branch for your issue

After creating the issue, create a new branch from `main` following a clear naming convention. For example:

```bash
git checkout -b fix/rotate-galaxy-angle
```

Work on your changes in this branch. Make sure to write tests and update documentation if necessary.

### 3. Submit a pull request

Once your changes pass all tests locally and the branch is up to date with `main`, create a pull request (PR) on GitHub. Describe the problem, your approach, and link the original issue so that the issue is automatically closed upon merge.

### 4. Merge and get recognition

After your PR is reviewed and merged into `main`, your contributions will be recognized automatically. Thanks to our All Contributors setup, a bot or a maintainer will add you to the contributors list in the README file. You'll then appear in the All Contributors section below.

Thank you for helping improve `rubix`!

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).


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
