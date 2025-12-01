# Contributing

Please feel free to submit issues and pull requests to this repository.
The GitHub workflow will automatically run [Black](https://black.readthedocs.io/en/stable/) and (isort)[https://pycqa.github.io/isort/] as well as (pydoclint)[https://pypi.org/project/pydoclint/] on any contributions; builds that fail these tests will not be accepted. Further notes on code style are detailed below.


**Contents:**

- [Setting up your development environment](#setting-up-your-development-environment)
- [Using Black, isort and pydoclint](#using-black)
- [Style guide](#style-guide)
- [Development Documentation](#development-documentation)
- [Contributing to the Documentation](#contributing-to-the-documentation)
- [Getting set up](#getting-set-up)
- [Adding notebooks](#adding-notebooks)
- [Adding example scripts](#adding-example-scripts)
- [Summary of how to contribute](#summary-of-how-to-contribute)

## Settting up your development environment

To begin developing in RUBIX first set up a new environment.

```bash
python3 -m venv rubix-dev-env
source rubix-dev-env/bin/activate
```

You can then clone the repo and install it in editable mode with the extra development dependencies. This project depends on [jax](https://github.com/google/jax) which comes in two flavours `cpu` and `cuda` so at install time you have to choose if you want to run on CPUs or GPUs.

```bash
git clone https://github.com/AstroAI-Lab/rubix
cd rubix
pip install -e .[cpu,dev]
```

Note: if you are planning to use RUBIX on the GPU you need to replace the `cpu` dependency with the corresponding `cuda` dependency. If you additionally plan to build the docs locally you'll also need to include the `docs` dependency group.

### Setting up pre-commit hooks

Once you have developed your new functionality, you'll want to commit it to the repo. We employ a pre-commit hook to ensure any code you commit will pass our tests and you won't be stuck with a failing Pull Request. This pre-commit hook will guard against files containing merge conflict strings, guard against the committing of large files, sanitise Jupyter notebooks (using `nbstripout`), check the yaml files, and, most importantly, will run `black`, `isort` and `pydoclint`.

This requires a small amount of set-up on your part, some of which was done when you installed the optional development dependencies above. The rest of the setup requires you run

```bash
pre-commit install
```

at the root of the repo to activate the pre-commit hooks.

If you would like to test whether it works you can run `pre-commit run --all-files` to run the pre-commit hook on the whole repo. You should see each stage complete without issue in a clean clone.


## Using Black

We use [Black](https://black.readthedocs.io/en/stable/) for code formatting. Assuming you installed the development dependencies (if not you can install `black` with pip: `pip install black`), you can run the linting with `black {source_file_or_directory}`. For more details see the [Black documentation](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html).

The `Black` configuration is defined in our `pyproject.toml` so there's no need to configure it yourself, we've made all the decisions for you (for better or worse). Any merge request will be checked with `Black` and must pass before being eligable to merge.

In addition we run `isort` to sort imports alphabetically and automatically separate them  into sections and by import type.
For more information see the [isort documentation](https://pycqa.github.io/isort/).

Similarly, we use `pydoclint` for docstring linting. [`pydoclint`](https://pypi.org/project/pydoclint/) checks whether a docstring's sections (arguments, returns, raises, ...) match the function signature or function implementation. We adhere to the [Google docstring format](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings). See also next section for more details.

Note that using the pre-commit hook will mean all of this is done automatically for you.

## Style guide

All new PRs should follow these guidelines. We adhere to the PEP-8 style guide, and as described above this is verified with `Black`.

We use the [Google docstring format](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
In addition, we require type annotations using [jaxtyping](https://docs.kidger.site/jaxtyping/) which will be runtime typechecked via [beratype](https://beartype.readthedocs.io/en/latest/?badge=latest).

Some specific examples of common style issues:

- Do not use capitalised single letters for attributes. For example, `T` could be transmission or temperature. Instead, one should write out the full word.
- We use `get_` and/or `calculate_` nomenclature for methods that perform a calculation and return the result to the user.
- Variables should adhere to `snake_case` style while class names should be in `PascalCase`.
- Block comments should have their first letter capitalised, i.e.

```
# This is a block comment
x = y
```

- While inline comments should be preceded by two whitespaces and start with a lowercase letter, i.e.

```
z = x * 2  # this is an inline comment
```

## Development documentation

The [published documentation](https://astro-rubix.web.app) reflects the current distribution available on PyPI. If you would like to see the current development version in your branch or the main branch, you will have to build the documentation locally. To do so, navigate to the ``docs`` directory and run:

```bash
make clean; make html
```
This will build a local copy of the documentation representative of the currently checked out branch.

## Contributing to the Documentation

The RUBIX documentation is written in a combination of reStructuredText, Jupyter notebooks and Python scripts.
Adding content should be relatively simple if you follow the instructions below.

### Adding notebooks

To add Jupyter notebooks to the documentation:

1. Add your Jupyter notebook to the `notebooks` directory under the `docs` folder. Make sure that you 'Restart Kernel and run all cells' to ensure that the notebook is producing up to date, consistent outputs.
2. Add your notebook to the relevant toctree. See below for an example toctree. Each toctree is contained within a Sphinx `.rst` file in each documentation source directory. The top-level file is `docs/index.rst`. If your file is in a subfolder, you need to update the `.rst` file in that directory.

- If you're creating a new sub-directory of documentation, you will need to carry out a couple more steps:

1.  Create a new `.rst` file in that directory
2.  Update `docs/index.rst` with the path to that `.rst` file
3.  Currently we do not run pytests on jupyter notebooks. So please make sure your notebooks are actually working fine.

Example toctree:

    .. toctree::
       :maxdepth: 2
       :caption: RUBIX documentation

        self
        installation
        versions
        publications
        license
        acknowledgments

### Adding example scripts

The `examples/` top level directory contains a number of self-contained example scripts (Python, `.py`) for particular use cases that may not belong in the main documentation, but are still useful for many users. We use the [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html) extension to build a gallery of our examples in the documentation. A helpful side effect of this is that we can use the examples suite as a further test suite of more advanced use cases (though this requires certain conventions to be followed, see below)

**Important**: If an example is named `plot_*.py`, then `sphinx-gallery` will attempt to run the script and use any images generated in the gallery thumbnail. Images should be generated using `plt.show()` and not saved to disk. If examples are not preceded with `plot_`, then they will **not** be run when compiling the documentation, and no errors will be caught.

Each script (`.py`) should have a top-level docstring written in reST, with a header. Examples that do not will fail the automated build process. Further details are provided [here](https://sphinx-gallery.github.io/stable/syntax.html). For example:

    """
    "This" is my example-script
    ===========================

    This example doesn't do much, it just makes a simple plot
    """

Subfolders of examples should contain a `README.rst` with a section heading (please follow the template in other subfolders).

### Building the Documentation locally

The documentation is build via `Sphinx`. You can build the documentation locally by following these steps:

#### Setup and Installation

First, ensure you have Sphinx installed in your environment:

```
pip install sphinx
```

If you are setting up the documentation source files for the first time, use the quickstart utility (this step is usually skipped if the source files already exist):

```
sphinx-quickstart
```

#### Configuration and Content

The core of the documentation setup resides in the `docs` folder:

- `conf.py`: This is the main configuration file where you define extensions (like myst_nb for notebooks), set the theme, and manage global build settings.

- `index.rst`: This file serves as the main page and table of contents for the entire documentation. You can link all other .rst files, notebooks, and content pages here.

By default, the documentation build process is configured to use pre-rendered outputs embedded within your Jupyter Notebooks (.ipynb files). This is done to significantly speed up the build time by skipping the execution step. If you want to compile the notebooks during the build process, you can set the `nb_execution_mode` variable in the `conf.py` file to `auto` or `force`.

#### Building the Documentation

Once your content and configuration are ready, run the following command from the root of your Sphinx project (where the Makefile is located):

````
make html
````

This command will compile all source files (RST, Notebooks, etc.) and generate the final HTML output files. you can find them in the `build/html` folder.

#### Viewing the Documentation

The generated files will be placed in the `build/html` folder.

To view the documentation, simply open the main file in your web browser:

```
build/html/index.html
```

## Summary of how to contribute
### 1. File your issue

If you find a bug or think of an enhancement, please open an issue on GitHub. For example, you might write an issue like:

- **Title:** Fix incorrect galaxy rotation calculation
- **Description:**
  The galaxy rotation function (rotate_galaxy) does not properly convert angle inputs, causing unexpected behavior when non-scalar JAX arrays are passed. Please investigate and fix this conversion so that it accepts a Python float.

### 2. Create a branch for your issue

After creating the issue, create a new branch from `main` following a clear naming convention - e.g. name it such that the following sentence makes sense: ```If applied, this branch does/adds/ *name-of-branch*.```
For example:

```bash
git checkout -b fix/rotate-galaxy-angle
```

Work on your changes in this branch. Make sure to write tests and update documentation if necessary.

### 3. Submit a pull request

Once your changes pass all tests locally and the branch is up to date with `main`, create a pull request (PR) on GitHub. Describe the problem, your approach, and link the original issue so that the issue is automatically closed upon merge.

### 4. Merge and get recognition

After your PR is reviewed and merged into `main`, your contributions will be recognized automatically. Thanks to our [All Contributors](https://allcontributors.org) setup, a bot or a maintainer will add you to the contributors list in the README file. You'll then appear in the All Contributors section below.
