import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rubix"
copyright = "2024, Ufuk, Anna Lena, Tobias"
author = "Ufuk, Anna Lena, Tobias"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",  # Add this to enable Jupyter Notebook support
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# Optional: You can add theme options to customize your documentation
html_theme_options = {
    "repository_url": "https://github.com/your-username/your-repo",
    "use_repository_button": True,
}

html_static_path = ["_static"]
