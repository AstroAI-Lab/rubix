import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Rubix"
copyright = "2025, Anna Lena Schaible, Ufuk Cakir, Tobias Buck"
author = "Anna Lena Schaible, Ufuk Cakir, Tobias Buck"
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

nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# Optional: You can add theme options to customize your documentation
html_theme_options = {
    "repository_url": "https://github.com/AstroAI-Lab/rubix",
    "use_repository_button": True,
}

html_static_path = ["_static"]
html_logo = "../build/html/_static/logo_rubix.png"
