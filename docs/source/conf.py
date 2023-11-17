# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

from mzbsuite import __version__ as mzb_version
from mzbsuite import __author__ as mzb_author
from mzbsuite import __project_name__ as mzb_project_name
from mzbsuite import __date__ as mzb_date

# -- Project information -----------------------------------------------------

project = mzb_project_name
date = mzb_date
author = mzb_author
version = mzb_version
release = f"{version} alpha"
copyright = f"{date} {author}"
language = "en"


# -- General configuration ---------------------------------------------------

source_suffix = ".rst"
exclude_patterns = []
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx_rtd_theme",
    "sphinx.ext.autosectionlabel", 
    "sphinx.ext.napoleon", 
    "nbsphinx",
]

# Automatic labels for sections etc, prefixed for each file for each file
# NOTE: cannot have the same section name within the same file!
autosectionlabel_prefix_document = True

# Napoleon is for formatting NumPy and Google style docstring

# Jupyter notebooks are handled by nbsphinx
nbsphinx_allow_errors = True # this allows errors in .ipynb file executions
nbsphinx_execute = 'never' # don't execute notebooks at each build

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "../assets/mzbsuite_logo_v2.1.svg"

html_theme_options = {
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    # 'analytics_anonymize_ip': False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # "vcs_pageview_mode": "",
    # "style_nav_header_background": "white",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
}
