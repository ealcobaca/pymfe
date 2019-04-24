# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys
import sphinx_rtd_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../sphinxext'))
from github_link import make_linkcode_resolve
# import sphinx_gallery


# -- Project information -----------------------------------------------------

project = 'pymfe'
copyright = '2019, Edesio Alcobaça, Felipe Alves Siqueira and Luis Paulo Faina Garcia'
author = 'Edesio Alcobaça, Felipe Alves Siqueira and Luis Paulo Faina Garcia'

# The full version, including alpha/beta/rc tags
release = '0.0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ['_build', '_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# see: https://github.com/phn/pytpm/issues/3#issuecomment-12133978
# see https://github.com/numpy/numpydoc/issues/69
# numpydoc_show_class_members = False


# generate autosummary even if no references
autosummary_generate = True

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve('pymfe',
                                         u'https://github.com/ealcobaca/'
                                         'pymfe/blob/{revision}/'
                                         '{package}/{path}#L{lineno}')

print(linkcode_resolve('py', {'module': 'pymfe.mfe', 'fullname': 'MFE'}))


# The master toctree document.
master_doc = 'index'
