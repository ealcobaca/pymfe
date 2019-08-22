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
import sphinx_gallery


# -- Project information -----------------------------------------------------
# General information about the project.

project = "pymfe"
copyright = "2019, E. Alcobaça, F. Siqueira and L. P. F. Garcia"
author = "E. Alcobaça, F. Siqueira and L. P. F. Garcia"

# The full version, including alpha/beta/rc tags.
from pymfe import __version__
release = __version__


# -- General configuration ---------------------------------------------------

# The master toctree document.
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
]

# see: https://github.com/phn/pytpm/issues/3#issuecomment-12133978
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates']
# exclude_patterns = []

autodoc_default_options = {'members': True, 'inherited-members': True}
# autodoc_default_flags = ['members', 'inherited-members']

# generate autosummary even if no references
autosummary_generate = True



# -- Options for HTML output -------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'pymfe-docs'


# -- Options for LaTeX output ---------------------------------------------

# The master toctree document.
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     "index", "pymfe_docs.tex", "pymfe Documentation",
#     "E. Alcobaça, F. Siqueira and L. Garcia",
#     "manual"
#     ]
#

# -- Options for manual page output ---------------------------------------

# If false, no module index is generated.
# latex_domain_indices = True


# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
# man_pages = [
#     ("index", "pymfe", "pymfe Documentation",
#      ["E. Alcobaça, F. Siqueira and L. P. F. Garcia"], 1)
# ]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
# texinfo_documents = [
#     ("index", "pymfe", "pymfe Documentation",
#      "E. Alcobaça, F. Siqueira and L. P. F. Garcia", 'pymfe',
#      "Toolbox for meta-feature extraction of datasets in machine learning.",
#      "Miscellaneous")
# ]


# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'arff': ('https://pythonhosted.org/liac-arff/', None)
}

# ------ sphinx gallery ----------------------------------------------------

sphinx_gallery_conf = {
    'doc_module': 'pymfe',
    'backreferences_dir': os.path.join('generated'),
    # path to your example scripts
    'examples_dirs': '../../examples',
    # path where to save gallery generated examples
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        # The module you locally document uses None
        'pymfe': None,
    }
    # 'plot_gallery': True,
    # 'junit': '../test-results/sphinx-gallery/junit.xml',
    # 'log_level': {'backreference_missing': 'warning'},
    # 'subsection_order': ExplicitOrder(['../examples/sin_func',
    #                                    '../examples/no_output',
    #                                    '../tutorials/seaborn'])
}



# --- GitHun link -----------------------------------------------------------

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve('pymfe',
                                         u'https://github.com/ealcobaca/'
                                         'pymfe/blob/{revision}/'
                                         '{package}/{path}#L{lineno}')
# print(linkcode_resolve('py', {'module': 'pymfe.mfe', 'fullname': 'MFE'}))
