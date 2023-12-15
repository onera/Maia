# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Maia'
copyright = '2021, ONERA The French Aerospace Lab'
author = 'ONERA'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.graphviz",
              "sphinx.ext.autodoc", 
              "sphinx.ext.autosummary",
              "sphinx.ext.napoleon"]

add_module_names = False #Shorten function names
autodoc_typehints = 'none' #Hide typehints in doc

# -- Napoleon extension settings
napoleon_use_rtype = False  # Don't add a line for return type

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
#html_css_files = [
#    'custom.css',
#]

html_style = 'css/read_the_docs_custom.css'

# The name of the Pygments (syntax highlighting) style to use.
# list with >>> from pygments.styles import STYLE_MAP; STYLE_MAP.keys()
#pygments_style = 'monokai'


html_theme_options = {
  'navigation_depth': 6,
}

# Graphviz config
graphviz_output_format = "svg"

rst_prolog = """
.. role:: cpp(code)
   :language: c++

.. role:: cgns

.. role:: mono

.. role:: def

.. role:: titlelike
"""

# Generate cgns example files, some will be downloadable
import subprocess
subprocess.run(["../scripts/maia_yaml_examples_to_hdf5", "../share/_generated"], stdout=subprocess.DEVNULL)

############################
# SETUP THE RTD LOWER-LEFT #
############################
# See https://github.com/maltfield/rtd-github-pages for details
try:
   html_context
except NameError:
   html_context = dict()
html_context['display_lower_left'] = True

REPO_NAME = 'mesh/maia' #Namespace in the gitlab pages server
current_language = 'en'
current_version = 'dev'
 
# tell the theme which language to we're currently building
html_context['current_language'] = current_language
# tell the theme which version we're currently on ('current_version' affects
# the lower-left rtd menu and 'version' affects the logo-area version)
html_context['current_version'] = current_version
if current_version == 'dev':
  html_context['version'] = current_version
else:
  html_context['version'] = f'v{current_version}'
 
# POPULATE LINKS TO OTHER LANGUAGES
html_context['languages'] = list()
for lang in []:
   html_context['languages'].append( (lang, f'/{REPO_NAME}/{lang}/{current_version}/') )
 
# POPULATE LINKS TO OTHER VERSIONS
html_context['versions'] = list()
for version in ['dev', '1.2', '1.1', '1.0']:
   html_context['versions'].append( (version, f'/{REPO_NAME}/{version}/') )

# POPULATE LINKS TO OTHER FORMATS/DOWNLOADS
html_context['downloads'] = list()
