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
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'KXY (Lean AutoML, As A Service)'
copyright = '2021, KXY Technologies, Inc'
author = 'Dr. Yves-Laurent Kom Samo'
version = 'latest'
autodoc_inherit_docstrings = False


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', \
    'sphinx.ext.todo', 'sphinx.ext.githubpages', 'sphinxcontrib.bibtex', \
    'sphinx.ext.mathjax', 'sphinx.ext.autosectionlabel', 'nbsphinx', \
    'sphinx_copybutton', 'sphinxcontrib.googleanalytics', 'sphinx_sitemap', \
    'sphinxcontrib.httpdomain']

# imgmath_image_format = 'svg'
# imgmath_font_size = 13

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
html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    'display_github': True,
    # Set the following variables to generate the resulting github URL for each page. 
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    'github_user': 'kxytechnologies',
    'github_repo': 'kxy-python',
    'github_version': 'master/docs/' 
}

html_theme = 'sphinx_rtd_theme'
html_logo = 'images/logo.png'
html_favicon = 'images/favicon.png'
html_theme_options = {'logo_only': True, 'style_nav_header_background': '#2c3d5e'}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Notebook
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_output_prompt = 'Out[%s]:'
source_suffix = ['.rst', '.md', '.ipynb']

# Google Analytics
googleanalytics_id = 'UA-167632834-2'
googleanalytics_enabled = True


# Sitemap
html_baseurl = 'https://www.kxy.ai/reference/'
html_title = 'The KXY Platform: Lean AutoML, As A Service'
# html_extra_path = ['robots.txt']

