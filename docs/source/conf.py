# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys

sys.path.append("/home/yiyu/evoaug-tf/")

project = 'EvoAug-TF'
copyright = '2023, Yiyang Yu'
author = 'Yiyang Yu'

release = '0.1'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
#html_title = 'EvoAug-TF'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

autoapi_type = 'python'
autoapi_dirs = ['../../evoaug_tf']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
