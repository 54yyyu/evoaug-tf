# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Configuration file for the Sphinx documentation builder.
project = 'EvoAug-TF'
copyright = '2024, Yiyang Yu'
author = 'Yiyang Yu'

release = '1.0.3'
version = '1.0.3'

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
autoapi_dirs = ['../evoaug_tf']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'tensorflow': ('https://www.tensorflow.org/versions/r2.11/api_docs/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

#root_doc = 'indexs'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'
