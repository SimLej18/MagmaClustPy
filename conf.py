import os
import sys

# Project information
project = 'MagmaClustPy'
author = 'Simon Lejoly'
release = '0.0.1'

# List of extensions
extensions = [
    'sphinx.ext.autodoc',
    #  'sphinx.ext.napoleon',  # To support Google and NumPy docstrings
    'sphinx.ext.viewcode',
]

sys.path.insert(0, os.path.abspath('./MagmaClustPy'))

# The theme to use for HTML and HTML Help pages.
html_theme = 'alabaster'

# Custom static path
html_static_path = ['_static']

# Options for autodoc
autodoc_default_options = {
    'members': True,  # Document all members (functions, methods, attributes)
    'undoc-members': True,  # Include undocumented members
    'private-members': True,  # Include private members (those starting with _)
    'special-members': '__init__',  # Include special members (like __init__)
}

# Enable support for Google style and NumPy style docstrings
#napoleon_google_docstring = True
#napoleon_numpy_docstring = True

language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



