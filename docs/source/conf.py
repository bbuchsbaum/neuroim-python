# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

import neuroimpy  # noqa: E402

# -- Project information -----------------------------------------------------

project = "neuroimpy"
copyright = "2024, Brad Buchsbaum"
author = "Brad Buchsbaum"
release = neuroimpy.__version__
version = ".".join(release.split(".")[:2])
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
    "sphinx_remove_toctrees",
]

source_suffix = {
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = [
    "**/.ipynb_checkpoints",
    "**/notebooks/*.py",
    "**/notebooks/*.md",
    "**/notebooks/00_index.ipynb",
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

# -- Autosummary settings ----------------------------------------------------

autosummary_generate = True

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx settings ----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# -- Theme settings ----------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "use_edit_page_button": True,
    "search_bar_text": "Search the docs...",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bbuchsbaum/neuroimpy",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/neuroimpy",
            "icon": "fa-brands fa-python",
        },
    ],
}

html_context = {
    "default_mode": "auto",
    "github_user": "bbuchsbaum",
    "github_repo": "neuroimpy",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Performance settings ----------------------------------------------------

remove_from_toctrees = ["api/generated/*"]

# -- Notebook execution settings ---------------------------------------------

nb_execution_mode = "off"  # Don't execute notebooks during build

# -- Copybutton settings -----------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
