# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
import sys
from pathlib import Path

# Make the package importable without installation (local builds);
# on Read the Docs the package is also pip-installed (see .readthedocs.yaml).
DOCS_SOURCE = Path(__file__).resolve().parent
REPO_ROOT = DOCS_SOURCE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import grstbx

# The notebooks of the tutorials live in notebook/ (where they are run); they are
# copied into the documentation source at each build (git-ignored copies).
NOTEBOOKS = {
    'tutorials/l2a_image': ['notebook/grstbx_l2a_visu.ipynb'],
    'tutorials/subset_export': ['notebook/grstbx_l2a_visu_subset_export.ipynb'],
    'tutorials/validation': ['notebook/validation/grstbx_l2a_hypernets_matchup.ipynb',
                             'notebook/validation/grstbx_l2b_hypernets_matchup.ipynb'],
    'tutorials/case_studies': ['notebook/case_study/clear_lake/grstbx_rgb_dem_multitemp.ipynb',
                               'notebook/case_study/bagre/grstbx_l2a_datacube_matchup.ipynb'],
}
for _target, _notebooks in NOTEBOOKS.items():
    (DOCS_SOURCE / _target).mkdir(exist_ok=True)
    for _notebook in _notebooks:
        shutil.copy2(REPO_ROOT / _notebook, DOCS_SOURCE / _target / Path(_notebook).name)

# -- Project information -----------------------------------------------------

project = 'grstbx'
copyright = '2026, Tristan Harmel'
author = 'Tristan Harmel'
version = grstbx.__version__
release = grstbx.__version__
today_fmt = '%Y-%m-%d'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_nb',
]

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autoclass_content = 'class'
autodoc_typehints = 'none'          # types are given in the docstrings
autodoc_member_order = 'bysource'
# 'members' is set in the autosummary templates (_templates/autosummary/) to
# avoid documenting objects twice
add_module_names = False

# The docstrings mix NumPy sections ("Example") and reST fields (":param x:");
# napoleon converts the first ones and leaves the others.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_preprocess_types = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'geopandas': ('https://geopandas.org/en/stable', None),
    'rioxarray': ('https://corteva.github.io/rioxarray/stable', None),
    'dask': ('https://docs.dask.org/en/stable', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
pygments_style = 'sphinx'

html_theme_options = {
    'repository_url': 'https://github.com/Tristanovsk/grstbx',
    'repository_branch': 'main',
    'path_to_docs': 'docs/source',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'use_download_button': True,
    'navigation_with_keys': True,
    'show_toc_level': 2,
}

html_title = ''
html_logo = '_static/GRS_TBX.svg'
html_favicon = '_static/GRS_TBX.png'

html_static_path = ['_static']
html_css_files = ['custom.css']
html_show_sourcelink = False
html_last_updated_fmt = today_fmt

htmlhelp_basename = 'grstbxdoc'

# -- MyST / notebook rendering -----------------------------------------------

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_admonition',
    'html_image',
    'linkify',
    'smartquotes',
]
myst_heading_anchors = 3

# The notebooks need GRS images and in situ data that are not available on
# Read the Docs: they are rendered with their stored outputs, never executed.
nb_execution_mode = 'off'
nb_merge_streams = True
suppress_warnings = ['mystnb.unknown_mime_type', 'myst.header']
