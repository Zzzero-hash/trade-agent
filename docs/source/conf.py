import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('../../src'))

project = 'trade-agent'
copyright = f"{datetime.now():%Y}, Trading RL Platform Team"
author = 'Trading RL Platform Team'
release = '0.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinxcontrib.mermaid',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_autodoc_typehints',
    'sphinx_multiversion',
]

autosummary_generate = True
autoclass_content = 'both'
autodoc_typehints = 'description'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

templates_path = ['_templates']
exclude_patterns: list[str] = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_external_links': True,
}

# Mermaid configuration (if needed for dark theme adjustments)
mermaid_output_format = 'raw'

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'html_image',
    'substitution',
    'linkify'
]

# Anchor headings up to depth 3 for direct linking
myst_heading_anchors = 3

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for TODO extension -----------------------------------------------
todo_include_todos = True

# Fallback for GitHub pages
html_baseurl = os.environ.get('DOCS_BASE_URL', '')

# sphinx-multiversion configuration
smv_tag_whitelist = r'^v?\d+\.\d+\.\d+$'
smv_branch_whitelist = r'^(master|main)$'
smv_remote_whitelist = r'^origin$'
smv_released_pattern = r'^tags/.+$'
smv_latest_version = 'master'
