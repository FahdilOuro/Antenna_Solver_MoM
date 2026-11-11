"""
Sphinx configuration for the Antenna Solver MoM documentation.

Build (from the repository root, see documentation/README.md):

    python -m sphinx -b html documentation/source documentation/_build/html
"""

import inspect
import io
import os
import sys
from pathlib import Path

# The solver is imported (never executed) by autodoc to read its docstrings.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")   # importing plotting modules must not open windows

# -- Project ---------------------------------------------------------------
project = "Antenna Solver MoM"
author = "FahdilOuro"
copyright = "2025-2026, FahdilOuro (MIT License)"
release = "1.1.0"            # same as setup.py
version = release

GITHUB_USER = "FahdilOuro"
GITHUB_REPO = "Antenna_Solver_MoM"
GITHUB_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"

# -- Extensions ------------------------------------------------------------
extensions = [
    "myst_parser",               # pages are written in Markdown (MyST)
    "sphinx.ext.autodoc",        # API reference read from the solver's docstrings
    "sphinx.ext.napoleon",       # NumPy-style docstring sections
    "sphinx.ext.viewcode",       # highlighted source code pages, linked from the API
    "sphinx.ext.mathjax",        # equations
    "sphinx.ext.githubpages",    # .nojekyll for GitHub Pages
    "sphinx_design",             # cards, grids, badges, tabs
    "sphinx_copybutton",         # copy button on code blocks
    "sphinxcontrib.bibtex",      # citations from references.bib
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
# _static/ and _logos/ hold assets (and the font README), not pages.
exclude_patterns = ["_build", "_static", "_logos", "**/.ipynb_checkpoints"]
templates_path = ["_templates"]

# -- MyST (Markdown) -------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",      # $...$ and $$...$$
    "amsmath",         # \begin{aligned} ... environments
    "colon_fence",     # ::: directives
    "deflist",
    "attrs_inline",
    "attrs_block",
    "fieldlist",
]
myst_heading_anchors = 3
myst_dmath_double_inline = True

# -- Equations, figures ----------------------------------------------------
numfig = True
math_numfig = True
math_eqref_format = "Eq. ({number})"
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

# -- Bibliography ----------------------------------------------------------
# Every entry of references.bib is checked against its DOI (Crossref) or ISBN
# before it is added. Pages cite with {footcite:p}`key`; the full list is on
# bibliography.md.
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"
bibtex_foot_reference_style = "foot"

# -- Autodoc ---------------------------------------------------------------
# gmsh (needs system GL libraries) and CuPy (needs CUDA) are not installed in
# the documentation environment; autodoc replaces them with empty modules.
autodoc_mock_imports = ["gmsh", "cupy", "cupyx", "threadpoolctl"]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "signature"
# Docstrings name attributes such as `eigenvalues` that exist on several classes
# (ModalData, CMASweepData); Sphinx then warns that the reference is ambiguous.
# Those references come from the solver's text, which the documentation does
# not edit, so only this category of warning is silenced.
suppress_warnings = ["ref.python"]
autoclass_content = "both"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False


from docutils.parsers.rst import directives as _rst_directives, roles as _rst_roles

# docutils' own lookup functions, captured now: while reading the sources Sphinx
# replaces them by lookups into its domains, which need a build environment.
_DOCUTILS_DIRECTIVE_LOOKUP = _rst_directives.directive
_DOCUTILS_ROLE_LOOKUP = _rst_roles.role


def _rst_warnings(text):
    """docutils warnings (level >= 2) produced by parsing `text` as reStructuredText."""
    from docutils.core import publish_doctree

    directives, roles = _rst_directives, _rst_roles
    # Check with plain docutils only: Sphinx's lookups and the directives and
    # roles it registered globally need a Sphinx environment. Anything they
    # would have handled is reported as unknown here, and the docstring is then
    # shown verbatim, which is always safe.
    saved = (directives.directive, roles.role, dict(directives._directives), dict(roles._roles))
    directives.directive, roles.role = _DOCUTILS_DIRECTIVE_LOOKUP, _DOCUTILS_ROLE_LOOKUP
    directives._directives.clear()
    roles._roles.clear()
    stream = io.StringIO()
    try:
        publish_doctree(
            text,
            settings_overrides={
                "report_level": 2,
                "halt_level": 5,
                "warning_stream": stream,
                "file_insertion_enabled": False,
                "raw_enabled": False,
            },
        )
    finally:
        directives.directive, roles.role = saved[0], saved[1]
        directives._directives.clear()
        directives._directives.update(saved[2])
        roles._roles.clear()
        roles._roles.update(saved[3])
    # Sphinx roles (:class:, :math:, ...) are unknown to bare docutils but fine
    # in Sphinx: do not count them.
    return [line for line in stream.getvalue().splitlines()
            if "WARNING" in line or "ERROR" in line or "SEVERE" in line
            if "Unknown interpreted text role" not in line and "No role entry" not in line
            and "Trying" not in line]


def _docstring_verbatim_fallback(app, what, name, obj, options, lines):
    """
    Show a docstring verbatim when it is not valid reStructuredText.

    Many solver docstrings are free-form text (indented bullets, ASCII tables,
    equations). Parsed as reStructuredText they come out mangled and raise
    warnings, which the documentation build treats as errors. Rather than
    editing the solver's docstrings, such a docstring is shown exactly as
    written, in a literal block -- the same text help() would print. NumPy-style
    docstrings that parse cleanly keep their formatted rendering.
    """
    if not lines:
        return
    if not _rst_warnings("\n".join(lines)):
        return
    original = inspect.getdoc(obj) or "\n".join(lines)
    lines[:] = ["::", ""] + ["    " + line for line in original.splitlines()] + [""]


def setup(app):
    # After napoleon (priority 500), so the NumPy conversion is what gets checked.
    app.connect("autodoc-process-docstring", _docstring_verbatim_fallback, priority=600)


# -- HTML output -----------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "Antenna Solver MoM"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# The logos live outside _static/: Sphinx and the theme copy them there themselves.
html_favicon = "_logos/antenna_solver_logo_light.svg"
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = False

html_theme_options = {
    "logo": {
        "text": "Antenna Solver MoM",
        "image_light": "_logos/antenna_solver_logo_light.svg",
        "image_dark": "_logos/antenna_solver_logo_dark.svg",
        "alt_text": "Antenna Solver MoM - home",
    },
    "icon_links": [
        {"name": "GitHub", "url": GITHUB_URL, "icon": "fa-brands fa-github", "type": "fontawesome"},
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "show_toc_level": 2,
    "show_nav_level": 1,
    "navigation_depth": 3,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
}

# "Edit this page" opens the Markdown source on GitHub (main branch).
html_context = {
    "github_user": GITHUB_USER,
    "github_repo": GITHUB_REPO,
    "github_version": "main",
    "doc_path": "documentation/source",
    "default_mode": "auto",
}

# Pages with no left sidebar (the landing page is a single column).
html_sidebars = {"index": [], "bibliography": []}
