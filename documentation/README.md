# Antenna Solver MoM — documentation website

Source of the documentation site (Sphinx + Markdown/MyST + PyData theme). The site explains how the
solver works (theory) and how to use and extend it (user and developer guides), and includes an API
reference generated from the solver's docstrings.

Once published: <https://fahdilouro.github.io/Antenna_Solver_MoM/>

## Build it locally

From the repository root. The build tools go in their own environment, separate from the one used to
run the solver.

```bash
# once
python -m venv documentation/.venv
documentation/.venv/Scripts/python -m pip install -r documentation/requirements-docs.txt     # Windows
# documentation/.venv/bin/python -m pip install -r documentation/requirements-docs.txt       # Linux/macOS

# every time
documentation/.venv/Scripts/python -m sphinx -W --keep-going -b html documentation/source documentation/_build/html
python -m http.server 8000 -d documentation/_build/html
```

Then open <http://localhost:8000>. `-W` makes every warning (broken link, missing figure, unknown
reference) an error.

## Where things are

| Path | Content |
|---|---|
| `source/conf.py` | configuration |
| `source/index.md` | landing page |
| `source/<section>/` | pages, in Markdown; each folder has an `index.md` listing its pages |
| `source/references.bib` | bibliography (every entry checked against its DOI or ISBN) |
| `source/figures/generated/` | figures produced by `figure_notebooks/`, with a provenance `.json` |
| `figure_notebooks/` | notebooks that regenerate the figures from the current solver code |
| `requirements-docs.txt` | pinned build tools |

How to add a page, a figure, an equation or a reference: see the page
*Developer guide → Writing documentation* (`source/developer-guide/writing-documentation.md`).

## Publication

`.github/workflows/docs.yml` builds the site for every push and pull request that changes
`documentation/` or `solver/`, and publishes it to GitHub Pages from the `main` branch. Publication
requires, once, in the repository settings: *Settings → Pages → Build and deployment → Source:
GitHub Actions*.
