# Writing documentation

{.mom-status}
{bdg-info}`Draft` Describes the documentation system as built in version 0.1.

The documentation lives in `documentation/`, separate from the solver. It is a
[Sphinx](https://www.sphinx-doc.org/) site written in Markdown
([MyST](https://myst-parser.readthedocs.io/)), with the
[PyData theme](https://pydata-sphinx-theme.readthedocs.io/).

## Layout

```text
documentation/
├── README.md                  build the site in two commands
├── requirements-docs.txt      tools to build the site (not needed to run the solver)
├── figure_notebooks/          notebooks that produce the figures from the solver
└── source/
    ├── conf.py                configuration
    ├── index.md               landing page
    ├── references.bib         bibliography
    ├── _static/               logo, CSS
    ├── figures/generated/     figures written by figure_notebooks/ (+ provenance .json)
    ├── getting-started/  theory/  user-guide/  developer-guide/
    ├── api/                   API reference, generated from the docstrings
    ├── validation/  examples/
    └── bibliography.md
```

## Build and preview

```bash
python -m venv documentation/.venv              # once
documentation/.venv/Scripts/python -m pip install -r documentation/requirements-docs.txt   # once (Windows path)

documentation/.venv/Scripts/python -m sphinx -W --keep-going -b html documentation/source documentation/_build/html
python -m http.server 8000 -d documentation/_build/html     # then open http://localhost:8000
```

`-W` turns every warning into an error: a broken internal link, a missing figure or an unknown
reference stops the build. The same command runs on GitHub for every push (see below).

## Adding a page

1. Create `source/<section>/<name>.md`. Start it with a title and a status line:

   ````markdown
   # Title of the page

   {.mom-status}
   {bdg-info}`Draft` One sentence: what the page was written from (files, commit).
   ````

2. Add `<name>` to the `toctree` of `source/<section>/index.md`. The navigation is built from the
   toctrees: there is no menu to edit by hand.
3. Build. A page missing from every toctree raises a warning, hence an error.

### Page status

| Badge | Meaning |
|---|---|
| {bdg-secondary-line}`Planned` | Outline only: what the page will cover. |
| {bdg-info}`Draft` | Written and checked against the code, not yet reviewed. |
| {bdg-success}`Reviewed` | Reviewed and validated by the maintainer. |

For features, the same idea applies with {bdg-success-line}`Stable`,
{bdg-warning-line}`Experimental`, {bdg-danger-line}`Known issue` and
{bdg-danger}`Not functional`, as on {doc}`../validation/feature-status`.

### Theory pages

Follow the structure of {doc}`../theory/rwg-basis-functions`: the idea, the definition and
equations, a table of symbols with their name in the code, an **In this solver** section that says
exactly what the code does (and nothing it does not do), assumptions and limits, where to look in
the code, references.

## Equations

Inline: `$\nabla_s\cdot\mathbf f_n$`. Numbered and referenced:

````markdown
```{math}
:label: eq-my-label
Z\,I = V
```

As shown in {eq}`eq-my-label`, ...
````

Reuse the symbols of {doc}`../theory/notation`.

## References

Cite with `` {footcite:p}`rao1982` `` (parenthetical) or `` {footcite:t}`rao1982` `` (textual), and end
the page with a `footbibliography` block. Every key must exist in `source/references.bib`.

**Before adding an entry**, check it against its DOI on Crossref
(`https://api.crossref.org/works/<DOI>`) or, for a book without a DOI, against its ISBN record, and
write where and when it was checked in a comment above the entry. Never add a reference from memory.

## Figures

Figures that show solver results are produced by a notebook in `figure_notebooks/`, never drawn by
hand and never copied from older images:

- the notebook calls the current solver code, shows the figure inline and saves it to
  `source/figures/generated/` (SVG preferred);
- it also writes a `.json` provenance file next to the figure (solver commit, date, the checks it
  ran);
- the page embeds the figure with the class `mom-generated` and says in the caption which notebook
  and which commit produced it.

To refresh a figure after a change in the solver, re-run its notebook with the `antenna_solver`
environment, from the `figure_notebooks/` folder:

```bash
jupyter nbconvert --to notebook --execute --inplace theory_rwg_basis_function.ipynb
```

## Design rules

All of them are in `source/_static/css/custom.css`.

- **Font.** Iosevka for text, headings and code, served with the site
  (`_static/fonts/iosevka/`, SIL Open Font License). Base size `--mom-font-size: 12pt` (16 px on
  screen); every other size scales with it. Figure notebooks load the same font files, so figures
  and pages match.
- **Colours.** Those of the logo (`_logos/`): blue for $T^+$, amber for $T^-$. Use them with the same
  meaning in figures.
- **Clickable areas.** At least 44 × 44 px. A link that leads somewhere else in the site is better
  as a card (`grid-item-card` with `:link:`) than as a short link in a sentence: the whole card is
  the target.
- **Hover.** A clickable block reacts as a whole (title, text and image together): it rises by
  2 px with a light shadow (`0 2px 8px rgba(0,0,0,0.04)`) in 180 ms. No rotation, no large scaling.
  Nothing moves when the visitor's system asks for reduced motion.
- **Generated figures** use the class `mom-generated` and a `:target:` pointing to the image itself:
  the whole figure, caption included, then opens the full-size image.
- **Resizable columns.** `_static/js/resizable-sidebars.js` adds a handle between each sidebar and the
  page (drag, double-click to reset, arrow keys when focused); the width is kept in the reader's
  browser. It only acts where the sidebars are columns (from 960 px wide for the left one, 1200 px for
  the right one) and never while the left one is collapsed.

## API reference

The pages in `source/api/` contain only `automodule` directives: the text comes from the docstrings
of the solver at build time, so it cannot fall out of date. Docstrings written in NumPy style are
formatted; other docstrings are shown verbatim, exactly as `help()` would print them (see
`_docstring_verbatim_fallback` in `conf.py`). To document a new module, add an `automodule` block
to the matching page.

`gmsh`, `cupy` and `threadpoolctl` are not installed in the documentation environment: `conf.py`
mocks them so that the modules importing them can still be read.

## Publication

`.github/workflows/docs.yml` builds the site with `-W` on every push and pull request that touches
`documentation/` or `solver/`. On the `main` branch it also publishes the result to GitHub Pages.
