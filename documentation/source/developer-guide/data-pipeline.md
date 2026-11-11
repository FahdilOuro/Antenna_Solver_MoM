# Data pipeline and file managers

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- The `DataManager_rwg1` … `DataManager_rwg4`, `DataManager_cma` and `DataManager_cma_sweep`
  classes: what each saves and loads, and the keys of each `.mat` file.
- The `path` namespace from `setup_save_file_paths` and why it is passed through the whole pipeline.
- Mesh fingerprints and the freshness rules (which functions always rebuild from the `.msh`, which
  ones load saved files).
