"""
Collection settings for test/.

The files below are early-stage scripts, not tests: they run the pipeline at
import time on .mat files that are not part of the repository
(data/antennas_mesh/plate.mat, data/antennas_mesh2/dipole_mesh2.mat, ...), and
test_efield1-3 still call the file-name API that preceded the `path` namespace
of setup_save_file_paths(). Collected as tests they raised at import, and
pytest then stopped the whole run ("Interrupted: 9 errors during collection"),
so `python -m pytest test/` could not run at all.

They are kept, unchanged, for reference and excluded from collection until
they are rewritten as self-contained tests (building their own mesh, as
test_scattering_algorithm.py does).
"""

collect_ignore = [
    "test_rwg1.py",
    "test_rwg2.py",
    "test_rwg3.py",
    "test_rwg4.py",
    "test_rwg5.py",
    "test_efield1.py",
    "test_efield2.py",
    "test_efield3.py",
    "test_feed_edge.py",
]
