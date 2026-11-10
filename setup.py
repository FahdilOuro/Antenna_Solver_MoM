import os

from setuptools import setup, find_namespace_packages, Extension

# Optional Cython acceleration layer.
#
# Every .pyx file in solver/rwg/cython_rwg/ is compiled as an extension module
# (e.g. solver/rwg/cython_rwg/kernel.pyx -> solver.rwg.cython_rwg.kernel).
# That folder does not exist yet, and nothing in the solver depends on it, so
# the package installs as pure Python and Cython is only needed once .pyx files
# are added:
#     python setup.py build_ext --inplace
cython_dir = os.path.join("solver", "rwg", "cython_rwg")
pyx_files = sorted(f for f in os.listdir(cython_dir) if f.endswith(".pyx")) if os.path.isdir(cython_dir) else []

ext_modules = []
if pyx_files:
    import numpy as np
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            Extension(
                "solver.rwg.cython_rwg." + f[:-4],
                [os.path.join(cython_dir, f)],
                include_dirs=[np.get_include()],
            )
            for f in pyx_files
        ],
        compiler_directives={"language_level": "3"},
    )

setup(
    name='Antenna_Solver_MoM',
    version='1.1.0',
    # solver/ has no __init__.py (namespace package), hence find_namespace_packages.
    # The example notebooks, their data and the profiling scripts are not part of
    # the installed package.
    packages=find_namespace_packages(
        include=["solver", "solver.*"],
        exclude=["solver.antenna_simulation", "solver.antenna_simulation.*",
                 "solver.performance", "solver.performance.*", "*.__pycache__"],
    ),
    ext_modules=ext_modules,
)
