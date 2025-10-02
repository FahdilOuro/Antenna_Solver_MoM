from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Search for all .pyx files in rwg/cython_rwg/
cython_modules = []
cython_dir = os.path.join("rwg", "cython_rwg")

for filename in os.listdir(cython_dir):
    if filename.endswith(".pyx"):
        module_name = f"rwg.cython_rwg.{filename[:-4]}"  # Example: rwg.cython_rwg.rwg1
        file_path = os.path.join(cython_dir, filename)
        cython_modules.append(
            Extension(
                module_name,
                [file_path],
                include_dirs=[np.get_include()],
            )
        )

setup(
    name='Antenna_Solver_MoM',
    version='1.1.0',
    packages=find_packages(),
    ext_modules=cythonize(
        cython_modules,
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[np.get_include()],
)

# To compile the Cython extensions, you can use the following command:
# python setup.py build_ext --inplace
# Make sure Cython is installed in your Python environment.