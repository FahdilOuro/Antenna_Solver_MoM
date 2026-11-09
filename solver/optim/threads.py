"""
BLAS thread control for loops over small matrices.

Measured on the development machine (OpenBLAS 0.3.27, 16 threads): a 157 x 157
complex LU takes 170-330 ms with the default thread pool and 0.5-1 ms with one
thread -- the pool's start-up/synchronisation dominates small problems. Inner
optimisation loops (P x P solves, LU of a few hundred unknowns) should therefore
run with a single BLAS thread; large factorisations should keep the pool.

    with limit_blas_threads(1):
        for z_loads in candidates:
            model.solve(emf, z_loads)

Uses threadpoolctl when it is installed; otherwise a no-op that warns once.
"""

import contextlib
import warnings

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:          # pragma: no cover - depends on the environment
    _threadpool_limits = None

_warned = False

#: Below this many unknowns, a single BLAS thread is faster for an LU on the
#: development machine (see module docstring). Indicative, not a hard rule.
SMALL_MATRIX_THRESHOLD = 800


@contextlib.contextmanager
def limit_blas_threads(n_threads=1):
    """Context manager limiting BLAS/OpenMP threads to `n_threads` (None = no limit)."""
    global _warned
    if n_threads is None:
        yield
        return
    if _threadpool_limits is None:
        if not _warned:
            warnings.warn("threadpoolctl is not installed: BLAS threads are not limited "
                          "(pip install threadpoolctl). Small-matrix loops may be much slower.")
            _warned = True
        yield
        return
    with _threadpool_limits(limits=n_threads):
        yield


def threads_for_size(n_unknowns):
    """1 for small problems (< SMALL_MATRIX_THRESHOLD unknowns), None (all threads) otherwise."""
    return 1 if n_unknowns < SMALL_MATRIX_THRESHOLD else None
