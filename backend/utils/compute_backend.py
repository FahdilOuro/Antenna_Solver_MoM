"""
GPU/CPU compute-backend selection for the MoM solver.

The project's numerically heavy kernels (impedance-matrix assembly, in the
first instance) are written array-library-agnostic against an `xp` module
that is either NumPy or CuPy. This module is the single place that decides
which one that is: detect a usable CUDA GPU once at import time, and fall
back to NumPy (CPU) automatically if there isn't one -- no per-machine
configuration needed, matching the project's requirement that the code
transparently uses whatever hardware is actually present.

"Usable" deliberately means more than "cupy imports": a machine can have
cupy installed, a driver that reports a device, and still fail to run a
kernel (missing CUDA toolkit headers, a WDDM TDR timeout, a remote-desktop
session with the GPU detached, etc.) -- see workflow.md, "GPU acceleration
(CuPy)". Detection therefore runs one tiny real kernel, not just a device
count check.
"""

import warnings

import numpy as np

_cupy = None
_gpu_available = False
_gpu_device_name = None
_gpu_detection_error = None


def _detect_gpu():
    global _cupy, _gpu_available, _gpu_device_name, _gpu_detection_error
    try:
        with warnings.catch_warnings():
            # Benign on machines where CuPy's wheel-bundled CUDA toolkit
            # component is used instead of a system-wide install (our case):
            # cupy still resolves headers/libraries fine via cuda-pathfinder.
            warnings.filterwarnings("ignore", message="CUDA path could not be detected")
            import cupy as cp

        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("no CUDA device visible to the driver")

        # A "visible" device can still fail to actually run anything (see
        # module docstring) -- confirm with one small real kernel + solve,
        # since that's the actual code path we depend on.
        probe = (cp.arange(4, dtype=cp.float64) * 2.0).sum()
        cp.cuda.Stream.null.synchronize()
        float(probe)  # forces a device->host sync/readback

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props['name']
        _gpu_device_name = name.decode() if isinstance(name, bytes) else str(name)
        _cupy = cp
        _gpu_available = True

    except Exception as exc:
        _cupy = None
        _gpu_available = False
        _gpu_device_name = None
        _gpu_detection_error = exc


_detect_gpu()

GPU_AVAILABLE = _gpu_available


def get_array_module(use_gpu=None):
    """
    Return the array module to compute with.

    Parameters
    ----------
    use_gpu : bool or None
        None (default): auto-detect -- CuPy if a usable GPU was found at
            import time, NumPy otherwise.
        True: force GPU. Raises RuntimeError if none is available (fails
            loudly rather than silently computing on CPU when the caller
            explicitly asked for GPU).
        False: force CPU even if a GPU is present (useful for testing and
            for CPU-vs-GPU cross-validation).
    """
    if use_gpu is False:
        return np
    if use_gpu is True:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "use_gpu=True was requested but no usable CUDA GPU was "
                "detected on this machine "
                f"({_gpu_detection_error!r} during detection). Install a "
                "CuPy wheel matching your driver's CUDA version (see "
                "`nvidia-smi`), e.g. `pip install cupy-cuda12x[ctk]`, or "
                "pass use_gpu=None/False to run on CPU."
            )
        return _cupy
    return _cupy if GPU_AVAILABLE else np


def to_numpy(array):
    """Bring an array back to host (NumPy) memory, whichever backend produced it."""
    if GPU_AVAILABLE and isinstance(array, _cupy.ndarray):
        return _cupy.asnumpy(array)
    return np.asarray(array)


def describe_backend():
    """One-line, human-readable summary of the active backend (for logging)."""
    if GPU_AVAILABLE:
        return f"GPU ({_gpu_device_name}, CuPy {_cupy.__version__})"
    return f"CPU (no usable CUDA GPU detected: {_gpu_detection_error!r})"
