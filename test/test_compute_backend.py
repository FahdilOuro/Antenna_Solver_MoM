"""
Tests for backend/utils/compute_backend.py.

Context
-------
GPU_AVAILABLE / get_array_module() detect a usable CUDA GPU once at import
time (see the module docstring for why "usable" requires running a real
kernel, not just checking device count). should_use_gpu() adds a mesh-size
threshold on top: impedance_matrix_z_sweep_gpu's batched assembly only pays
off once there are enough triangle pairs to amortise its overhead, measured
on the reference NVIDIA T600 to cross over somewhere between E=39 (loses)
and E=57 (wins) -- see workflow.md, "GPU acceleration (CuPy)" for the full
threshold benchmark across mesh sizes up to E=1007.

These tests check the *decision logic* itself (deterministic, independent
of whether a GPU is actually present on the machine running the tests) --
not the underlying kernels, which are covered by
backend/utils/quadrature/tests/test_impedance_sweep_gpu_regression.py.
"""

import numpy as np
import pytest

from backend.utils.compute_backend import (
    GPU_AVAILABLE, DEFAULT_GPU_EDGE_THRESHOLD, should_use_gpu,
    get_array_module, to_numpy, describe_backend,
)


def test_should_use_gpu_respects_explicit_override_regardless_of_size():
    assert should_use_gpu(1, use_gpu=True) is True
    assert should_use_gpu(10_000_000, use_gpu=False) is False


def test_should_use_gpu_auto_below_threshold_is_false():
    assert should_use_gpu(DEFAULT_GPU_EDGE_THRESHOLD - 1, use_gpu=None) is False


def test_should_use_gpu_auto_at_or_above_threshold_matches_gpu_availability():
    assert should_use_gpu(DEFAULT_GPU_EDGE_THRESHOLD, use_gpu=None) == GPU_AVAILABLE
    assert should_use_gpu(DEFAULT_GPU_EDGE_THRESHOLD * 100, use_gpu=None) == GPU_AVAILABLE


def test_should_use_gpu_custom_threshold():
    assert should_use_gpu(50, use_gpu=None, threshold=10) == GPU_AVAILABLE
    assert should_use_gpu(5, use_gpu=None, threshold=10) is False


def test_get_array_module_auto_matches_gpu_availability():
    xp = get_array_module(None)
    assert (xp.__name__ == "cupy") == GPU_AVAILABLE


def test_get_array_module_force_cpu_is_always_numpy():
    assert get_array_module(False) is np


def test_get_array_module_force_gpu_without_device_raises(monkeypatch):
    if GPU_AVAILABLE:
        pytest.skip("a real GPU is available on this machine; cannot test the no-GPU error path")
    with pytest.raises(RuntimeError, match="no usable CUDA GPU"):
        get_array_module(True)


def test_to_numpy_roundtrip_on_active_backend():
    xp = get_array_module(None)
    arr = xp.array([1.0, 2.0, 3.0])
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_describe_backend_is_a_nonempty_string():
    desc = describe_backend()
    assert isinstance(desc, str) and len(desc) > 0
    assert ("GPU" in desc) == GPU_AVAILABLE
