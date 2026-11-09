"""
Tests for the GPU availability guards in compute_backend / rwg3.

The device is shared with the rest of the machine -- on a laptop it also
drives the display -- so "a GPU was detected at import time" does not mean
"the GPU has room right now". Two guards cover that, and they behave
differently on purpose:

  * before dispatching, a memory check can decline the GPU and say so, so the
    user learns the run will be slower BEFORE waiting on it;
  * once a run is under way, a CUDA failure is reported, not papered over.
    Silently redoing a seconds-long GPU assembly on the CPU turns it into
    minutes with nothing to explain why, which is the worse outcome.

These tests exercise the decision logic, not CUDA itself, so they run
identically on a machine with or without a GPU.
"""

import numpy as np
import pytest

import solver.rwg.rwg3 as rwg3
from solver.utils.compute_backend import (
    GPU_AVAILABLE, gpu_free_memory_bytes, gpu_has_headroom_for, gpu_runtime_errors,
)


def test_free_memory_is_reported_only_when_a_gpu_exists():
    free = gpu_free_memory_bytes()
    if GPU_AVAILABLE:
        assert free is None or (isinstance(free, int) and free >= 0)
    else:
        assert free is None


def test_headroom_check_never_forces_the_gpu_when_it_cannot_measure():
    """
    With no GPU (or a driver that will not report memory) the check must return
    True: it exists to VETO the GPU, and a machine without one is already on
    the CPU path, so a False here would be meaningless noise.
    """
    if not GPU_AVAILABLE:
        assert gpu_has_headroom_for(10 ** 15) is True


@pytest.mark.skipif(not GPU_AVAILABLE, reason="needs a real CUDA device to query memory")
def test_headroom_check_rejects_an_impossible_allocation():
    assert gpu_has_headroom_for(1) is True
    assert gpu_has_headroom_for(10 ** 15) is False


def test_runtime_error_tuple_is_usable_in_an_except_clause():
    errors = gpu_runtime_errors()
    assert isinstance(errors, tuple)
    assert all(isinstance(e, type) and issubclass(e, BaseException) for e in errors)


def test_worth_trying_declines_and_explains_when_the_matrix_cannot_fit(capsys):
    """A Z of E=20000 x 41 frequencies is ~244 GB: no device has that."""
    assert rwg3._gpu_is_worth_trying(20000, 41, "unit-test") is False
    out = capsys.readouterr().out
    assert "GPU skipped" in out
    assert "slower" in out, "the message must warn that the CPU path will be slower"


def test_cuda_failure_is_raised_with_context_not_swallowed(monkeypatch):
    """
    A CUDA failure mid-assembly must surface as an actionable RuntimeError.
    The original exception is kept as __cause__ so nothing is lost.
    """
    class FakeCudaError(Exception):
        pass

    monkeypatch.setattr(rwg3, "gpu_runtime_errors", lambda: (FakeCudaError,))

    def boom(*args, **kwargs):
        raise FakeCudaError("out of memory allocating 512 MB")

    with pytest.raises(RuntimeError) as excinfo:
        rwg3._run_gpu_assembly("unit-test", boom)

    message = str(excinfo.value)
    assert "use_gpu=False" in message, "must point at the deliberate CPU escape hatch"
    assert "slower" in message, "must set expectations about the CPU alternative"
    assert isinstance(excinfo.value.__cause__, FakeCudaError)


def test_non_cuda_bugs_are_not_disguised_as_gpu_failures(monkeypatch):
    """
    The guard must not behave like a bare `except Exception`: an ordinary bug
    inside the assembly has to propagate unchanged, or it becomes impossible
    to debug.
    """
    class FakeCudaError(Exception):
        pass

    monkeypatch.setattr(rwg3, "gpu_runtime_errors", lambda: (FakeCudaError,))

    def buggy(*args, **kwargs):
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError, match="shape mismatch"):
        rwg3._run_gpu_assembly("unit-test", buggy)
