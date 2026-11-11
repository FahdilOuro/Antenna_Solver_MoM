"""
Display range of the polar cuts of efield3.

A sample that falls exactly in a null of the pattern gives -200 dB or lower; the
plotting functions floor every curve at `dynamic_range` dB below the common maximum
so that the radial axis shows the useful part of the pattern.
"""
import numpy as np
import pytest

from solver.efield.efield3 import DEFAULT_POLAR_DYNAMIC_RANGE_DB, clip_to_dynamic_range


def test_floor_is_common_to_all_curves():
    a = np.array([2.0, -5.0, -210.0])
    b = np.array([-1.0, -60.0, -np.inf])          # log10(0) of an exact null
    (ca, cb), (floor, top) = clip_to_dynamic_range([a, b], 40.0)
    assert top == 2.0 and floor == -38.0
    np.testing.assert_array_equal(ca, [2.0, -5.0, -38.0])
    np.testing.assert_array_equal(cb, [-1.0, -38.0, -38.0])


def test_values_above_the_floor_are_unchanged():
    curve = np.linspace(0.0, -39.0, 40)
    (clipped,), _ = clip_to_dynamic_range([curve], 40.0)
    np.testing.assert_array_equal(clipped, curve)


def test_none_returns_the_raw_values():
    curve = np.array([0.0, -200.0])
    (raw,), limits = clip_to_dynamic_range([curve], None)
    assert limits is None
    np.testing.assert_array_equal(raw, curve)


def test_default_range_and_invalid_input():
    assert DEFAULT_POLAR_DYNAMIC_RANGE_DB == 40.0
    with pytest.raises(ValueError):
        clip_to_dynamic_range([np.zeros(3)], 0.0)
    with pytest.raises(ValueError):
        clip_to_dynamic_range([np.full(3, -np.inf)], 40.0)
