"""Tests for ROIVecWindow."""

import numpy as np
import pytest

from neuroimpy.neuro_space import NeuroSpace
from neuroimpy.roi_vec_window import ROIVecWindow


@pytest.fixture
def space():
    return NeuroSpace([64, 64, 64])


@pytest.fixture
def coords():
    return np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]])


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 3))


@pytest.fixture
def window(space, coords, data):
    return ROIVecWindow(space, coords, data, parent_index=5, center_index=0)


class TestInit:
    def test_basic_construction(self, space, coords, data):
        win = ROIVecWindow(space, coords, data)
        assert win.num_voxels == 3
        assert win.num_timepoints == 100

    def test_invalid_space(self, coords, data):
        with pytest.raises(TypeError, match="space must be a NeuroSpace"):
            ROIVecWindow("bad", coords, data)

    def test_coords_shape_mismatch(self, space, data):
        bad_coords = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="N x 3"):
            ROIVecWindow(space, bad_coords, data)

    def test_data_coords_mismatch(self, space, coords):
        bad_data = np.random.randn(100, 5)
        with pytest.raises(ValueError, match="data columns"):
            ROIVecWindow(space, coords, bad_data)

    def test_defaults(self, space, coords, data):
        win = ROIVecWindow(space, coords, data)
        assert win.parent_index == 0
        assert np.array_equal(win.parent_grid, np.array([0, 0, 0]))
        assert win.center_index == 0


class TestTimeSeries:
    def test_returns_correct_column(self, window, data):
        for i in range(3):
            np.testing.assert_array_equal(window.time_series(i), data[:, i])

    def test_shape(self, window):
        ts = window.time_series(0)
        assert ts.shape == (100,)


class TestMeanSeries:
    def test_shape(self, window):
        ms = window.mean_series()
        assert ms.shape == (100,)

    def test_value(self, window, data):
        expected = np.mean(data, axis=1)
        np.testing.assert_allclose(window.mean_series(), expected)


class TestProperties:
    def test_num_voxels(self, window):
        assert window.num_voxels == 3

    def test_num_timepoints(self, window):
        assert window.num_timepoints == 100

    def test_len(self, window):
        assert len(window) == 3


class TestRepr:
    def test_repr(self, window):
        r = repr(window)
        assert "ROIVecWindow" in r
        assert "n_voxels=3" in r
        assert "n_timepoints=100" in r
        assert "parent_index=5" in r
        assert "center_index=0" in r


class TestSingleVoxel:
    def test_single_voxel_window(self, space):
        coords = np.array([[5, 5, 5]])
        data = np.arange(50, dtype=float).reshape(50, 1)
        win = ROIVecWindow(space, coords, data)
        assert win.num_voxels == 1
        assert win.num_timepoints == 50
        np.testing.assert_array_equal(win.time_series(0), np.arange(50, dtype=float))
        np.testing.assert_array_equal(win.mean_series(), np.arange(50, dtype=float))
