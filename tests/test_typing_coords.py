"""Tests for VoxelCoord / WorldCoord newtypes and their validating constructors."""

import numpy as np
import pytest

from neuroim.typing import VoxelCoord, WorldCoord, voxel_coord, world_coord


class TestVoxelCoord:
    def test_1d_int_tuple_accepted(self):
        v = voxel_coord((10, 20, 30))
        assert isinstance(v, np.ndarray)
        assert v.shape == (3,)
        assert v.dtype.kind == "i"

    def test_2d_int_array_accepted(self):
        arr = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.int32)
        v = voxel_coord(arr)
        assert v.shape == (3, 3)
        assert v.dtype.kind == "i"

    def test_float_input_coerced_to_int(self):
        v = voxel_coord([1.0, 2.0, 3.0])
        assert v.dtype.kind == "i"
        np.testing.assert_array_equal(v, [1, 2, 3])

    def test_wrong_1d_length_rejected(self):
        with pytest.raises(ValueError, match="length 3"):
            voxel_coord([1, 2])

    def test_wrong_2d_shape_rejected(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            voxel_coord(np.zeros((4, 2), dtype=int))

    def test_3d_input_rejected(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            voxel_coord(np.zeros((2, 3, 3), dtype=int))


class TestWorldCoord:
    def test_1d_float_tuple_accepted(self):
        w = world_coord((1.5, 2.5, 3.5))
        assert isinstance(w, np.ndarray)
        assert w.shape == (3,)
        assert w.dtype == np.float64

    def test_2d_float_array_accepted(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]])
        w = world_coord(arr)
        assert w.shape == (2, 3)
        assert w.dtype == np.float64

    def test_int_input_coerced_to_float(self):
        w = world_coord([1, 2, 3])
        assert w.dtype == np.float64

    def test_wrong_1d_length_rejected(self):
        with pytest.raises(ValueError, match="length 3"):
            world_coord([1.0, 2.0, 3.0, 4.0])

    def test_wrong_2d_shape_rejected(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            world_coord(np.zeros((5, 4)))


class TestNewTypeIdentity:
    """NewType is a no-op at runtime, but it does mark intent for mypy."""

    def test_voxel_coord_is_ndarray(self):
        v = voxel_coord((1, 2, 3))
        assert isinstance(v, np.ndarray)

    def test_world_coord_is_ndarray(self):
        w = world_coord((1.0, 2.0, 3.0))
        assert isinstance(w, np.ndarray)
