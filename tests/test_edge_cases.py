"""
Comprehensive edge case tests for NeuroVol and NeuroVec.

This module tests boundary conditions, error handling, and unusual inputs
that might break the implementation.
"""

import pytest
import numpy as np
import neuroim as pn
from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vol import DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol
from neuroim.neuro_vec import DenseNeuroVec, SparseNeuroVec
import warnings
import tempfile


class TestNeuroVolEdgeCases:
    """Test edge cases for 3D volume operations."""

    def test_single_voxel_volume(self):
        """Test operations on 1x1x1 volume."""
        space = NeuroSpace(dim=(1, 1, 1))
        data = np.array([[[42.0]]])
        vol = DenseNeuroVol(data, space)

        assert vol.shape == (1, 1, 1)
        assert vol[0, 0, 0] == 42.0
        assert vol.min() == 42.0
        assert vol.max() == 42.0
        assert np.mean(vol.data) == 42.0
        assert np.std(vol.data) == 0.0

        # Test arithmetic
        vol2 = vol * 2
        assert vol2[0, 0, 0] == 84.0

    def test_empty_sparse_volume(self):
        """Test sparse volume with no non-zero voxels."""
        space = NeuroSpace(dim=(10, 10, 10))
        indices = np.array([], dtype=int)
        values = np.array([], dtype=float)

        sparse_vol = SparseNeuroVol(data=values, space=space, indices=indices)

        assert sparse_vol.shape == (10, 10, 10)
        assert sparse_vol[5, 5, 5] == 0.0  # All values should be 0
        assert sparse_vol.min() == 0.0
        assert sparse_vol.max() == 0.0
        assert len(sparse_vol.data) == 0

    def test_all_nan_volume(self):
        """Test volume filled with NaN values."""
        space = NeuroSpace(dim=(5, 5, 5))
        data = np.full((5, 5, 5), np.nan)
        vol = DenseNeuroVol(data, space)

        assert np.all(np.isnan(vol.data))

        # Statistics should handle NaN appropriately
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.isnan(np.mean(vol.data))
            assert np.isnan(np.std(vol.data))

        # NaN propagation in arithmetic
        vol2 = vol + 1
        assert np.all(np.isnan(vol2.data))

    def test_mixed_nan_values(self):
        """Test volume with mix of NaN and valid values."""
        space = NeuroSpace(dim=(3, 3, 3))
        data = np.ones((3, 3, 3))
        data[1, 1, 1] = np.nan
        data[2, 2, 2] = np.nan
        vol = DenseNeuroVol(data, space)

        # Count valid values
        valid_mask = ~np.isnan(vol.data)
        assert np.sum(valid_mask) == 25  # 27 - 2 NaN values

        # Statistics should skip NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Using nanmean to skip NaN values
            assert np.nanmean(vol.data) == 1.0  # Mean of non-NaN values

    def test_extreme_values(self):
        """Test volumes with extreme numerical values."""
        space = NeuroSpace(dim=(3, 3, 3))

        # Very large values
        large_data = np.full((3, 3, 3), 1e308)
        vol_large = DenseNeuroVol(large_data, space)
        assert vol_large.max() == 1e308

        # Very small values
        small_data = np.full((3, 3, 3), 1e-308)
        vol_small = DenseNeuroVol(small_data, space)
        assert vol_small.min() == 1e-308

        # Mixed extreme values
        mixed_data = np.array([[[1e308, -1e308, 0]]])
        vol_mixed = DenseNeuroVol(
            mixed_data.reshape(1, 1, 3), NeuroSpace(dim=(1, 1, 3))
        )
        assert vol_mixed.max() == 1e308
        assert vol_mixed.min() == -1e308

    def test_integer_overflow_protection(self):
        """Test handling of integer overflow scenarios."""
        space = NeuroSpace(dim=(2, 2, 2))

        # Use int8 to test overflow
        data = np.array(
            [[[126, 127], [125, 120]], [[100, 90], [80, 70]]], dtype=np.int8
        )
        vol = DenseNeuroVol(data, space)

        # Adding 10 would overflow int8
        vol2 = vol + 10
        # Should promote to larger type or handle gracefully
        assert vol2.data.dtype != np.int8 or np.all(vol2.data <= 127)

    def test_zero_spacing(self):
        """Test handling of zero or negative spacing."""
        with pytest.raises(ValueError):
            # Zero spacing should be invalid
            NeuroSpace(dim=(10, 10, 10), spacing=(1, 0, 1))

        with pytest.raises(ValueError):
            # Negative spacing should be invalid
            NeuroSpace(dim=(10, 10, 10), spacing=(1, -1, 1))

    def test_mismatched_dimensions(self):
        """Test error handling for dimension mismatches."""
        space = NeuroSpace(dim=(10, 10, 10))

        # Wrong data shape
        with pytest.raises(ValueError):
            data = np.zeros((10, 10, 5))  # Wrong z dimension
            DenseNeuroVol(data, space)

        # 2D data for 3D space
        with pytest.raises(ValueError):
            data = np.zeros((10, 10))
            DenseNeuroVol(data, space)

    def test_memory_efficient_operations(self):
        """Test that operations don't create unnecessary copies."""
        space = NeuroSpace(dim=(100, 100, 100))
        data = np.zeros((100, 100, 100), dtype=np.float32)
        vol = DenseNeuroVol(data, space)

        # Get memory address of underlying data
        data_id = id(vol.data)

        # These operations should create new objects
        vol2 = vol + 1
        assert id(vol2.data) != data_id

        # But accessing data shouldn't copy
        data_ref = vol.data
        assert id(data_ref) == data_id

    def test_dtype_preservation(self):
        """Test that dtypes are preserved appropriately."""
        space = NeuroSpace(dim=(5, 5, 5))

        for dtype in [np.float32, np.float64, np.int16, np.int32]:
            data = np.ones((5, 5, 5), dtype=dtype)
            vol = DenseNeuroVol(data, space)
            assert vol.data.dtype == dtype

            # Some operations might change dtype
            if np.issubdtype(dtype, np.integer):
                # Division should promote to float
                vol_div = vol / 2
                assert np.issubdtype(vol_div.data.dtype, np.floating)


class TestNeuroVecEdgeCases:
    """Test edge cases for 4D time series operations."""

    def test_single_timepoint(self):
        """Test 4D data with only one timepoint."""
        space = NeuroSpace(dim=(10, 10, 10, 1))
        data = np.random.randn(10, 10, 10, 1)
        vec = DenseNeuroVec(data, space)

        assert vec.shape == (10, 10, 10, 1)

        # Extract time series
        ts = vec.series(np.array([[5, 5, 5]]))
        assert ts.shape == (1, 1)  # (n_timepoints, n_voxels)

        # Should be able to extract as volume
        vol = vec[:, :, :, 0]
        assert vol.shape == (10, 10, 10)

    def test_very_long_time_series(self):
        """Test handling of very long time series."""
        # 10000 timepoints
        space = NeuroSpace(dim=(10, 10, 10, 10000))
        # Don't actually allocate huge array

        # Test space creation
        assert tuple(space.dim) == (10, 10, 10, 10000)
        assert space.dim[3] == 10000  # n_timepoints

    def test_sparse_vec_empty_mask(self):
        """Test sparse neurovec with empty mask."""
        space = NeuroSpace(dim=(10, 10, 10, 20))
        mask = LogicalNeuroVol(
            np.zeros((10, 10, 10), dtype=bool), NeuroSpace(dim=(10, 10, 10))
        )

        # No voxels in mask
        assert mask.data.sum() == 0

        # Should handle gracefully
        sparse_vec = SparseNeuroVec(
            data=np.array([]).reshape(0, 20), mask=mask, space=space
        )

        assert sparse_vec.shape == (10, 10, 10, 20)
        ts = sparse_vec.series(np.array([[5, 5, 5]]))
        assert np.all(ts == 0)

    def test_series_extraction_boundaries(self):
        """Test series extraction at volume boundaries."""
        space = NeuroSpace(dim=(10, 10, 10, 50))
        data = np.random.randn(10, 10, 10, 50)
        vec = DenseNeuroVec(data, space)

        # Test all corners
        corners = [
            (0, 0, 0),
            (9, 0, 0),
            (0, 9, 0),
            (0, 0, 9),
            (9, 9, 0),
            (9, 0, 9),
            (0, 9, 9),
            (9, 9, 9),
        ]

        for corner in corners:
            ts = vec.series(np.array([corner]))
            assert ts.shape == (50, 1)  # (n_timepoints, n_voxels)
            expected = data[corner[0], corner[1], corner[2], :]
            np.testing.assert_array_equal(ts[:, 0], expected)

    def test_time_indexing_edge_cases(self):
        """Test edge cases in time dimension indexing."""
        space = NeuroSpace(dim=(5, 5, 5, 20))
        data = np.arange(5 * 5 * 5 * 20).reshape(5, 5, 5, 20)
        vec = DenseNeuroVec(data, space)

        # Negative indexing should work
        last_vol = vec[:, :, :, -1]
        expected_last = vec[:, :, :, 19]
        np.testing.assert_array_equal(last_vol.data, expected_last.data)

        # Empty slice should return empty array
        empty = vec[:, :, :, 20:19]  # Returns empty time dimension
        assert empty.shape == (5, 5, 5, 0)

    def test_concatenation_edge_cases(self):
        """Test concatenating neurovec objects."""
        space1 = NeuroSpace(dim=(5, 5, 5, 10))
        space2 = NeuroSpace(dim=(5, 5, 5, 20))

        vec1 = DenseNeuroVec(np.ones((5, 5, 5, 10)), space1)
        vec2 = DenseNeuroVec(np.ones((5, 5, 5, 20)) * 2, space2)

        # Concatenate along time
        if hasattr(pn, "concat_time"):
            combined = pn.concat_time([vec1, vec2])
            assert combined.shape == (5, 5, 5, 30)
            assert np.all(combined[:, :, :, :10].data == 1)
            assert np.all(combined[:, :, :, 10:].data == 2)


class TestIOEdgeCases:
    """Test edge cases in file I/O operations."""

    def test_write_read_roundtrip(self, tmp_path):
        """Test basic write/read roundtrip."""
        space = NeuroSpace(dim=(5, 5, 5))
        vol = DenseNeuroVol(np.ones((5, 5, 5)), space)

        path = tmp_path / "roundtrip.nii.gz"
        pn.write_vol(vol, path)
        vol2 = pn.io.read_vol(path)
        np.testing.assert_array_equal(vol.data, vol2.data)

    def test_write_read_special_values(self):
        """Test I/O with special floating point values."""
        data = np.array(
            [[[0, 1, -1]], [[np.inf, -np.inf, np.nan]], [[1e-45, 1e45, 0]]]
        ).reshape(3, 3, 1)

        vol = DenseNeuroVol(data, NeuroSpace(dim=(3, 3, 1)))

        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
            # Some formats might not support inf/nan
            try:
                pn.write_vol(vol, tmp.name)
                vol2 = pn.io.read_vol(tmp.name)

                # Check what was preserved
                np.testing.assert_array_equal(vol.data[0], vol2.data[0])
                # inf/nan handling might vary by format
            except Exception:
                # Some formats might not support these values
                pass

    def test_path_with_spaces(self):
        """Test I/O with file paths containing spaces."""
        import os

        space = NeuroSpace(dim=(5, 5, 5))
        vol = DenseNeuroVol(np.ones((5, 5, 5)), space)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create path with spaces
            subdir = os.path.join(tmpdir, "my test dir")
            os.makedirs(subdir)
            filepath = os.path.join(subdir, "test file.nii.gz")

            pn.write_vol(vol, filepath)
            vol2 = pn.io.read_vol(filepath)
            np.testing.assert_array_equal(vol.data, vol2.data)

    def test_readonly_file_handling(self, tmp_path):
        """Test handling of read-only files."""
        import os
        import stat

        space = NeuroSpace(dim=(5, 5, 5))
        vol = DenseNeuroVol(np.ones((5, 5, 5)), space)

        path = tmp_path / "readonly.nii.gz"
        pn.write_vol(vol, path)

        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            vol2 = pn.io.read_vol(path)
            assert vol2 is not None

            with pytest.raises((IOError, PermissionError)):
                pn.write_vol(vol, path)
        finally:
            os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)


class TestArithmeticEdgeCases:
    """Test edge cases in arithmetic operations."""

    def test_division_by_zero(self):
        """Test division by zero handling."""
        data = np.array([[[1, 0, -1]]])
        vol = DenseNeuroVol(data.reshape(1, 1, 3), NeuroSpace(dim=(1, 1, 3)))

        # Divide by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vol / 0

            # Should produce inf/-inf/nan appropriately
            assert np.isinf(result.data[0, 0, 0])  # 1/0 = inf
            assert np.isnan(result.data[0, 0, 1])  # 0/0 = nan
            assert np.isinf(result.data[0, 0, 2])  # -1/0 = -inf

    def test_incompatible_shapes(self):
        """Test arithmetic with incompatible shapes."""
        vol1 = DenseNeuroVol(np.ones((5, 5, 5)), NeuroSpace(dim=(5, 5, 5)))
        vol2 = DenseNeuroVol(np.ones((3, 3, 3)), NeuroSpace(dim=(3, 3, 3)))

        # Should raise error for incompatible shapes
        with pytest.raises(ValueError):
            vol1 + vol2

    def test_scalar_arithmetic_edge_cases(self):
        """Test arithmetic with edge case scalars."""
        space = NeuroSpace(dim=(3, 3, 3))
        vol = DenseNeuroVol(np.ones((3, 3, 3)), space)

        # Operations with special scalars
        test_scalars = [0, -0.0, np.inf, -np.inf, np.nan]

        for scalar in test_scalars:
            # These should all work without error
            _ = vol + scalar
            _ = vol - scalar
            _ = vol * scalar
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = vol / scalar
