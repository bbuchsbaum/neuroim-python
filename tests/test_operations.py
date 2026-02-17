"""Tests for data manipulation operations."""

import pytest
import numpy as np
from neuroimpy import NeuroSpace, DenseNeuroVol, DenseNeuroVec, SparseNeuroVol
from neuroimpy.operations import concat, scale_series, mapf, downsample


class TestConcat:
    """Test concat function."""

    def setup_method(self):
        self.space4d_a = NeuroSpace(dim=[4, 5, 6, 3])
        self.space4d_b = NeuroSpace(dim=[4, 5, 6, 2])
        self.data_a = np.random.randn(4, 5, 6, 3)
        self.data_b = np.random.randn(4, 5, 6, 2)
        self.vec_a = DenseNeuroVec(self.data_a, self.space4d_a)
        self.vec_b = DenseNeuroVec(self.data_b, self.space4d_b)

    def test_concat_two(self):
        result = concat(self.vec_a, self.vec_b)
        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (4, 5, 6, 5)
        np.testing.assert_array_equal(result.data[..., :3], self.data_a)
        np.testing.assert_array_equal(result.data[..., 3:], self.data_b)

    def test_concat_three(self):
        space_c = NeuroSpace(dim=[4, 5, 6, 4])
        data_c = np.random.randn(4, 5, 6, 4)
        vec_c = DenseNeuroVec(data_c, space_c)
        result = concat(self.vec_a, self.vec_b, vec_c)
        assert result.shape == (4, 5, 6, 9)

    def test_concat_mismatched_spatial(self):
        bad_space = NeuroSpace(dim=[3, 5, 6, 2])
        bad_vec = DenseNeuroVec(np.random.randn(3, 5, 6, 2), bad_space)
        with pytest.raises(ValueError, match="spatial dimensions"):
            concat(self.vec_a, bad_vec)

    def test_concat_too_few(self):
        with pytest.raises(ValueError, match="at least two"):
            concat(self.vec_a)


class TestScaleSeries:
    """Test scale_series function."""

    def setup_method(self):
        self.space = NeuroSpace(dim=[3, 4, 5, 10])
        np.random.seed(42)
        self.data = np.random.randn(3, 4, 5, 10) * 5 + 100
        self.vec = DenseNeuroVec(self.data, self.space)

    def test_zscore(self):
        result = scale_series(self.vec, method="zscore")
        assert result.shape == self.vec.shape
        # Each voxel time series should have mean ~0 and std ~1
        ts = result.data[0, 0, 0, :]
        assert abs(np.mean(ts)) < 1e-10
        assert abs(np.std(ts) - 1.0) < 1e-10

    def test_mean_center(self):
        result = scale_series(self.vec, method="mean_center")
        ts = result.data[1, 2, 3, :]
        assert abs(np.mean(ts)) < 1e-10
        # Std should NOT be 1 (just centered)
        assert np.std(ts) > 0

    def test_unit_scale(self):
        result = scale_series(self.vec, method="unit_scale")
        ts = result.data[2, 1, 0, :]
        assert np.min(ts) >= -1e-10
        assert np.max(ts) <= 1.0 + 1e-10

    def test_zscore_constant_voxel(self):
        """Constant time series should not produce NaN."""
        data = np.ones((3, 4, 5, 10))
        vec = DenseNeuroVec(data, self.space)
        result = scale_series(vec, method="zscore")
        assert not np.any(np.isnan(result.data))

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            scale_series(self.vec, method="bad")


class TestMapf:
    """Test mapf function."""

    def setup_method(self):
        self.vol_space = NeuroSpace(dim=[4, 5, 6])
        self.vol_data = np.random.randn(4, 5, 6)
        self.vol = DenseNeuroVol(self.vol_data, self.vol_space)

        self.vec_space = NeuroSpace(dim=[4, 5, 6, 3])
        self.vec_data = np.random.randn(4, 5, 6, 3)
        self.vec = DenseNeuroVec(self.vec_data, self.vec_space)

    def test_mapf_vol_square(self):
        result = mapf(self.vol, np.square)
        assert isinstance(result, DenseNeuroVol)
        np.testing.assert_allclose(result.data, self.vol_data ** 2)

    def test_mapf_vol_abs(self):
        result = mapf(self.vol, np.abs)
        assert np.all(result.data >= 0)

    def test_mapf_vec(self):
        result = mapf(self.vec, lambda x: x * 2)
        assert isinstance(result, DenseNeuroVec)
        np.testing.assert_allclose(result.data, self.vec_data * 2)

    def test_mapf_wrong_type(self):
        with pytest.raises(TypeError):
            mapf("not_a_vol", np.abs)


class TestDownsample:
    """Test downsample function."""

    def setup_method(self):
        self.space = NeuroSpace(dim=[8, 8, 8], spacing=[1.0, 1.0, 1.0])
        self.data = np.ones((8, 8, 8))
        self.vol = DenseNeuroVol(self.data, self.space)

    def test_downsample_factor2(self):
        result = downsample(self.vol, factor=2)
        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (4, 4, 4)
        np.testing.assert_allclose(result.data, 1.0)
        np.testing.assert_allclose(result.spacing, [2.0, 2.0, 2.0])

    def test_downsample_factor4(self):
        result = downsample(self.vol, factor=4)
        assert result.shape == (2, 2, 2)
        np.testing.assert_allclose(result.spacing, [4.0, 4.0, 4.0])

    def test_downsample_identity(self):
        result = downsample(self.vol, factor=1)
        assert result.shape == self.vol.shape
        np.testing.assert_array_equal(result.data, self.vol.data)

    def test_downsample_averaging(self):
        """Verify block averaging works correctly."""
        data = np.zeros((4, 4, 4))
        data[0, 0, 0] = 8.0  # Average of 2x2x2 block should be 1.0
        vol = DenseNeuroVol(data, NeuroSpace(dim=[4, 4, 4]))
        result = downsample(vol, factor=2)
        assert result.shape == (2, 2, 2)
        assert result.data[0, 0, 0] == 1.0

    def test_downsample_non_divisible(self):
        """Non-divisible dimensions should be trimmed."""
        space = NeuroSpace(dim=[7, 7, 7])
        vol = DenseNeuroVol(np.ones((7, 7, 7)), space)
        result = downsample(vol, factor=2)
        assert result.shape == (3, 3, 3)  # 7 // 2 = 3

    def test_downsample_invalid_factor(self):
        with pytest.raises(ValueError, match="factor must be"):
            downsample(self.vol, factor=0)
