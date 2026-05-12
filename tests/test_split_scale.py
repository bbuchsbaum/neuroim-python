"""
Test suite for split_reduce and split_scale functions.

This module tests the statistical utility functions in neuroim,
corresponding to the R neuroim2 test-splitscale.R tests.
"""

import pytest
import numpy as np
from neuroim import (
    NeuroSpace, DenseNeuroVol, DenseNeuroVec, SparseNeuroVec,
    LogicalNeuroVol, split_reduce, split_scale
)
from neuroim.stats import split_blocks, split_clusters, split_fill


class TestSplitScale:
    """Test cases for split_reduce and split_scale functionality."""
    
    @pytest.fixture
    def create_test_data(self):
        """Create test data similar to R test."""
        # Create a 4x3 matrix as 4D neuroimaging data
        # In R: mat <- matrix(1:12, nrow=4)
        # This creates: [[1,5,9], [2,6,10], [3,7,11], [4,8,12]]
        
        # Create as 1x1x3x4 NeuroVec (3 voxels, 4 time points)
        space_4d = NeuroSpace((1, 1, 3, 4), spacing=(1, 1, 1, 1))
        
        # Create data matching R matrix structure
        data = np.zeros((1, 1, 3, 4))
        data[0, 0, 0, :] = [1, 2, 3, 4]   # First column of R matrix
        data[0, 0, 1, :] = [5, 6, 7, 8]   # Second column
        data[0, 0, 2, :] = [9, 10, 11, 12]  # Third column
        
        vec = DenseNeuroVec(data, space_4d)
        
        # Factor: c(1, 1, 2, 2) in R
        fac = np.array([1, 1, 2, 2])
        
        return vec, fac
    
    def test_split_reduce_basic(self, create_test_data):
        """Test split_reduce with basic sum function."""
        vec, fac = create_test_data
        
        # Define sum function
        def custom_sum(x):
            return np.sum(x)
        
        # Apply split_reduce
        result = split_reduce(vec, fac, custom_sum)
        
        # Check result type
        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (1, 1, 3)
        
        # Expected values from R test:
        # For voxel 0: group 1 has [1,2], group 2 has [3,4]
        #   sum(1,2)=3, sum(3,4)=7, mean=5
        # For voxel 1: group 1 has [5,6], group 2 has [7,8]
        #   sum(5,6)=11, sum(7,8)=15, mean=13
        # For voxel 2: group 1 has [9,10], group 2 has [11,12]
        #   sum(9,10)=19, sum(11,12)=23, mean=21
        
        expected = np.array([[[5, 13, 21]]])
        np.testing.assert_array_almost_equal(result.data, expected)
    
    def test_split_reduce_with_mean(self, create_test_data):
        """Test split_reduce with mean function."""
        vec, fac = create_test_data
        
        # Apply split_reduce with mean
        result = split_reduce(vec, fac, np.mean)
        
        # Expected values:
        # For voxel 0: mean([1,2])=1.5, mean([3,4])=3.5, overall mean=2.5
        # For voxel 1: mean([5,6])=5.5, mean([7,8])=7.5, overall mean=6.5
        # For voxel 2: mean([9,10])=9.5, mean([11,12])=11.5, overall mean=10.5
        
        expected = np.array([[[2.5, 6.5, 10.5]]])
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_split_reduce_uses_fortran_voxel_order(self):
        """split_reduce should map voxel indices with Fortran indexing."""
        space = NeuroSpace((2, 2, 2, 2), spacing=(1, 1, 1, 1))
        data = np.arange(np.prod(space.dim), dtype=float).reshape(space.dim, order="F")
        vec = DenseNeuroVec(data, space)

        factor = np.array([0, 1])
        result = split_reduce(vec, factor, np.mean)

        expected = np.zeros(space.dim[:3])
        data_2d = data.reshape(-1, space.dim[-1], order="F")
        for vox_idx in range(data_2d.shape[0]):
            coords = np.array(np.unravel_index(vox_idx, space.dim[:3], order="F"))
            expected[tuple(coords)] = np.mean(data_2d[vox_idx])

        assert result.shape == tuple(space.dim[:3])
        np.testing.assert_array_equal(result.data, expected)
    
    def test_split_scale_basic(self, create_test_data):
        """Test split_scale with center and scale."""
        vec, fac = create_test_data
        
        # Apply split_scale
        result = split_scale(vec, fac, center=True, scale=True)
        
        # Check result type and shape
        assert isinstance(result, DenseNeuroVec)
        assert result.shape == vec.shape
        
        # Expected behavior: within each factor level, data is centered and scaled
        # For factor level 1 (indices 0,1):
        #   Voxel 0: [1,2] -> centered: [-0.5, 0.5] -> scaled: [-1, 1] * 1/sqrt(0.5)
        #   Voxel 1: [5,6] -> centered: [-0.5, 0.5] -> scaled: [-1, 1] * 1/sqrt(0.5)
        #   Voxel 2: [9,10] -> centered: [-0.5, 0.5] -> scaled: [-1, 1] * 1/sqrt(0.5)
        
        # Check that within each factor level, mean is 0 and std is 1
        for level in np.unique(fac):
            level_mask = fac == level
            for voxel in range(3):
                voxel_data = result.data[0, 0, voxel, level_mask]
                np.testing.assert_almost_equal(np.mean(voxel_data), 0, decimal=10)
                np.testing.assert_almost_equal(np.std(voxel_data, ddof=0), 1, decimal=10)
    
    def test_split_scale_center_only(self, create_test_data):
        """Test split_scale with only centering."""
        vec, fac = create_test_data
        
        # Apply split_scale with center only
        result = split_scale(vec, fac, center=True, scale=False)
        
        # Check that within each factor level, mean is 0
        for level in np.unique(fac):
            level_mask = fac == level
            for voxel in range(3):
                voxel_data = result.data[0, 0, voxel, level_mask]
                np.testing.assert_almost_equal(np.mean(voxel_data), 0, decimal=10)
    
    def test_split_scale_scale_only(self, create_test_data):
        """Test split_scale with only scaling."""
        vec, fac = create_test_data
        
        # Apply split_scale with scale only
        result = split_scale(vec, fac, center=False, scale=True)
        
        # Original data should be divided by SD within each level
        for level in np.unique(fac):
            level_mask = fac == level
            for voxel in range(3):
                original_data = vec.data[0, 0, voxel, level_mask]
                scaled_data = result.data[0, 0, voxel, level_mask]
                original_std = np.std(original_data)
                # Check that scaling was applied correctly
                np.testing.assert_array_almost_equal(
                    scaled_data, original_data / original_std
                )
    
    def test_split_reduce_larger_data(self):
        """Test split_reduce with larger, more realistic data."""
        # Create 10x10x10x20 data
        space_4d = NeuroSpace((10, 10, 10, 20), spacing=(1, 1, 1, 1))
        data = np.random.randn(10, 10, 10, 20)
        vec = DenseNeuroVec(data, space_4d)
        
        # Create factor with 4 levels
        fac = np.repeat([1, 2, 3, 4], 5)
        
        # Apply split_reduce with std function
        result = split_reduce(vec, fac, np.std)
        
        assert result.shape == (10, 10, 10)
        assert not np.any(np.isnan(result.data))
    
    def test_split_scale_larger_data(self):
        """Test split_scale with larger data."""
        # Create 10x10x10x20 data
        space_4d = NeuroSpace((10, 10, 10, 20), spacing=(1, 1, 1, 1))
        data = np.random.randn(10, 10, 10, 20) * 10 + 50
        vec = DenseNeuroVec(data, space_4d)
        
        # Create factor with 4 levels
        fac = np.repeat([1, 2, 3, 4], 5)
        
        # Apply split_scale
        result = split_scale(vec, fac, center=True, scale=True)
        
        # Check standardization within each level
        for level in [1, 2, 3, 4]:
            level_mask = fac == level
            level_data = result.data[:, :, :, level_mask]
            
            # Mean should be close to 0
            voxel_means = np.mean(level_data, axis=3)
            assert np.allclose(voxel_means, 0, atol=1e-10)
            
            # SD should be close to 1
            voxel_stds = np.std(level_data, axis=3)
            assert np.allclose(voxel_stds, 1, atol=1e-10)
    
    def test_split_reduce_error_handling(self, create_test_data):
        """Test error handling in split_reduce."""
        vec, fac = create_test_data
        
        # Wrong factor length
        wrong_fac = np.array([1, 2, 3])  # Too short
        with pytest.raises(ValueError, match="Length of factor"):
            split_reduce(vec, wrong_fac, np.sum)
    
    def test_split_scale_error_handling(self, create_test_data):
        """Test error handling in split_scale."""
        vec, fac = create_test_data
        
        # Wrong factor length
        wrong_fac = np.array([1, 2, 3])  # Too short
        with pytest.raises(ValueError, match="Length of factor"):
            split_scale(vec, wrong_fac)
    
    def test_split_reduce_with_sparse(self):
        """Test split_reduce with sparse data."""
        # Create sparse test data
        space_4d = NeuroSpace((10, 10, 10, 4), spacing=(1, 1, 1, 1))
        
        # Only 100 active voxels
        n_active = 100
        sparse_data = np.random.randn(4, n_active)  # time x voxels
        indices = np.random.choice(1000, n_active, replace=False)
        
        vec = SparseNeuroVec(sparse_data, space_4d, indices)
        fac = np.array([1, 1, 2, 2])
        
        # Apply split_reduce
        result = split_reduce(vec, fac, np.mean)
        
        # Should return a sparse volume
        assert hasattr(result, 'data') or hasattr(result, '_data')
    
    def test_split_scale_constant_values(self):
        """Test split_scale with constant values (edge case)."""
        # Create data where some voxels have constant values
        space_4d = NeuroSpace((2, 2, 2, 10), spacing=(1, 1, 1, 1))
        data = np.ones((2, 2, 2, 10))
        data[0, 0, 0, :] = 5  # Constant value of 5
        data[1, 1, 1, :] = np.arange(10)  # Varying values
        
        vec = DenseNeuroVec(data, space_4d)
        fac = np.repeat([1, 2], 5)
        
        # Apply split_scale - should handle zero variance gracefully
        result = split_scale(vec, fac, center=True, scale=True)
        
        # Constant voxel should be centered but not scaled (avoid div by 0)
        const_voxel = result.data[0, 0, 0, :]
        assert np.allclose(const_voxel[fac == 1], 0)  # Centered to 0
        assert np.allclose(const_voxel[fac == 2], 0)  # Centered to 0
    
    def test_split_functions_integration(self):
        """Test integration with other split functions."""
        # Create test data
        space_4d = NeuroSpace((10, 10, 10, 20), spacing=(1, 1, 1, 1))
        data = np.random.randn(10, 10, 10, 20)
        vec = DenseNeuroVec(data, space_4d)
        
        # Test split_blocks
        blocks = split_blocks(vec, nblocks=4)
        assert len(blocks) == 4
        
        # Test split_clusters if we have a mask
        mask_data = np.ones((10, 10, 10), dtype=bool)
        mask = LogicalNeuroVol(mask_data, vec.space.get_subspace(range(3)))
        
        clusters = split_clusters(vec, mask=mask, k=5)
        assert len(clusters) == 5
