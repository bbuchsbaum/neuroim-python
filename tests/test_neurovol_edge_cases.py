"""
Comprehensive edge case tests for NeuroVol operations
Tests boundary conditions, invalid inputs, and special cases
"""

import pytest
import numpy as np
from neuroimpy import NeuroSpace
from neuroimpy.neuro_vol import DenseNeuroVol, LogicalNeuroVol, SparseNeuroVol
import warnings


class TestNeuroVolEdgeCases:
    """Test NeuroVol edge cases and boundary conditions"""
    
    def test_empty_volume(self):
        """Test operations on empty volumes"""
        # Zero-sized volume
        with pytest.raises(ValueError):
            space = NeuroSpace(dim=(0, 0, 0))
            DenseNeuroVol(np.array([]), space)
    
    def test_single_voxel_volume(self):
        """Test operations on single voxel volumes"""
        space = NeuroSpace(dim=(1, 1, 1))
        data = np.array([[[42]]])
        vol = DenseNeuroVol(data, space)
        
        # Test indexing
        assert vol[0, 0, 0] == 42
        
        # Test arithmetic
        vol2 = vol + 10
        assert vol2[0, 0, 0] == 52
        
        # Test slicing
        assert vol[:, :, :].shape == (1, 1, 1)
    
    def test_extreme_dimensions(self):
        """Test volumes with extreme dimensions"""
        # Very large along one dimension
        space = NeuroSpace(dim=(1000, 1, 1))
        data = np.random.randn(1000, 1, 1)
        vol = DenseNeuroVol(data, space)
        
        # Test memory-efficient operations
        assert vol.shape == (1000, 1, 1)
        assert vol[500, 0, 0] == data[500, 0, 0]
        
        # Very small spacing
        space = NeuroSpace(dim=(10, 10, 10), spacing=(0.001, 0.001, 0.001))
        data = np.ones((10, 10, 10))
        vol = DenseNeuroVol(data, space)
        assert np.allclose(vol.space.spacing, [0.001, 0.001, 0.001])
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        space = NeuroSpace(dim=(10, 10, 10))
        
        # Wrong data shape
        with pytest.raises(ValueError):
            DenseNeuroVol(np.ones((5, 5, 5)), space)
        
        # Invalid data type (complex numbers)
        complex_data = np.ones((10, 10, 10), dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol = DenseNeuroVol(complex_data, space)
            # Should convert to real
            assert vol.data.dtype in [np.float32, np.float64]
    
    def test_boundary_indexing(self):
        """Test indexing at volume boundaries"""
        space = NeuroSpace(dim=(10, 10, 10))
        data = np.arange(1000).reshape(10, 10, 10)
        vol = DenseNeuroVol(data, space)
        
        # Valid boundary indices
        assert vol[0, 0, 0] == data[0, 0, 0]
        assert vol[9, 9, 9] == data[9, 9, 9]
        
        # Invalid indices
        with pytest.raises(IndexError):
            vol[10, 0, 0]
        
        with pytest.raises(IndexError):
            vol[0, 0, -11]  # Too negative
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values"""
        space = NeuroSpace(dim=(5, 5, 5))
        
        # Create data with NaN and Inf
        data = np.ones((5, 5, 5))
        data[0, 0, 0] = np.nan
        data[1, 1, 1] = np.inf
        data[2, 2, 2] = -np.inf
        
        vol = DenseNeuroVol(data, space)
        
        # Test that values are preserved
        assert np.isnan(vol[0, 0, 0])
        assert np.isinf(vol[1, 1, 1])
        assert np.isneginf(vol[2, 2, 2])
        
        # Test arithmetic with NaN/Inf
        vol2 = vol + 10
        assert np.isnan(vol2[0, 0, 0])
        assert np.isinf(vol2[1, 1, 1])
        
        # Test comparison operations
        mask = vol > 0
        assert not mask[0, 0, 0]  # NaN comparison is False
    
    def test_all_false_mask(self):
        """Test operations with all-false masks"""
        space = NeuroSpace(dim=(10, 10, 10))
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)
        
        # Operations should handle empty masks gracefully
        data = np.ones((10, 10, 10))
        vol = DenseNeuroVol(data, space)
        
        # Sparse volume with empty mask
        sparse_vol = SparseNeuroVol(mask=mask, data=np.array([]), space=space)
        assert sparse_vol.data.shape == (0,)
    
    def test_all_true_mask(self):
        """Test operations with all-true masks"""
        space = NeuroSpace(dim=(5, 5, 5))
        mask_data = np.ones((5, 5, 5), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)
        
        # Should be equivalent to dense volume
        dense_data = np.arange(125).reshape(5, 5, 5)
        sparse_data = dense_data.ravel(order="F")[mask_data.ravel(order="F")]
        
        sparse_vol = SparseNeuroVol(mask=mask, data=sparse_data, space=space)
        
        # Verify all values accessible
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    assert sparse_vol[i, j, k] == dense_data[i, j, k]
    
    def test_dtype_preservation(self):
        """Test that data types are preserved correctly"""
        space = NeuroSpace(dim=(5, 5, 5))
        
        # Test different dtypes
        for dtype in [np.int8, np.int16, np.int32, np.float32, np.float64]:
            data = np.ones((5, 5, 5), dtype=dtype)
            vol = DenseNeuroVol(data, space)
            
            # Check dtype is preserved
            assert vol.data.dtype == dtype
    
    def test_arithmetic_edge_cases(self):
        """Test arithmetic operations edge cases"""
        space = NeuroSpace(dim=(3, 3, 3))
        
        # Division by zero
        data = np.ones((3, 3, 3))
        vol = DenseNeuroVol(data, space)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol_div = vol / 0
            assert np.all(np.isinf(vol_div.data))
        
        # Very large numbers
        large_data = np.full((3, 3, 3), 1e308)
        vol_large = DenseNeuroVol(large_data, space)
        
        # Multiplication could overflow
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol_mult = vol_large * 10
            # Should handle overflow gracefully
    
    def test_slice_edge_cases(self):
        """Test edge cases in slicing operations"""
        space = NeuroSpace(dim=(10, 10, 10))
        data = np.arange(1000).reshape(10, 10, 10)
        vol = DenseNeuroVol(data, space)
        
        # Empty slices
        empty_slice = vol[5:5, :, :]
        assert empty_slice.shape == (0, 10, 10)
        
        # Reversed slices
        reversed_slice = vol[5:2:-1, :, :]
        assert reversed_slice.shape == (3, 10, 10)  # indices 5, 4, 3
        
        # Step slices
        step_slice = vol[::3, ::3, ::3]
        assert step_slice.shape == (4, 4, 4)
    
    def test_coordinate_edge_cases(self):
        """Test edge cases in coordinate operations"""
        # Non-standard origin
        space = NeuroSpace(dim=(10, 10, 10), 
                          spacing=(2, 2, 2),
                          origin=(-20, -20, -20))
        
        data = np.ones((10, 10, 10))
        vol = DenseNeuroVol(data, space)
        
        # Test that coordinates are computed correctly
        coords = vol.space.grid_to_world(np.array([0, 0, 0]))
        np.testing.assert_array_equal(coords, [-20, -20, -20])
        
        coords = vol.space.grid_to_world(np.array([9, 9, 9]))
        np.testing.assert_array_equal(coords, [-2, -2, -2])
    
    def test_memory_efficiency(self):
        """Test memory efficiency for large volumes"""
        # Create a large volume
        space = NeuroSpace(dim=(100, 100, 100))
        data = np.ones((100, 100, 100), dtype=np.float32)
        vol = DenseNeuroVol(data, space)
        
        # Test that views don't copy data
        view = vol[50:60, 50:60, 50:60]
        view.data[0, 0, 0] = 999
        
        # Original should be modified if it's a view
        assert vol[50, 50, 50] == 999
    
    def test_comparison_edge_cases(self):
        """Test comparison operations with edge cases"""
        space = NeuroSpace(dim=(5, 5, 5))
        
        # Data with NaN
        data1 = np.ones((5, 5, 5))
        data1[0, 0, 0] = np.nan
        vol1 = DenseNeuroVol(data1, space)
        
        data2 = np.ones((5, 5, 5))
        vol2 = DenseNeuroVol(data2, space)
        
        # Comparison with NaN
        result = vol1 > vol2
        assert not result[0, 0, 0]  # NaN comparison is False
        
        # Comparison with Inf
        data1[1, 1, 1] = np.inf
        vol1 = DenseNeuroVol(data1, space)
        result = vol1 > vol2
        assert result[1, 1, 1]  # Inf > 1 is True


class TestSparseNeuroVolEdgeCases:
    """Test SparseNeuroVol edge cases"""
    
    def test_sparse_single_voxel(self):
        """Test sparse volume with single active voxel"""
        space = NeuroSpace(dim=(10, 10, 10))
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[5, 5, 5] = True
        mask = LogicalNeuroVol(mask_data, space)
        
        sparse_vol = SparseNeuroVol(mask=mask, data=np.array([42]), space=space)
        
        # Test access
        assert sparse_vol[5, 5, 5] == 42
        assert sparse_vol[0, 0, 0] == 0  # Non-mask voxel
    
    def test_sparse_diagonal_pattern(self):
        """Test sparse volume with diagonal pattern"""
        space = NeuroSpace(dim=(10, 10, 10))
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        
        # Create diagonal pattern
        for i in range(10):
            if i < 10:
                mask_data[i, i, i] = True
        
        mask = LogicalNeuroVol(mask_data, space)
        data = np.arange(10) * 10
        
        sparse_vol = SparseNeuroVol(mask=mask, data=data, space=space)
        
        # Verify diagonal values
        for i in range(10):
            assert sparse_vol[i, i, i] == i * 10
    
    def test_sparse_update_edge_cases(self):
        """Test updating sparse volume values"""
        space = NeuroSpace(dim=(5, 5, 5))
        mask_data = np.random.rand(5, 5, 5) > 0.8
        mask = LogicalNeuroVol(mask_data, space)
        
        n_active = mask_data.sum()
        data = np.ones(n_active)
        sparse_vol = SparseNeuroVol(mask=mask, data=data, space=space)
        
        # Test arithmetic that changes values
        sparse_vol2 = sparse_vol * 0
        assert np.all(sparse_vol2.data == 0)
        
        # Test with NaN
        sparse_vol.data[0] = np.nan
        # Should handle NaN in sparse data


class TestLogicalNeuroVolEdgeCases:
    """Test LogicalNeuroVol edge cases"""
    
    def test_logical_operations_edge_cases(self):
        """Test logical operations with edge cases"""
        space = NeuroSpace(dim=(5, 5, 5))
        
        # All true
        mask1 = LogicalNeuroVol(np.ones((5, 5, 5), dtype=bool), space)
        # All false
        mask2 = LogicalNeuroVol(np.zeros((5, 5, 5), dtype=bool), space)
        
        # AND operation
        result = mask1 & mask2
        assert not np.any(result.data)
        
        # OR operation
        result = mask1 | mask2
        assert np.all(result.data)
        
        # XOR operation
        result = mask1 ^ mask2
        assert np.all(result.data)
    
    def test_mask_from_threshold_edge_cases(self):
        """Test creating masks from thresholds with edge cases"""
        space = NeuroSpace(dim=(5, 5, 5))
        
        # Data with NaN
        data = np.ones((5, 5, 5))
        data[0, 0, 0] = np.nan
        vol = DenseNeuroVol(data, space)
        
        # Threshold with NaN
        mask = vol > 0.5
        assert not mask[0, 0, 0]  # NaN comparison is False
        assert mask[1, 1, 1]  # Normal comparison
        
        # Data with Inf
        data[1, 1, 1] = np.inf
        vol = DenseNeuroVol(data, space)
        mask = vol > 0.5
        assert mask[1, 1, 1]  # Inf > 0.5 is True
