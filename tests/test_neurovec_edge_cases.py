"""
Comprehensive edge case tests for NeuroVec operations
Tests boundary conditions, invalid inputs, and special cases for 4D data
"""

import pytest
import numpy as np
from neuroimpy import NeuroSpace
from neuroimpy.neuro_vec import DenseNeuroVec, SparseNeuroVec
from neuroimpy.neuro_vol import LogicalNeuroVol
import warnings


class TestNeuroVecEdgeCases:
    """Test NeuroVec edge cases and boundary conditions"""
    
    def test_single_timepoint_vec(self):
        """Test NeuroVec with single time point"""
        space = NeuroSpace(dim=(10, 10, 10, 1))
        data = np.random.randn(10, 10, 10, 1)
        vec = DenseNeuroVec(data, space)
        
        # Test series extraction
        ts = vec.series_3d(5, 5, 5)
        assert ts.shape == (1,)
        assert ts[0] == data[5, 5, 5, 0]
        
        # Test slicing
        vol = vec[..., 0]
        assert vol.shape == (10, 10, 10)
    
    def test_single_voxel_vec(self):
        """Test NeuroVec with single spatial voxel"""
        space = NeuroSpace(dim=(1, 1, 1, 100))
        data = np.random.randn(1, 1, 1, 100)
        vec = DenseNeuroVec(data, space)
        
        # Test series extraction
        ts = vec.series_3d(0, 0, 0)
        assert ts.shape == (100,)
        np.testing.assert_array_equal(ts, data[0, 0, 0, :])
    
    def test_extreme_time_dimensions(self):
        """Test NeuroVec with extreme time dimensions"""
        # Very long time series
        space = NeuroSpace(dim=(5, 5, 5, 10000))
        data = np.random.randn(5, 5, 5, 10000).astype(np.float32)
        vec = DenseNeuroVec(data, space)
        
        # Test memory efficiency
        ts = vec.series_3d(2, 2, 2)
        assert ts.shape == (10000,)
        assert ts.dtype == np.float32
    
    def test_invalid_vec_inputs(self):
        """Test handling of invalid inputs for NeuroVec"""
        # Wrong number of dimensions
        with pytest.raises(ValueError):
            space = NeuroSpace(dim=(10, 10, 10))  # 3D space
            data = np.ones((10, 10, 10, 20))  # 4D data
            DenseNeuroVec(data, space)
        
        # Mismatched dimensions
        with pytest.raises(ValueError):
            space = NeuroSpace(dim=(10, 10, 10, 20))
            data = np.ones((5, 5, 5, 20))
            DenseNeuroVec(data, space)
    
    def test_series_extraction_edge_cases(self):
        """Test edge cases in series extraction"""
        space = NeuroSpace(dim=(10, 10, 10, 50))
        data = np.random.randn(10, 10, 10, 50)
        vec = DenseNeuroVec(data, space)
        
        # Empty coordinate list
        empty_coords = np.array([]).reshape(0, 3)
        ts = vec.series(empty_coords)
        assert ts.shape == (50, 0)
        
        # Single coordinate as array
        single_coord = np.array([[5, 5, 5]])
        ts = vec.series(single_coord)
        assert ts.shape == (50, 1)
        
        # Duplicate coordinates
        dup_coords = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
        ts = vec.series(dup_coords)
        assert ts.shape == (50, 3)
        assert np.all(ts[:, 0] == ts[:, 1])
    
    def test_time_slicing_edge_cases(self):
        """Test edge cases in time slicing"""
        space = NeuroSpace(dim=(5, 5, 5, 100))
        data = np.arange(5*5*5*100).reshape(5, 5, 5, 100)
        vec = DenseNeuroVec(data, space)
        
        # Single time point slice - returns DenseNeuroVol
        vol = vec[..., 50]
        assert vol.shape == (5, 5, 5)

        # Empty time slice
        empty = vec[..., 100:100]
        assert empty.shape[-1] == 0

        # Reversed time slice: indices 10, 9, 8, 7, 6
        reversed_slice = vec[..., 10:5:-1]
        assert reversed_slice.shape[-1] == 5
        
        # Step slicing in time
        step_slice = vec[..., ::10]
        assert step_slice.shape == (5, 5, 5, 10)
    
    def test_nan_inf_in_time_series(self):
        """Test handling of NaN and Inf in time series data"""
        space = NeuroSpace(dim=(3, 3, 3, 10))
        data = np.ones((3, 3, 3, 10))
        
        # Add NaN and Inf
        data[1, 1, 1, 0] = np.nan
        data[1, 1, 1, 5] = np.inf
        data[1, 1, 1, 9] = -np.inf
        
        vec = DenseNeuroVec(data, space)
        
        # Extract series with special values
        ts = vec.series_3d(1, 1, 1)
        assert np.isnan(ts[0])
        assert np.isinf(ts[5])
        assert np.isneginf(ts[9])
        
        # Operations should preserve special values
        vec2 = vec * 2
        ts2 = vec2.series_3d(1, 1, 1)
        assert np.isnan(ts2[0])
        assert np.isinf(ts2[5])
    
    def test_boundary_coordinates(self):
        """Test series extraction at volume boundaries"""
        space = NeuroSpace(dim=(10, 10, 10, 20))
        data = np.random.randn(10, 10, 10, 20)
        vec = DenseNeuroVec(data, space)
        
        # All corners of the volume
        corners = np.array([
            [0, 0, 0],
            [0, 0, 9],
            [0, 9, 0],
            [0, 9, 9],
            [9, 0, 0],
            [9, 0, 9],
            [9, 9, 0],
            [9, 9, 9]
        ])
        
        ts = vec.series(corners)
        assert ts.shape == (20, 8)
        
        # Out of bounds coords return zeros (graceful handling)
        oob = vec.series(np.array([[10, 5, 5]]))
        assert oob.shape[0] == 20  # time dimension preserved
    
    def test_memory_patterns(self):
        """Test different memory access patterns"""
        space = NeuroSpace(dim=(50, 50, 50, 100))
        data = np.random.randn(50, 50, 50, 100).astype(np.float32)
        vec = DenseNeuroVec(data, space)
        
        # Contiguous access
        ts1 = vec.series_3d(25, 25, 25)
        
        # Non-contiguous access (multiple voxels)
        coords = np.array([[i, i, i] for i in range(0, 50, 5)])
        ts2 = vec.series(coords)
        assert ts2.shape == (100, 10)
        
        # Verify data integrity
        for i, coord in enumerate(coords):
            expected = data[coord[0], coord[1], coord[2], :]
            np.testing.assert_array_equal(ts2[:, i], expected)
    
    def test_arithmetic_broadcasting(self):
        """Test arithmetic operations with broadcasting edge cases"""
        space = NeuroSpace(dim=(5, 5, 5, 10))
        data = np.ones((5, 5, 5, 10))
        vec = DenseNeuroVec(data, space)
        
        # Scalar operations
        vec2 = vec + 0
        np.testing.assert_array_equal(vec.data, vec2.data)
        
        # Time-varying operations (broadcast along time)
        time_weights = np.arange(10)
        vec3 = vec * time_weights
        assert vec3.series_3d(0, 0, 0)[5] == 5  # 1 * 5
        
        # Spatial operations (broadcast along space)
        spatial_weights = np.ones((5, 5, 5, 1))
        vec4 = vec * spatial_weights
        np.testing.assert_array_equal(vec.data, vec4.data)


class TestSparseNeuroVecEdgeCases:
    """Test SparseNeuroVec edge cases"""
    
    def test_sparse_vec_single_timepoint(self):
        """Test sparse vec with single time point"""
        space_3d = NeuroSpace(dim=(10, 10, 10))
        mask_data = np.random.rand(10, 10, 10) > 0.7
        mask = LogicalNeuroVol(mask_data, space_3d)
        
        space_4d = NeuroSpace(dim=(10, 10, 10, 1))
        n_active = mask_data.sum()
        data = np.random.randn(1, n_active)  # Single timepoint
        
        sparse_vec = SparseNeuroVec(data=data, mask=mask, space=space_4d)
        
        # Test series extraction
        if mask_data[5, 5, 5]:
            ts = sparse_vec.series_3d(5, 5, 5)
            assert ts.shape == (1,)
    
    def test_sparse_vec_all_zero_data(self):
        """Test sparse vec with all zero data"""
        space_3d = NeuroSpace(dim=(5, 5, 5))
        mask_data = np.ones((5, 5, 5), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space_3d)
        
        space_4d = NeuroSpace(dim=(5, 5, 5, 20))
        data = np.zeros((20, 125))  # All zeros
        
        sparse_vec = SparseNeuroVec(data=data, mask=mask, space=space_4d)
        
        # All series should be zero
        ts = sparse_vec.series_3d(2, 2, 2)
        assert np.all(ts == 0)
    
    def test_sparse_vec_extreme_sparsity(self):
        """Test sparse vec with very few active voxels"""
        space_3d = NeuroSpace(dim=(100, 100, 100))
        mask_data = np.zeros((100, 100, 100), dtype=bool)
        # Only 3 voxels active
        mask_data[10, 10, 10] = True
        mask_data[50, 50, 50] = True
        mask_data[90, 90, 90] = True
        mask = LogicalNeuroVol(mask_data, space_3d)
        
        space_4d = NeuroSpace(dim=(100, 100, 100, 1000))
        data = np.random.randn(1000, 3)
        
        sparse_vec = SparseNeuroVec(data=data, mask=mask, space=space_4d)
        
        # Test access to active voxels
        ts1 = sparse_vec.series_3d(10, 10, 10)
        ts2 = sparse_vec.series_3d(50, 50, 50)
        ts3 = sparse_vec.series_3d(90, 90, 90)
        
        assert ts1.shape == (1000,)
        assert ts2.shape == (1000,)
        assert ts3.shape == (1000,)
        
        # Test access to inactive voxel
        ts_zero = sparse_vec.series_3d(0, 0, 0)
        assert np.all(ts_zero == 0)
    
    def test_sparse_vec_coordinate_edge_cases(self):
        """Test coordinate handling in sparse vecs"""
        space_3d = NeuroSpace(dim=(10, 10, 10))
        mask_data = np.random.rand(10, 10, 10) > 0.5
        mask = LogicalNeuroVol(mask_data, space_3d)
        
        space_4d = NeuroSpace(dim=(10, 10, 10, 50))
        n_active = mask_data.sum()
        data = np.random.randn(50, n_active)
        
        sparse_vec = SparseNeuroVec(data=data, mask=mask, space=space_4d)
        
        # Get coordinates of active voxels
        active_coords = np.argwhere(mask_data)
        
        # Extract all active voxel series
        all_series = sparse_vec.series(active_coords)
        assert all_series.shape == (50, n_active)
        
        # Verify correct mapping
        for i, coord in enumerate(active_coords):
            single_ts = sparse_vec.series_3d(coord[0], coord[1], coord[2])
            np.testing.assert_array_equal(all_series[:, i], single_ts)
    
    def test_sparse_vec_dtype_handling(self):
        """Test data type handling in sparse vecs"""
        space_3d = NeuroSpace(dim=(5, 5, 5))
        mask_data = np.ones((5, 5, 5), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space_3d)
        
        space_4d = NeuroSpace(dim=(5, 5, 5, 10))
        
        # Test different dtypes
        for dtype in [np.float32, np.float64, np.int32]:
            data = np.ones((10, 125), dtype=dtype)
            sparse_vec = SparseNeuroVec(data=data, mask=mask, space=space_4d)
            
            ts = sparse_vec.series_3d(0, 0, 0)
            # Check dtype is preserved or appropriately promoted
            assert ts.dtype in [dtype, np.float32, np.float64]