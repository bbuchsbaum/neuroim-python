"""
Comprehensive tests for NeuroHyperVec (5D+ support)
Tests multi-dimensional neuroimaging data with features beyond time
"""

import pytest
import numpy as np
from neuroimpy import NeuroSpace
from neuroimpy.neuro_hypervec import DenseNeuroHyperVec, SparseNeuroHyperVec
from neuroimpy.neuro_vec import DenseNeuroVec
from neuroimpy.neuro_vol import LogicalNeuroVol
import warnings


class TestDenseNeuroHyperVec:
    """Test dense 5D+ neuroimaging hypervectors"""
    
    def test_basic_5d_creation(self):
        """Test creating a basic 5D hypervector"""
        # 5D: x, y, z, time, features
        space = NeuroSpace(dim=(10, 10, 10, 20, 5),
                          spacing=(2, 2, 2, 0.5, 1),
                          origin=(0, 0, 0, 0, 0))
        
        data = np.random.randn(10, 10, 10, 20, 5)
        hvec = DenseNeuroHyperVec(data, space, label="test_5d")
        
        # Test properties
        assert hvec.shape == (10, 10, 10, 20, 5)
        assert hvec.n_dims == 5
        assert hvec.spatial_shape == (10, 10, 10)
        assert hvec.n_timepoints == 20
        assert hvec.n_features == 5
        assert hvec.label == "test_5d"
    
    def test_invalid_dimensions(self):
        """Test that < 5D raises error"""
        with pytest.raises(ValueError):
            space = NeuroSpace(dim=(10, 10, 10, 20))  # Only 4D
            data = np.ones((10, 10, 10, 20))
            DenseNeuroHyperVec(data, space)
    
    def test_shape_mismatch(self):
        """Test that mismatched data/space shapes raise error"""
        space = NeuroSpace(dim=(10, 10, 10, 20, 5))
        data = np.ones((10, 10, 10, 20, 3))  # Wrong feature dimension
        
        with pytest.raises(ValueError):
            DenseNeuroHyperVec(data, space)
    
    def test_single_feature_extraction(self):
        """Test extracting a single feature returns NeuroVec"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 3))
        data = np.random.randn(5, 5, 5, 10, 3)
        hvec = DenseNeuroHyperVec(data, space)
        
        # Extract first feature using full indexing
        vec = hvec[:, :, :, :, 0]
        
        # Should return a NeuroVec
        assert isinstance(vec, DenseNeuroVec)
        assert vec.shape == (5, 5, 5, 10)
        np.testing.assert_array_equal(vec.data, data[:, :, :, :, 0])
    
    def test_series_extraction_single_voxel(self):
        """Test time series extraction for single voxel"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 3))
        data = np.arange(5*5*5*10*3).reshape(5, 5, 5, 10, 3)
        hvec = DenseNeuroHyperVec(data, space)
        
        # Extract time series for voxel (2, 2, 2), all features
        ts = hvec.series([2, 2, 2])
        assert ts.shape == (10, 3)  # time x features
        
        # Extract time series for specific feature
        ts_f1 = hvec.series([2, 2, 2], feature=1)
        assert ts_f1.shape == (10,)  # time only
        np.testing.assert_array_equal(ts_f1, data[2, 2, 2, :, 1])
    
    def test_series_extraction_multiple_voxels(self):
        """Test time series extraction for multiple voxels"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 3))
        data = np.random.randn(5, 5, 5, 10, 3)
        hvec = DenseNeuroHyperVec(data, space)
        
        # Multiple voxel coordinates
        coords = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        
        # All features
        ts = hvec.series(coords)
        assert ts.shape == (10, 3, 3)  # time x features x voxels
        
        # Specific feature
        ts_f0 = hvec.series(coords, feature=0)
        assert ts_f0.shape == (10, 3)  # time x voxels
    
    def test_mean_features(self):
        """Test averaging across feature dimension"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 4))
        # Create data where features have different values
        data = np.zeros((5, 5, 5, 10, 4))
        for i in range(4):
            data[..., i] = i  # Feature i has value i everywhere
        
        hvec = DenseNeuroHyperVec(data, space)
        
        # Average across features
        mean_vec = hvec.mean_features()
        
        # Should be a 4D NeuroVec
        assert isinstance(mean_vec, DenseNeuroVec)
        assert mean_vec.shape == (5, 5, 5, 10)
        
        # Mean should be (0+1+2+3)/4 = 1.5 everywhere
        expected = np.ones((5, 5, 5, 10)) * 1.5
        np.testing.assert_allclose(mean_vec.data, expected)
    
    def test_select_features(self):
        """Test selecting subset of features"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 6))
        data = np.random.randn(5, 5, 5, 10, 6)
        hvec = DenseNeuroHyperVec(data, space)
        
        # Select features 0, 2, 4
        subset = hvec.select_features([0, 2, 4])
        
        assert isinstance(subset, DenseNeuroHyperVec)
        assert subset.shape == (5, 5, 5, 10, 3)
        assert subset.n_features == 3
        
        # Check data matches
        expected = data[..., [0, 2, 4]]
        np.testing.assert_array_equal(subset.data, expected)
    
    def test_apply_feature_func(self):
        """Test applying function across features"""
        space = NeuroSpace(dim=(3, 3, 3, 5, 4))
        data = np.random.randn(3, 3, 3, 5, 4)
        hvec = DenseNeuroHyperVec(data, space)
        
        # Apply max across features
        max_vec = hvec.apply_feature_func(np.max)
        
        assert isinstance(max_vec, DenseNeuroVec)
        assert max_vec.shape == (3, 3, 3, 5)
        
        # Verify max is computed correctly
        expected = np.max(data, axis=-1)
        np.testing.assert_array_equal(max_vec.data, expected)
    
    def test_large_feature_dims(self):
        """Test hypervectors with many features"""
        # Test that we support large feature dimensions
        space = NeuroSpace(dim=(3, 3, 3, 5, 50))
        data = np.random.randn(3, 3, 3, 5, 50)
        hvec = DenseNeuroHyperVec(data, space)
        
        assert hvec.shape == (3, 3, 3, 5, 50)
        assert hvec.n_dims == 5
        assert hvec.n_features == 50
        
        # Operations should still work
        mean_vec = hvec.mean_features()
        assert mean_vec.shape == (3, 3, 3, 5)
    
    def test_indexing_edge_cases(self):
        """Test edge cases in indexing"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 3))
        data = np.arange(5*5*5*10*3).reshape(5, 5, 5, 10, 3)
        hvec = DenseNeuroHyperVec(data, space)
        
        # Single voxel, all time, all features
        subset = hvec[2, 2, 2, :, :]
        assert subset.shape == (10, 3)
        
        # Extract single feature - should return NeuroVec
        vec = hvec[:, :, :, :, 0]
        assert isinstance(vec, DenseNeuroVec)
        assert vec.shape == (5, 5, 5, 10)
        
        # Slice features - returns another array
        subset = hvec[..., 1:3]
        assert subset.shape == (5, 5, 5, 10, 2)
        assert isinstance(subset, np.ndarray)
    
    def test_memory_layout(self):
        """Test memory layout handling"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 3))
        
        # C-order (row-major)
        data_c = np.ones((5, 5, 5, 10, 3), order='C')
        hvec_c = DenseNeuroHyperVec(data_c, space)
        assert hvec_c.data.flags['C_CONTIGUOUS']
        
        # F-order (column-major) should be converted
        data_f = np.ones((5, 5, 5, 10, 3), order='F')
        hvec_f = DenseNeuroHyperVec(data_f, space)
        assert hvec_f.data.flags['C_CONTIGUOUS']


class TestSparseNeuroHyperVec:
    """Test sparse 5D+ neuroimaging hypervectors"""
    
    def test_sparse_hypervec_creation(self):
        """Test creating sparse hypervector"""
        # Create mask
        mask_space = NeuroSpace(dim=(10, 10, 10))
        mask_data = np.random.rand(10, 10, 10) > 0.7
        mask = LogicalNeuroVol(mask_data, mask_space)
        
        # Create 5D space
        space_5d = NeuroSpace(dim=(10, 10, 10, 20, 5))
        
        # Sparse data: features x time x voxels
        n_active = mask_data.sum()
        data = np.random.randn(5, 20, n_active)
        
        hvec = SparseNeuroHyperVec(data=data, mask=mask, space=space_5d)
        
        assert hvec.shape == (10, 10, 10, 20, 5)
        assert hvec.n_features == 5
        assert hvec.data.shape == (5, 20, n_active)
    
    def test_sparse_series_extraction(self):
        """Test series extraction from sparse hypervec"""
        # Create simple mask
        mask_space = NeuroSpace(dim=(5, 5, 5))
        mask_data = np.zeros((5, 5, 5), dtype=bool)
        mask_data[2, 2, 2] = True
        mask_data[3, 3, 3] = True
        mask = LogicalNeuroVol(mask_data, mask_space)
        
        space_5d = NeuroSpace(dim=(5, 5, 5, 10, 3))
        data = np.ones((3, 10, 2))  # features x time x 2 active voxels
        data[0, :, 0] = 1  # First voxel, feature 0
        data[1, :, 0] = 2  # First voxel, feature 1
        data[2, :, 0] = 3  # First voxel, feature 2
        
        hvec = SparseNeuroHyperVec(data=data, mask=mask, space=space_5d)
        
        # Extract from active voxel
        ts = hvec.series([2, 2, 2])
        assert ts.shape == (10, 3)
        assert np.all(ts[:, 0] == 1)
        assert np.all(ts[:, 1] == 2)
        assert np.all(ts[:, 2] == 3)
        
        # Extract from inactive voxel
        ts_zero = hvec.series([0, 0, 0])
        assert ts_zero.shape == (10, 3)
        assert np.all(ts_zero == 0)
    
    def test_sparse_mean_features(self):
        """Test averaging features in sparse hypervec"""
        mask_space = NeuroSpace(dim=(5, 5, 5))
        mask_data = np.ones((5, 5, 5), dtype=bool)
        mask = LogicalNeuroVol(mask_data, mask_space)
        
        space_5d = NeuroSpace(dim=(5, 5, 5, 10, 4))
        n_active = mask_data.sum()
        
        # Create data where features have different values
        data = np.zeros((4, 10, n_active))
        for i in range(4):
            data[i, :, :] = i
        
        hvec = SparseNeuroHyperVec(data=data, mask=mask, space=space_5d)
        mean_vec = hvec.mean_features()
        
        # Should be a sparse 4D vec
        assert hasattr(mean_vec, 'mask')
        assert mean_vec.shape == (5, 5, 5, 10)
        
        # Mean should be 1.5 for all active voxels
        ts = mean_vec.series(np.array([[2, 2, 2]]))
        expected = np.ones((10, 1)) * 1.5
        np.testing.assert_allclose(ts, expected)


class TestNeuroHyperVecEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_single_feature_dimension(self):
        """Test hypervec with single feature"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 1))
        data = np.random.randn(5, 5, 5, 10, 1)
        hvec = DenseNeuroHyperVec(data, space)
        
        assert hvec.n_features == 1
        
        # Mean should return same data but 4D
        mean_vec = hvec.mean_features()
        expected = data.squeeze(-1)
        np.testing.assert_array_equal(mean_vec.data, expected)
    
    def test_single_timepoint_hypervec(self):
        """Test hypervec with single timepoint"""
        space = NeuroSpace(dim=(5, 5, 5, 1, 3))
        data = np.random.randn(5, 5, 5, 1, 3)
        hvec = DenseNeuroHyperVec(data, space)
        
        assert hvec.n_timepoints == 1
        
        ts = hvec.series([2, 2, 2])
        assert ts.shape == (1, 3)
    
    def test_large_feature_dimension(self):
        """Test hypervec with many features"""
        space = NeuroSpace(dim=(5, 5, 5, 10, 100))
        data = np.random.randn(5, 5, 5, 10, 100).astype(np.float32)
        hvec = DenseNeuroHyperVec(data, space)
        
        assert hvec.n_features == 100
        
        # Select subset
        subset = hvec.select_features(list(range(0, 100, 10)))
        assert subset.n_features == 10
    
    def test_feature_operations_with_nan(self):
        """Test feature operations with NaN values"""
        space = NeuroSpace(dim=(3, 3, 3, 5, 4))
        data = np.ones((3, 3, 3, 5, 4))
        data[1, 1, 1, 2, 1] = np.nan
        
        hvec = DenseNeuroHyperVec(data, space)
        
        # Mean should handle NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_vec = hvec.mean_features()
            
        # Check that NaN is propagated
        ts = mean_vec.series_3d(1, 1, 1)
        assert np.isnan(ts[2])