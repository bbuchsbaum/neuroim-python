"""
Tests for NeuroHyperVec - 5D+ neuroimaging data structures.

NeuroHyperVec represents data with:
- 3 spatial dimensions (x, y, z)
- 1 time dimension
- 1+ feature dimensions (e.g., frequency bands, model parameters)
"""

import pytest
import numpy as np
import neuroim as pn
from neuroim.neuro_space import NeuroSpace


# Note: NeuroHyperVec is not yet implemented, so these tests will guide implementation
class TestNeuroHyperVec:
    """Test 5D neuroimaging data structure."""
    
    def test_basic_construction(self):
        """Test creating a basic NeuroHyperVec."""
        # 5D space: 10x10x10 spatial, 20 timepoints, 5 features
        space = NeuroSpace(dim=(10, 10, 10, 20, 5))
        data = np.random.randn(10, 10, 10, 20, 5)
        
        hvec = pn.DenseNeuroHyperVec(data, space)
        
        assert hvec.shape == (10, 10, 10, 20, 5)
        assert hvec.n_features == 5
        assert hvec.n_timepoints == 20
        assert hvec.spatial_shape == (10, 10, 10)
    
    def test_sparse_construction(self):
        """Test creating a sparse NeuroHyperVec."""
        # Common use case: multiple frequency bands at each voxel/timepoint
        space = NeuroSpace(dim=(64, 64, 32, 100, 8))  # 8 frequency bands
        mask = pn.LogicalNeuroVol(
            np.random.rand(64, 64, 32) > 0.7,
            NeuroSpace(dim=(64, 64, 32))
        )
        
        n_voxels = mask.data.sum()
        # Data: [features x time x voxels]
        sparse_data = np.random.randn(8, 100, n_voxels)
        
        hvec = pn.SparseNeuroHyperVec(
            data=sparse_data,
            mask=mask,
            space=space
        )
        
        assert hvec.shape == (64, 64, 32, 100, 8)
        assert hvec.data.shape == (8, 100, n_voxels)
    
    def test_indexing_5d(self):
        """Test various indexing operations on 5D data."""
        space = NeuroSpace(dim=(10, 10, 10, 20, 5))
        data = np.random.randn(10, 10, 10, 20, 5)
        hvec = pn.DenseNeuroHyperVec(data, space)
        
        # Extract single voxel across time and features
        voxel_data = hvec[5, 5, 5, :, :]
        assert voxel_data.shape == (20, 5)
        
        # Extract single feature across space and time
        feature_vol = hvec[:, :, :, :, 0]
        assert isinstance(feature_vol, pn.NeuroVec)
        assert feature_vol.shape == (10, 10, 10, 20)
        
        # Extract single timepoint
        time_vol = hvec[:, :, :, 10, :]
        assert time_vol.shape == (10, 10, 10, 5)
        
        # Extract spatial ROI
        roi_data = hvec[4:7, 4:7, 4:7, :, :]
        assert roi_data.shape == (3, 3, 3, 20, 5)
    
    def test_series_extraction(self):
        """Test time series extraction with features."""
        space = NeuroSpace(dim=(10, 10, 10, 50, 3))
        data = np.random.randn(10, 10, 10, 50, 3)
        hvec = pn.DenseNeuroHyperVec(data, space)
        
        # Extract series for single voxel
        series = hvec.series([5, 5, 5])
        assert series.shape == (50, 3)  # time x features
        
        # Extract series for multiple voxels
        coords = np.array([[3, 3, 3], [7, 7, 7]])
        multi_series = hvec.series(coords)
        assert multi_series.shape == (50, 3, 2)  # time x features x voxels
        
        # Extract series for specific feature
        feature_series = hvec.series([5, 5, 5], feature=1)
        assert feature_series.shape == (50,)
    
    def test_feature_operations(self):
        """Test operations across feature dimension."""
        space = NeuroSpace(dim=(10, 10, 10, 20, 4))
        data = np.random.randn(10, 10, 10, 20, 4)
        hvec = pn.DenseNeuroHyperVec(data, space)
        
        # Average across features
        avg_vec = hvec.mean_features()
        assert isinstance(avg_vec, pn.NeuroVec)
        assert avg_vec.shape == (10, 10, 10, 20)
        
        # Feature-wise standard deviation
        std_vol = hvec.std_features(time_idx=0)
        assert isinstance(std_vol, pn.NeuroVol)
        assert std_vol.shape == (10, 10, 10)
        
        # Extract specific features
        features_subset = hvec.select_features([0, 2])
        assert features_subset.shape == (10, 10, 10, 20, 2)
    
    def test_concatenate_features(self):
        """Test concatenating along feature dimension."""
        space1 = NeuroSpace(dim=(10, 10, 10, 20, 3))
        space2 = NeuroSpace(dim=(10, 10, 10, 20, 2))
        
        data1 = np.random.randn(10, 10, 10, 20, 3)
        data2 = np.random.randn(10, 10, 10, 20, 2)
        
        hvec1 = pn.NeuroHyperVec.create(data1, space1)
        hvec2 = pn.NeuroHyperVec.create(data2, space2)
        
        # Concatenate features
        combined = pn.concat_features([hvec1, hvec2])
        assert combined.shape == (10, 10, 10, 20, 5)
        assert combined.n_features == 5
    
    def test_apply_feature_function(self):
        """Test applying functions across features."""
        space = NeuroSpace(dim=(10, 10, 10, 30, 6))
        data = np.random.randn(10, 10, 10, 30, 6)
        hvec = pn.DenseNeuroHyperVec(data, space)
        
        # Apply PCA across features at each voxel/time
        def feature_pca(feature_vector):
            # Simplified: just return first PC score
            return feature_vector.mean()
        
        result = hvec.apply_feature_func(feature_pca)
        assert isinstance(result, pn.NeuroVec)
        assert result.shape == (10, 10, 10, 30)
    
    def test_real_world_use_cases(self):
        """Test realistic NeuroHyperVec applications."""
        # Use case 1: Multi-echo fMRI data
        # 3 echoes at different TEs
        echo_space = NeuroSpace(dim=(64, 64, 40, 200, 3))
        echo_data = np.random.randn(64, 64, 40, 200, 3)
        multi_echo = pn.NeuroHyperVec.create(echo_data, echo_space)
        
        # Combine echoes using weighted average
        te_weights = np.array([0.5, 0.3, 0.2])
        combined = multi_echo.weighted_mean_features(te_weights)
        assert isinstance(combined, pn.NeuroVec)
        
        # Use case 2: Spectral analysis results
        # Power in 5 frequency bands
        freq_space = NeuroSpace(dim=(64, 64, 40, 100, 5))
        freq_data = np.random.randn(64, 64, 40, 100, 5)
        spectral = pn.NeuroHyperVec.create(freq_data, freq_space)
        
        # Extract alpha band (index 2)
        alpha_power = spectral[:, :, :, :, 2]
        assert isinstance(alpha_power, pn.NeuroVec)
    
    def test_io_operations(self, tmp_path):
        """Test saving and loading NeuroHyperVec."""
        space = NeuroSpace(dim=(10, 10, 10, 20, 4))
        data = np.random.randn(10, 10, 10, 20, 4)
        hvec = pn.DenseNeuroHyperVec(data, space)

        path = tmp_path / "hypervec.h5"
        pn.write_neurohypervec(hvec, path)
        loaded = pn.read_neurohypervec(path)

        assert loaded.shape == hvec.shape
        np.testing.assert_array_equal(loaded.data, hvec.data)
    
class TestNeuroHyperVecEdgeCases:
    """Test edge cases and error handling for NeuroHyperVec."""
    
    def test_dimension_validation(self):
        """Test that dimensions are validated correctly."""
        # Must have at least 5 dimensions
        with pytest.raises(ValueError):
            space = NeuroSpace(dim=(10, 10, 10, 20))  # Only 4D
            data = np.random.randn(10, 10, 10, 20)
            pn.NeuroHyperVec.create(data, space)
        
        # Data shape must match space
        with pytest.raises(ValueError):
            space = NeuroSpace(dim=(10, 10, 10, 20, 5))
            data = np.random.randn(10, 10, 10, 20, 4)  # Wrong feature dim
            pn.NeuroHyperVec.create(data, space)
    
    def test_single_feature_handling(self):
        """Test behavior with single feature dimension."""
        space = NeuroSpace(dim=(10, 10, 10, 20, 1))
        data = np.random.randn(10, 10, 10, 20, 1)
        hvec = pn.DenseNeuroHyperVec(data, space)
        
        # Should be able to squeeze to NeuroVec
        vec = hvec.squeeze_features()
        assert isinstance(vec, pn.NeuroVec)
        assert vec.shape == (10, 10, 10, 20)
    
    def test_high_dimensional(self):
        """Test with more than 5 dimensions."""
        # 6D: spatial + time + 2 feature dimensions
        space = NeuroSpace(dim=(10, 10, 10, 20, 5, 3))
        data = np.random.randn(10, 10, 10, 20, 5, 3)
        hvec = pn.DenseNeuroHyperVec(data, space)
        
        assert hvec.shape == (10, 10, 10, 20, 5, 3)
        assert hvec.n_dims == 6
        
        # Should handle indexing appropriately
        subset = hvec[5, 5, 5, :, :, :]
        assert subset.shape == (20, 5, 3)
