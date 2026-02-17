"""Additional tests for statistical operations to improve coverage."""

import pytest
import numpy as np
from neuroimpy import (
    NeuroSpace, DenseNeuroVol, DenseNeuroVec, SparseNeuroVol,
    LogicalNeuroVol, ClusteredNeuroVol, ROIVol,
    split_blocks, split_clusters, split_fill, split_reduce,
    split_scale, partition, map_values, centroids
)
from neuroimpy import SparseNeuroVec


class TestSplitBlocksEdgeCases:
    """Test edge cases for split_blocks function."""
    
    def test_split_blocks_vec_time_last(self):
        """Test split_blocks with NeuroVec where time is last dimension."""
        # Create space where time is last
        space = NeuroSpace(dim=[10, 10, 10, 5])
        vec_data = np.random.randn(10, 10, 10, 5)
        vec = DenseNeuroVec(vec_data, space)
        
        # Create indices and block IDs
        indices = np.array([0, 100, 200, 300])
        block_ids = np.array([1, 1, 2, 2])
        
        blocks = split_blocks(vec, indices, block_ids)
        
        assert len(blocks) == 2
        for block in blocks:
            assert isinstance(block, SparseNeuroVec)


class TestSplitClustersEdgeCases:
    """Test edge cases for split_clusters function."""
    
    def test_split_clusters_neurovec(self):
        """Test split_clusters with NeuroVec input."""
        # Create 4D NeuroVec data with 10 time points
        vec_space = NeuroSpace(dim=[10, 5, 5, 5])  # time × x × y × z
        vec_data = np.random.randn(10, 5, 5, 5)
        vec = DenseNeuroVec(vec_data, vec_space)
        
        # Create cluster labels matching spatial dimensions
        cluster_space = NeuroSpace(dim=[5, 5, 5])
        cluster_data = np.zeros((5, 5, 5), dtype=int)
        cluster_data[0:2, 0:2, 0:2] = 1
        cluster_data[3:5, 3:5, 3:5] = 2
        cluster_vol = DenseNeuroVol(cluster_data, cluster_space)
        
        # This should extract mean time series for each cluster
        rois = split_clusters(vec, cluster_vol)
        
        assert len(rois) == 2
        for roi in rois:
            assert isinstance(roi, ROIVol)


class TestSplitFillEdgeCases:
    """Test edge cases for split_fill function."""
    
    def test_split_fill_sparse(self):
        """Test split_fill with SparseNeuroVec."""
        # Create sparse vector
        space = NeuroSpace(dim=[5, 5, 5, 10])
        mask_data = np.zeros((5, 5, 5), dtype=bool)
        mask_data[0:2, 0:2, 0:2] = True  # 8 voxels
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=[5, 5, 5]))
        
        # Create sparse data (10 timepoints × 8 voxels)
        sparse_data = np.random.randn(10, 8)
        sparse_vec = SparseNeuroVec(sparse_data, space, mask)
        
        factor = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
        
        result = split_fill(sparse_vec, factor)
        
        assert len(result) == 3
        assert all(isinstance(v, SparseNeuroVec) for v in result.values())
        
        # Check dimensions
        assert result[1].shape[3] == 3  # 3 timepoints for level 1
        assert result[2].shape[3] == 3  # 3 timepoints for level 2
        assert result[3].shape[3] == 4  # 4 timepoints for level 3


class TestSplitReduceEdgeCases:
    """Test edge cases for split_reduce function."""
    
    def test_split_reduce_sparse(self):
        """Test split_reduce with SparseNeuroVec."""
        # Create sparse vector
        space = NeuroSpace(dim=[3, 3, 3, 6])
        mask_data = np.zeros((3, 3, 3), dtype=bool)
        mask_data[1, 1, 1] = True  # 1 voxel
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=[3, 3, 3]))
        
        # Create sparse data with known values
        sparse_data = np.array([[1], [2], [3], [4], [5], [6]])  # 6 timepoints × 1 voxel
        sparse_vec = SparseNeuroVec(sparse_data, space, mask)
        
        factor = np.array([1, 1, 1, 2, 2, 2])
        
        # Test with sum function (sum of groups: 6 and 15, mean: 10.5)
        result = split_reduce(sparse_vec, factor, np.sum)
        
        assert isinstance(result, SparseNeuroVol)
        # Mean of sums: (6 + 15) / 2 = 10.5
        assert np.isclose(result.data[0], 10.5)
    
    def test_split_reduce_empty_result(self):
        """Test split_reduce when result would be empty."""
        # Create sparse vector with no data
        space = NeuroSpace(dim=[3, 3, 3, 6])
        mask_data = np.zeros((3, 3, 3), dtype=bool)
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=[3, 3, 3]))
        
        # Create empty sparse data
        sparse_data = np.zeros((6, 0))  # 6 timepoints × 0 voxels
        sparse_vec = SparseNeuroVec(sparse_data, space, mask)
        
        factor = np.array([1, 1, 1, 2, 2, 2])
        
        result = split_reduce(sparse_vec, factor, np.mean)
        
        assert isinstance(result, DenseNeuroVol)
        assert np.all(result.data == 0)


class TestSplitScaleEdgeCases:
    """Test edge cases for split_scale function."""
    
    def test_split_scale_sparse_works(self):
        """Test that split_scale works for sparse input."""
        # Create sparse vector
        space = NeuroSpace(dim=[3, 3, 3, 6])
        mask_data = np.ones((3, 3, 3), dtype=bool)
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=[3, 3, 3]))
        sparse_data = np.random.randn(27, 6)
        sparse_vec = SparseNeuroVec(sparse_data, space, mask)

        factor = np.array([1, 1, 1, 2, 2, 2])

        # split_scale now supports sparse input
        result = split_scale(sparse_vec, factor)
        assert result is not None


class TestPartitionEdgeCases:
    """Test edge cases for partition function."""
    
    def test_partition_with_mask(self):
        """Test partition with explicit mask."""
        space = NeuroSpace(dim=[10, 10, 10])
        vol_data = np.random.randn(10, 10, 10)
        vol = DenseNeuroVol(vol_data, space)
        
        # Create mask that only includes part of volume
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[2:8, 2:8, 2:8] = True
        mask = LogicalNeuroVol(mask_data, space)
        
        result = partition(vol, k=3, mask=mask)
        
        assert isinstance(result, ClusteredNeuroVol)
        assert result.num_clusters() == 3
        
        # Check that only masked voxels are clustered
        assert np.all((result.as_dense().data > 0) == mask_data)


class TestMapValuesEdgeCases:
    """Test edge cases for map_values function."""
    
    def test_map_values_generic_vol(self):
        """Test map_values with generic NeuroVol subclass."""
        # Create a simple subclass for testing
        class CustomNeuroVol(DenseNeuroVol):
            pass
        
        space = NeuroSpace(dim=[5, 5, 5])
        vol_data = np.array([1, 2, 3, 1, 2] * 25).reshape(5, 5, 5)
        custom_vol = CustomNeuroVol(vol_data.astype(float), space)
        
        lookup = {1.0: 10.0, 2.0: 20.0, 3.0: 30.0}
        
        result = map_values(custom_vol, lookup)
        
        # Should return same type
        assert isinstance(result, CustomNeuroVol)
        assert np.all(result.data[custom_vol.data == 1.0] == 10.0)


class TestCentroidsEdgeCases:
    """Test edge cases for centroids function."""
    
    def test_centroids_median_implementation(self):
        """Test that median centroid calculation works correctly."""
        space = NeuroSpace(dim=[10, 10, 10])
        
        # Create a simple clustered volume with known clusters
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        # Cluster 1: cube from (0,0,0) to (2,2,2)
        mask_data[0:3, 0:3, 0:3] = True
        # Cluster 2: cube from (5,5,5) to (7,7,7)
        mask_data[5:8, 5:8, 5:8] = True
        
        mask = LogicalNeuroVol(mask_data, space)
        
        # Create cluster labels
        clusters = np.ones(np.sum(mask_data), dtype=int)
        clusters[27:] = 2  # Second cluster
        
        clustered = ClusteredNeuroVol(mask, clusters)
        
        # Get median centroids
        centers = centroids(clustered, method="median")
        
        assert len(centers) == 2
        assert 1 in centers
        assert 2 in centers
        
        # Check approximate median positions
        # For a 3x3x3 cube, median should be at center (1,1,1) and (6,6,6)
        assert np.allclose(centers[1], [1, 1, 1], atol=0.5)
        assert np.allclose(centers[2], [6, 6, 6], atol=0.5)