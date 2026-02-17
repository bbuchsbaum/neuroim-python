"""Tests for statistical operations."""

import pytest
import numpy as np
from neuroimpy import (
    NeuroSpace, DenseNeuroVol, DenseNeuroVec, SparseNeuroVol,
    LogicalNeuroVol, ClusteredNeuroVol, ROIVol,
    split_blocks, split_clusters, split_fill, split_reduce,
    split_scale, partition, map_values, centroids
)


class TestSplitBlocks:
    """Test split_blocks function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[10, 10, 10])
        
        # Create test volume with known patterns
        self.vol_data = np.zeros((10, 10, 10))
        # Fill specific locations that we'll index
        flat_data = self.vol_data.ravel(order='F')
        flat_data[0:3] = 1.0  # First 3 indices
        flat_data[500:503] = 2.0  # Indices 500-502
        self.vol_data = flat_data.reshape((10, 10, 10), order='F')
        self.vol = DenseNeuroVol(self.vol_data, self.space)
        
        # Create indices and block IDs
        self.indices = np.array([0, 1, 2, 500, 501, 502])  # Mix of blocks
        self.block_ids = np.array([1, 1, 1, 2, 2, 2])
    
    def test_split_blocks_volume(self):
        """Test splitting a volume into blocks."""
        blocks = split_blocks(self.vol, self.indices, self.block_ids)
        
        assert len(blocks) == 2
        
        # Check each block
        for i, block in enumerate(blocks):
            assert isinstance(block, SparseNeuroVol)
            assert block.space == self.space
            
            if i == 0:  # Block 1
                assert len(block.data) == 3
                assert np.all(block.data == 1.0)
            else:  # Block 2
                assert len(block.data) == 3
                assert np.all(block.data == 2.0)
    
    def test_split_blocks_vector(self):
        """Test splitting a vector into blocks."""
        # Create test vector with 4D space (x, y, z, time)
        vec_space = NeuroSpace(dim=[10, 10, 10, 5])
        vec_data = np.random.randn(10, 10, 10, 5)
        vec = DenseNeuroVec(vec_data, vec_space)
        
        blocks = split_blocks(vec, self.indices, self.block_ids)
        
        assert len(blocks) == 2
        for block in blocks:
            assert block.space == vec_space
    
    def test_split_blocks_error(self):
        """Test error conditions."""
        with pytest.raises(ValueError, match="same length"):
            split_blocks(self.vol, self.indices, self.block_ids[:-1])


class TestSplitClusters:
    """Test split_clusters function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[10, 10, 10])
        
        # Create data volume
        self.data = np.random.randn(10, 10, 10)
        self.vol = DenseNeuroVol(self.data, self.space)
        
        # Create cluster labels
        self.cluster_data = np.zeros((10, 10, 10), dtype=int)
        self.cluster_data[0:3, 0:3, 0:3] = 1
        self.cluster_data[5:8, 5:8, 5:8] = 2
        self.cluster_vol = DenseNeuroVol(self.cluster_data, self.space)
    
    def test_split_clusters_neurovol(self):
        """Test splitting volume by clusters."""
        rois = split_clusters(self.vol, self.cluster_vol)
        
        assert len(rois) == 2  # Two clusters
        
        for roi in rois:
            assert isinstance(roi, ROIVol)
            assert roi.space == self.space
        
        # Check cluster sizes
        assert len(rois[0].data) == 27  # 3x3x3
        assert len(rois[1].data) == 27  # 3x3x3
    
    def test_split_clusters_clustered_neurovol(self):
        """Test with ClusteredNeuroVol input."""
        # Create ClusteredNeuroVol
        mask = LogicalNeuroVol(self.cluster_data > 0, self.space)
        cluster_labels = self.cluster_data[self.cluster_data > 0]
        clustered = ClusteredNeuroVol(mask, cluster_labels)
        
        rois = split_clusters(self.vol, clustered)
        
        assert len(rois) == 2
        assert sum(len(roi.data) for roi in rois) == 54  # Total non-zero voxels
    
    def test_split_clusters_error(self):
        """Test error conditions."""
        wrong_space = NeuroSpace(dim=[5, 5, 5])
        wrong_vol = DenseNeuroVol(np.zeros((5, 5, 5)), wrong_space)
        
        with pytest.raises(ValueError, match="same space"):
            split_clusters(wrong_vol, self.cluster_vol)


class TestSplitFill:
    """Test split_fill function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[10, 10, 10, 10])  # 4D space for NeuroVec
        self.vec_data = np.random.randn(10, 10, 10, 10)
        self.vec = DenseNeuroVec(self.vec_data, self.space)
        
        # Create factor with 3 levels
        self.factor = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    
    def test_split_fill_basic(self):
        """Test basic split_fill operation."""
        result = split_fill(self.vec, self.factor)
        
        assert len(result) == 3  # Three levels
        assert set(result.keys()) == {1, 2, 3}
        
        # Check each split
        assert result[1].shape[0] == 3  # 3 volumes
        assert result[2].shape[0] == 3  # 3 volumes
        assert result[3].shape[0] == 4  # 4 volumes
        
        # All should have same spatial dimensions
        for vec in result.values():
            assert vec.shape[1:] == self.vec.shape[1:]
    
    def test_split_fill_error(self):
        """Test error conditions."""
        wrong_factor = np.array([1, 2])  # Wrong length
        
        with pytest.raises(ValueError, match="number of volumes"):
            split_fill(self.vec, wrong_factor)


class TestSplitReduce:
    """Test split_reduce function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[5, 5, 5, 6])  # 4D space for NeuroVec (x, y, z, time)
        
        # Create vector with known pattern
        vec_data = np.zeros((5, 5, 5, 6))
        vec_data[0, 0, 0, 0:3] = 1.0  # First group (first 3 time points)
        vec_data[0, 0, 0, 3:6] = 2.0  # Second group (last 3 time points)
        self.vec = DenseNeuroVec(vec_data, self.space)
        
        self.factor = np.array([1, 1, 1, 2, 2, 2])
    
    def test_split_reduce_mean(self):
        """Test split_reduce with mean function."""
        result = split_reduce(self.vec, self.factor, np.mean)
        
        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (5, 5, 5)
        
        # Check the reduced value at (0,0,0)
        # Should be mean of means: (1.0 + 2.0) / 2 = 1.5
        assert result.data[0, 0, 0] == 1.5
    
    def test_split_reduce_max(self):
        """Test split_reduce with max function."""
        result = split_reduce(self.vec, self.factor, np.max)
        
        # Should be mean of maxes: (1.0 + 2.0) / 2 = 1.5
        assert result.data[0, 0, 0] == 1.5
    
    def test_split_reduce_error(self):
        """Test error conditions."""
        wrong_factor = np.array([1, 2])  # Wrong length
        
        with pytest.raises(ValueError, match="number of volumes"):
            split_reduce(self.vec, wrong_factor, np.mean)


class TestSplitScale:
    """Test split_scale function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[5, 5, 5, 10])  # 4D space for NeuroVec (x, y, z, time)
        
        # Create vector with different means/SDs per group
        vec_data = np.random.randn(5, 5, 5, 10)
        vec_data[:, :, :, 0:5] = vec_data[:, :, :, 0:5] + 10  # Shift first group
        vec_data[:, :, :, 5:10] = vec_data[:, :, :, 5:10] * 3  # Scale second group
        
        self.vec = DenseNeuroVec(vec_data, self.space)
        self.factor = np.array([1]*5 + [2]*5)
    
    def test_split_scale_center_only(self):
        """Test centering only."""
        result = split_scale(self.vec, self.factor, center=True, scale=False)
        
        assert isinstance(result, DenseNeuroVec)
        assert result.shape == self.vec.shape
        
        # Check that each group is centered
        # Get time series for voxel (0,0,0)
        voxel_series = result.data[0, 0, 0, :]
        
        # Check each group separately
        for level in [1, 2]:
            mask = self.factor == level
            group_data = voxel_series[mask]
            # Mean should be close to 0
            assert np.abs(np.mean(group_data)) < 1e-10
    
    def test_split_scale_both(self):
        """Test both centering and scaling."""
        result = split_scale(self.vec, self.factor, center=True, scale=True)
        
        # Check that each group is centered and scaled
        # Get time series for voxel (0,0,0)
        voxel_series = result.data[0, 0, 0, :]
        
        for level in [1, 2]:
            mask = self.factor == level
            group_data = voxel_series[mask]
            
            # Mean should be close to 0
            assert np.abs(np.mean(group_data)) < 1e-10
            
            # SD should be close to 1
            sd = np.std(group_data)
            if sd > 0:  # Only check if not constant
                assert np.abs(sd - 1.0) < 1e-10
    
    def test_split_scale_error(self):
        """Test error conditions."""
        wrong_factor = np.array([1, 2])  # Wrong length
        
        with pytest.raises(ValueError, match="number of volumes"):
            split_scale(self.vec, wrong_factor)


class TestPartition:
    """Test partition function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[10, 10, 10])
        
        # Create volume with distinct regions
        vol_data = np.zeros((10, 10, 10))
        vol_data[0:5, :, :] = 1.0
        vol_data[5:10, :, :] = 5.0
        # Add some noise
        vol_data += np.random.randn(10, 10, 10) * 0.1
        
        self.vol = DenseNeuroVol(vol_data, self.space)
    
    def test_partition_kmeans(self):
        """Test k-means partitioning."""
        result = partition(self.vol, k=2, method="kmeans")
        
        assert isinstance(result, ClusteredNeuroVol)
        assert result.num_clusters() == 2
        
        # Check that clusters roughly correspond to our regions
        cluster_sizes = result.cluster_sizes()
        assert all(s > 400 for s in cluster_sizes.values())  # Each should be ~500
    
    def test_partition_error_too_few_voxels(self):
        """Test error when k > number of voxels."""
        small_vol = DenseNeuroVol(np.array([[[1, 0], [0, 0]]]), 
                                  NeuroSpace(dim=[1, 2, 2]))
        
        with pytest.raises(ValueError, match="less than k"):
            partition(small_vol, k=5)
    
    def test_partition_error_unknown_method(self):
        """Test error with unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            partition(self.vol, k=2, method="unknown")


class TestMapValues:
    """Test map_values function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[5, 5, 5])
        
        # Create volume with specific values
        vol_data = np.array([1, 2, 3, 1, 2, 3] * 20 + [0] * 5).reshape(5, 5, 5)  # Total 125 elements
        self.vol = DenseNeuroVol(vol_data.astype(float), self.space)
        
        self.lookup = {1.0: 10.0, 2.0: 20.0, 3.0: 30.0}
    
    def test_map_values_dense(self):
        """Test mapping values in dense volume."""
        result = map_values(self.vol, self.lookup)
        
        assert isinstance(result, DenseNeuroVol)
        
        # Check mapping
        assert np.all(result.data[self.vol.data == 1.0] == 10.0)
        assert np.all(result.data[self.vol.data == 2.0] == 20.0)
        assert np.all(result.data[self.vol.data == 3.0] == 30.0)
        assert np.all(result.data[self.vol.data == 0.0] == 0.0)  # Unchanged
    
    def test_map_values_sparse(self):
        """Test mapping values in sparse volume."""
        # Create sparse volume
        indices = np.array([0, 1, 2, 3, 4])
        values = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
        sparse_vol = SparseNeuroVol(values, self.space, indices)
        
        result = map_values(sparse_vol, self.lookup)
        
        assert isinstance(result, SparseNeuroVol)
        assert np.array_equal(result.indices, sparse_vol.indices)
        
        # Check mapping
        expected = np.array([10.0, 20.0, 30.0, 10.0, 20.0])
        assert np.array_equal(result.data, expected)


class TestCentroids:
    """Test centroids function."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[10, 10, 10])
        
        # Create clustered volume
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[0:5, 0:5, 0:5] = True
        mask_data[5:10, 5:10, 5:10] = True
        
        mask = LogicalNeuroVol(mask_data, self.space)
        clusters = np.ones(np.sum(mask_data), dtype=int)
        clusters[125:] = 2  # Second half is cluster 2
        
        self.clustered = ClusteredNeuroVol(mask, clusters)
    
    def test_centroids_center_of_mass(self):
        """Test center of mass calculation."""
        centers = centroids(self.clustered, method="center_of_mass")
        
        assert len(centers) == 2
        assert 1 in centers
        assert 2 in centers
        
        # Each center should be 3D coordinate
        for center in centers.values():
            assert len(center) == 3
    
    def test_centroids_median(self):
        """Test median calculation."""
        centers = centroids(self.clustered, method="median")
        
        assert len(centers) == 2
        
        # Median might differ from mean
        com_centers = centroids(self.clustered, method="center_of_mass")
        # They could be different but both valid
        assert all(len(c) == 3 for c in centers.values())
    
    def test_centroids_error(self):
        """Test error with unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            centroids(self.clustered, method="unknown")