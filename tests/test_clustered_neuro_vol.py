"""
Test suite for ClusteredNeuroVol class.

This module tests the ClusteredNeuroVol functionality in neuroimpy,
corresponding to the R neuroim2 test-clusteredneurovol.R tests.
"""

import pytest
import numpy as np
from sklearn.cluster import KMeans
from neuroimpy import (
    NeuroSpace, DenseNeuroVol, LogicalNeuroVol, 
    ClusteredNeuroVol, SparseNeuroVol
)


class TestClusteredNeuroVol:
    """Test cases for ClusteredNeuroVol functionality."""
    
    @pytest.fixture
    def setup_clustered_vol(self):
        """Create a test ClusteredNeuroVol object."""
        # Create space and grid (16x16x16 like in R tests)
        space = NeuroSpace((16, 16, 16), spacing=(1, 1, 1))
        
        # Create coordinate grid
        indices = np.arange(16**3)
        coords = np.array(np.unravel_index(indices, (16, 16, 16))).T
        
        # Perform k-means clustering with 10 centers
        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Create mask (all ones like in R test)
        mask_data = np.ones((16, 16, 16), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)
        
        # Create ClusteredNeuroVol
        clusvol = ClusteredNeuroVol(mask, cluster_labels)
        
        return clusvol, space, cluster_labels
    
    def test_clustered_neurovol_creation(self, setup_clustered_vol):
        """Test ClusteredNeuroVol constructor works correctly."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        assert isinstance(clusvol, ClusteredNeuroVol)
        assert clusvol.space == space
        assert clusvol.num_clusters() == 10
        assert len(clusvol.clusters) == 16**3
    
    def test_clustered_neurovol_with_label_map(self):
        """Test ClusteredNeuroVol with label map."""
        space = NeuroSpace((8, 8, 8), spacing=(1, 1, 1))
        mask_data = np.ones((8, 8, 8), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)
        
        # Create simple clusters
        clusters = np.array([0, 0, 1, 1, 2, 2] * 85 + [0, 0])  # 512 elements
        label_map = {"region_a": 0, "region_b": 1, "region_c": 2}
        
        clusvol = ClusteredNeuroVol(mask, clusters, label_map)
        
        assert clusvol.label_map == label_map
        assert clusvol.num_clusters() == 3
    
    def test_as_dense_conversion(self, setup_clustered_vol):
        """Test conversion to DenseNeuroVol."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Test as_dense method
        dense_vol = clusvol.as_dense()
        
        assert isinstance(dense_vol, DenseNeuroVol)
        assert dense_vol.space == space
        assert dense_vol.shape == (16, 16, 16)
        
        # Verify the cluster values are preserved
        flat_data = dense_vol.data.flatten()
        assert np.array_equal(flat_data, cluster_labels)
    
    def test_as_logical_conversion(self, setup_clustered_vol):
        """Test conversion to LogicalNeuroVol."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Test as_logical method
        logical_vol = clusvol.as_logical()
        
        assert isinstance(logical_vol, LogicalNeuroVol)
        assert logical_vol.space == space
        assert np.all(logical_vol.data)  # All voxels should be True since mask is all ones
    
    def test_to_sparse_conversion(self, setup_clustered_vol):
        """Test conversion to SparseNeuroVol."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Test to_sparse method
        sparse_vol = clusvol.to_sparse()
        
        assert isinstance(sparse_vol, SparseNeuroVol)
        assert sparse_vol.space == space
        assert len(sparse_vol.data) == np.sum(clusvol.mask.data)
    
    def test_cluster_centers(self, setup_clustered_vol):
        """Test centroids/cluster_centers functionality."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Get cluster centers (center of mass)
        centers = clusvol.cluster_centers()
        
        assert isinstance(centers, dict)
        assert len(centers) == clusvol.num_clusters()
        
        # Each center should be a 3D coordinate
        for cluster_id, center in centers.items():
            assert isinstance(center, np.ndarray)
            assert center.shape == (3,)
            # Centers should be within the volume bounds
            assert np.all(center >= 0)
            assert np.all(center < 16)
    
    def test_split_clusters(self, setup_clustered_vol):
        """Test splitting data by clusters."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Create a test data volume with random values
        data = np.random.rand(16, 16, 16)
        vol = DenseNeuroVol(data, space)
        
        # Split the volume by clusters
        split_data = clusvol.split_clusters(vol)
        
        assert isinstance(split_data, dict)
        assert len(split_data) == clusvol.num_clusters()
        
        # Verify each cluster's data
        for cluster_id, cluster_data in split_data.items():
            # Get expected data for this cluster
            cluster_indices = clusvol.cluster_map[cluster_id]
            expected_data = data.flatten()[cluster_indices]
            assert np.array_equal(cluster_data, expected_data)
    
    def test_num_clusters(self, setup_clustered_vol):
        """Test num_clusters method."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        num_clus = clusvol.num_clusters()
        
        assert num_clus == 10
        assert num_clus == len(np.unique(cluster_labels))
    
    def test_cluster_sizes(self, setup_clustered_vol):
        """Test cluster_sizes method."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        sizes = clusvol.cluster_sizes()
        
        assert isinstance(sizes, dict)
        assert len(sizes) == clusvol.num_clusters()
        
        # Total size should equal total voxels
        total_size = sum(sizes.values())
        assert total_size == 16**3
        
        # Verify sizes match actual cluster counts
        for cluster_id, size in sizes.items():
            expected_size = np.sum(cluster_labels == cluster_id)
            assert size == expected_size
    
    def test_get_cluster_mask(self, setup_clustered_vol):
        """Test getting mask for a specific cluster."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Get mask for cluster 0
        cluster_mask = clusvol.get_cluster_mask(0)
        
        assert isinstance(cluster_mask, LogicalNeuroVol)
        assert cluster_mask.space == space
        
        # Verify the mask contains only the cluster 0 voxels
        expected_mask = (cluster_labels == 0).reshape(16, 16, 16)
        assert np.array_equal(cluster_mask.data, expected_mask)
    
    def test_get_cluster_data(self, setup_clustered_vol):
        """Test extracting data for a specific cluster."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Create test data
        data = np.random.rand(16, 16, 16)
        vol = DenseNeuroVol(data, space)
        
        # Get data for cluster 0
        cluster_data = clusvol.get_cluster_data(vol, 0)
        
        assert isinstance(cluster_data, np.ndarray)
        
        # Verify we got the right data
        cluster_indices = np.where(cluster_labels == 0)[0]
        expected_data = data.flatten()[cluster_indices]
        assert np.array_equal(cluster_data, expected_data)
    
    def test_getitem_with_cluster_id(self, setup_clustered_vol):
        """Test indexing with cluster ID."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Get indices for cluster 0
        cluster_indices = clusvol[0]
        
        assert isinstance(cluster_indices, np.ndarray)
        assert len(cluster_indices) == np.sum(cluster_labels == 0)
    
    def test_getitem_with_label(self):
        """Test indexing with cluster label string."""
        space = NeuroSpace((8, 8, 8), spacing=(1, 1, 1))
        mask_data = np.ones((8, 8, 8), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)
        
        clusters = np.array([0, 0, 1, 1, 2, 2] * 85 + [0, 0])
        label_map = {"region_a": 0, "region_b": 1, "region_c": 2}
        
        clusvol = ClusteredNeuroVol(mask, clusters, label_map)
        
        # Get indices for "region_a"
        region_indices = clusvol["region_a"]
        
        assert isinstance(region_indices, np.ndarray)
        assert len(region_indices) == np.sum(clusters == 0)
    
    def test_invalid_cluster_label(self):
        """Test error handling for invalid cluster labels."""
        space = NeuroSpace((8, 8, 8), spacing=(1, 1, 1))
        mask_data = np.ones((8, 8, 8), dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)
        
        clusters = np.zeros(512, dtype=int)
        label_map = {"region_a": 0}
        
        clusvol = ClusteredNeuroVol(mask, clusters, label_map)
        
        # Should raise KeyError for non-existent label
        with pytest.raises(KeyError):
            _ = clusvol["non_existent_region"]
    
    def test_show_method(self, setup_clustered_vol, capsys):
        """Test string representation of ClusteredNeuroVol."""
        clusvol, space, cluster_labels = setup_clustered_vol
        
        # Call str() which should invoke __str__ or __repr__
        str_output = str(clusvol)
        
        # Should contain class name and basic info
        assert "ClusteredNeuroVol" in str_output
        
    def test_setitem_not_implemented(self, setup_clustered_vol):
        """Test that item assignment raises TypeError (read-only)."""
        clusvol, space, cluster_labels = setup_clustered_vol

        with pytest.raises(TypeError, match="read-only"):
            clusvol[0] = 1
    
    def test_partial_mask(self):
        """Test ClusteredNeuroVol with a partial mask."""
        space = NeuroSpace((10, 10, 10), spacing=(1, 1, 1))
        
        # Create a mask with only center voxels active
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[3:7, 3:7, 3:7] = True
        mask = LogicalNeuroVol(mask_data, space)
        
        # Create clusters only for active voxels
        n_active = np.sum(mask_data)
        clusters = np.repeat([0, 1, 2, 3], n_active // 4 + 1)[:n_active]
        
        clusvol = ClusteredNeuroVol(mask, clusters)
        
        assert clusvol.num_clusters() == 4
        assert sum(clusvol.cluster_sizes().values()) == n_active