"""Tests for connected components functionality."""

import pytest
import numpy as np
import pandas as pd
from neuroimpy import (
    NeuroSpace, DenseNeuroVol, LogicalNeuroVol,
    conn_comp, conn_comp_3D,
    ClusteredNeuroVol, ConnCompResult
)


class TestConnComp:
    """Test conn_comp function."""
    
    def setup_method(self):
        """Set up test data."""
        # Create test volume with known clusters
        self.data = np.zeros((10, 10, 10))
        
        # Cluster 1: small cube at origin
        self.data[0:2, 0:2, 0:2] = 5.0
        
        # Cluster 2: larger cube in middle
        self.data[4:7, 4:7, 4:7] = 10.0
        
        # Cluster 3: single voxel
        self.data[8, 8, 8] = 3.0
        
        # Add some noise below threshold
        self.data[9, 0, 0] = 0.5
        self.data[0, 9, 0] = -1.0
        
        self.space = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2])
        self.vol = DenseNeuroVol(self.data, self.space)
    
    def test_conn_comp_basic(self):
        """Test basic connected component labeling."""
        result = conn_comp(self.vol, threshold=1.0)
        
        assert isinstance(result, ConnCompResult)
        assert isinstance(result.index, ClusteredNeuroVol)
        assert isinstance(result.size, DenseNeuroVol)
        assert isinstance(result.voxels, list)
        
        # Should have 3 clusters
        assert len(result.voxels) == 3
        
        # Check cluster sizes
        assert len(result.voxels[0]) == 8  # 2x2x2 cube
        assert len(result.voxels[1]) == 27  # 3x3x3 cube
        assert len(result.voxels[2]) == 1   # single voxel
    
    def test_conn_comp_with_mask(self):
        """Test conn_comp with LogicalNeuroVol input."""
        mask = LogicalNeuroVol(self.data > 1.0, self.space)
        result = conn_comp(mask)
        
        # Should have same 3 clusters
        assert len(result.voxels) == 3
        
        # Size volume should match mask
        assert np.all((result.size.data > 0) == mask.data)
    
    def test_conn_comp_cluster_table(self):
        """Test cluster table generation."""
        result = conn_comp(self.vol, threshold=1.0, cluster_table=True)
        
        assert result.cluster_table is not None
        assert isinstance(result.cluster_table, pd.DataFrame)
        
        # Check columns
        expected_columns = ['index', 'x', 'y', 'z', 'N', 'Area', 'value']
        assert list(result.cluster_table.columns) == expected_columns
        
        # Check number of rows
        assert len(result.cluster_table) == 3
        
        # Check cluster properties
        table = result.cluster_table.sort_values('N')
        
        # Smallest cluster (single voxel)
        assert table.iloc[0]['N'] == 1
        assert table.iloc[0]['Area'] == 8.0  # 2x2x2 spacing
        assert table.iloc[0]['value'] == 3.0
        
        # Medium cluster (2x2x2)
        assert table.iloc[1]['N'] == 8
        assert table.iloc[1]['Area'] == 64.0
        assert table.iloc[1]['value'] == 5.0
        
        # Largest cluster (3x3x3)
        assert table.iloc[2]['N'] == 27
        assert table.iloc[2]['Area'] == 216.0
        assert table.iloc[2]['value'] == 10.0
    
    def test_conn_comp_local_maxima(self):
        """Test local maxima detection."""
        # Create volume with multiple peaks
        data = np.zeros((20, 20, 20))
        
        # Add gaussian-like peaks
        # Set surrounding values first, then the peaks
        data[4:7, 4:7, 4:7] = 5.0
        data[5, 5, 5] = 10.0  # Overwrite center with peak
        
        data[14:17, 14:17, 14:17] = 4.0
        data[15, 15, 15] = 8.0  # Overwrite center with peak
        
        # Connect them with a bridge
        data[7:14, 10, 10] = 2.0
        
        space = NeuroSpace(dim=[20, 20, 20], spacing=[1, 1, 1])
        vol = DenseNeuroVol(data, space)
        
        result = conn_comp(vol, threshold=1.0, local_maxima=True, 
                          local_maxima_dist=5)
        
        assert result.local_maxima is not None
        assert result.local_maxima.shape[1] == 5  # index, x, y, z, value
        
        # Should find at least 2 maxima (the two peaks)
        # There might be additional maxima at cluster boundaries
        assert len(result.local_maxima) >= 2
        
        # Check values
        values = result.local_maxima[:, 4]
        assert 10.0 in values
        assert 8.0 in values
    
    def test_conn_comp_no_clusters(self):
        """Test with no clusters above threshold."""
        result = conn_comp(self.vol, threshold=100.0)
        
        assert len(result.voxels) == 0
        assert result.cluster_table is None
        assert result.local_maxima is None
        assert np.all(result.size.data == 0)
    
    def test_conn_comp_connectivity_6(self):
        """Test 6-connectivity."""
        # Create diagonal structure
        data = np.zeros((5, 5, 5))
        data[0, 0, 0] = 1
        data[1, 1, 1] = 1  # Diagonal neighbor
        data[1, 0, 0] = 1  # Face neighbor
        
        space = NeuroSpace(dim=[5, 5, 5])
        vol = DenseNeuroVol(data, space)
        
        # With 6-connect, diagonal should be separate cluster
        result = conn_comp(vol, threshold=0.5, connect="6-connect", 
                          cluster_table=False)
        
        # Should have 2 clusters
        assert len(result.voxels) == 2
        assert len(result.voxels[0]) == 2  # Face-connected pair
        assert len(result.voxels[1]) == 1  # Isolated diagonal
    
    def test_conn_comp_connectivity_18(self):
        """Test 18-connectivity."""
        # Create two voxels that are corner neighbors only
        data = np.zeros((5, 5, 5))
        data[1, 1, 1] = 1  # First voxel
        data[2, 2, 2] = 1  # Second voxel - corner neighbor to first
        
        space = NeuroSpace(dim=[5, 5, 5])
        vol = DenseNeuroVol(data, space)
        
        # With 18-connect, corner neighbors should NOT connect
        result = conn_comp(vol, threshold=0.5, connect="18-connect",
                          cluster_table=False)
        
        # Should have 2 separate clusters
        assert len(result.voxels) == 2
        assert all(len(v) == 1 for v in result.voxels)  # Each cluster has 1 voxel
    
    def test_conn_comp_connectivity_26(self):
        """Test 26-connectivity (default)."""
        # Create structure with all types of neighbors
        data = np.zeros((5, 5, 5))
        data[1, 1, 1] = 1  # Center
        data[0, 1, 1] = 1  # Face neighbor
        data[0, 0, 1] = 1  # Edge neighbor
        data[0, 0, 0] = 1  # Corner neighbor
        
        space = NeuroSpace(dim=[5, 5, 5])
        vol = DenseNeuroVol(data, space)
        
        # With 26-connect, all should be connected
        result = conn_comp(vol, threshold=0.5, connect="26-connect",
                          cluster_table=False)
        
        # Should have 1 cluster with all 4 voxels
        assert len(result.voxels) == 1
        assert len(result.voxels[0]) == 4


class TestConnComp3D:
    """Test conn_comp_3D function."""
    
    def test_conn_comp_3d_array(self):
        """Test with numpy array input."""
        # Create binary mask with known components
        mask = np.zeros((10, 10, 10), dtype=bool)
        
        # Component 1
        mask[0:3, 0:3, 0:3] = True
        
        # Component 2
        mask[5:8, 5:8, 5:8] = True
        
        # Component 3 (single voxel)
        mask[9, 9, 9] = True
        
        result = conn_comp_3D(mask)
        
        assert isinstance(result, dict)
        assert 'index' in result
        assert 'size' in result
        
        # Check dimensions
        assert result['index'].shape == mask.shape
        assert result['size'].shape == mask.shape
        
        # Check number of components
        assert np.max(result['index']) == 3
        
        # Check component sizes
        sizes = np.unique(result['size'])[1:]  # Skip 0
        assert set(sizes) == {1, 27}  # Two 3x3x3 cubes and one single
    
    def test_conn_comp_3d_logical_neurovol(self):
        """Test with LogicalNeuroVol input."""
        mask_data = np.zeros((8, 8, 8), dtype=bool)
        mask_data[2:5, 2:5, 2:5] = True
        
        space = NeuroSpace(dim=[8, 8, 8])
        mask = LogicalNeuroVol(mask_data, space)
        
        result = conn_comp_3D(mask)
        
        # Should have one component of size 27
        assert np.max(result['index']) == 1
        assert np.sum(result['size'] > 0) == 27
        assert np.all(result['size'][mask_data] == 27)
    
    def test_conn_comp_3d_connectivity(self):
        """Test different connectivity options."""
        # Create pattern that tests connectivity
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1, 1, 1] = True
        mask[2, 2, 2] = True  # Diagonal only
        mask[1, 2, 1] = True  # Face connected to first
        
        # Test 6-connectivity
        result_6 = conn_comp_3D(mask, connect="6-connect")
        assert np.max(result_6['index']) == 2  # Two components
        
        # Test 26-connectivity  
        result_26 = conn_comp_3D(mask, connect="26-connect")
        assert np.max(result_26['index']) == 1  # One component
    
    def test_conn_comp_3d_empty_mask(self):
        """Test with empty mask."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        result = conn_comp_3D(mask)
        
        assert np.all(result['index'] == 0)
        assert np.all(result['size'] == 0)
    
    def test_conn_comp_3d_full_mask(self):
        """Test with full mask."""
        mask = np.ones((5, 5, 5), dtype=bool)
        result = conn_comp_3D(mask)
        
        assert np.all(result['index'] == 1)
        assert np.all(result['size'] == 125)
    
    def test_conn_comp_3d_errors(self):
        """Test error conditions."""
        # Non-3D array
        with pytest.raises(ValueError, match="Mask must be 3D"):
            conn_comp_3D(np.zeros((5, 5), dtype=bool))
        
        # Non-boolean array
        with pytest.raises(ValueError, match="Mask must be boolean"):
            conn_comp_3D(np.zeros((5, 5, 5), dtype=float))
        
        # Invalid connectivity
        mask = np.zeros((5, 5, 5), dtype=bool)
        with pytest.raises(ValueError, match="Invalid connectivity"):
            conn_comp_3D(mask, connect="4-connect")


class TestIntegration:
    """Integration tests for connected components."""
    
    def test_clustered_neurovol_from_conn_comp(self):
        """Test that ClusteredNeuroVol works with conn_comp output."""
        # Create test data
        data = np.zeros((10, 10, 10))
        data[2:5, 2:5, 2:5] = 5.0
        data[7:9, 7:9, 7:9] = 3.0
        
        space = NeuroSpace(dim=[10, 10, 10])
        vol = DenseNeuroVol(data, space)
        
        # Run connected components
        result = conn_comp(vol, threshold=1.0)
        
        # Check ClusteredNeuroVol
        clustered = result.index
        assert isinstance(clustered, ClusteredNeuroVol)
        assert clustered.num_clusters() == 2
        
        # Check cluster sizes
        sizes = clustered.cluster_sizes()
        assert sizes[1] == 27  # 3x3x3
        assert sizes[2] == 8   # 2x2x2
        
        # Check cluster centers
        centers = clustered.cluster_centers()
        assert len(centers) == 2
        
        # Check get_cluster_mask
        mask1 = clustered.get_cluster_mask(1)
        assert isinstance(mask1, LogicalNeuroVol)
        assert np.sum(mask1.data) == 27
    
    def test_conn_comp_with_real_world_coords(self):
        """Test that cluster table uses world coordinates."""
        # Create volume with non-trivial spacing and origin
        space = NeuroSpace(dim=[10, 10, 10], 
                          spacing=[2.5, 3.0, 4.0],
                          origin=[-10, -20, -30])
        
        data = np.zeros((10, 10, 10))
        data[5, 5, 5] = 1.0
        vol = DenseNeuroVol(data, space)
        
        result = conn_comp(vol, threshold=0.5, cluster_table=True)
        
        # Check that coordinates are in world space
        table = result.cluster_table
        assert len(table) == 1
        
        # Expected world coordinates for voxel [5,5,5]
        # world = origin + spacing * grid
        expected_x = -10 + 2.5 * 5
        expected_y = -20 + 3.0 * 5  
        expected_z = -30 + 4.0 * 5
        
        assert np.isclose(table.iloc[0]['x'], expected_x)
        assert np.isclose(table.iloc[0]['y'], expected_y)
        assert np.isclose(table.iloc[0]['z'], expected_z)