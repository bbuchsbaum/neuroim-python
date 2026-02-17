"""Tests for searchlight functionality."""

import pytest
import numpy as np
from neuroimpy import (
    NeuroSpace, LogicalNeuroVol, DenseNeuroVol,
    searchlight, searchlight_coords, random_searchlight,
    bootstrap_searchlight, clustered_searchlight,
    ROIVolWindow, ROIVol
)
from neuroimpy.utils import LazyList


class TestSearchlight:
    """Test exhaustive searchlight function."""
    
    def setup_method(self):
        """Set up test data."""
        # Create a small test mask
        self.mask_data = np.zeros((10, 10, 10), dtype=bool)
        self.mask_data[3:7, 3:7, 3:7] = True  # 4x4x4 cube = 64 voxels
        self.space = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2])
        self.mask = LogicalNeuroVol(self.mask_data, self.space)
    
    def test_searchlight_lazy(self):
        """Test lazy searchlight generation."""
        sl = searchlight(self.mask, radius=4, eager=False, nonzero=True)
        
        # Should return LazyList
        assert isinstance(sl, LazyList)
        assert len(sl) == 64  # Number of nonzero voxels
        
        # Access first searchlight
        first_sl = sl[0]
        assert isinstance(first_sl, ROIVolWindow)
        assert first_sl.space == self.space
        assert first_sl.parent_index is not None
        center_grid = first_sl.parent_grid
        center_row = np.where(np.all(first_sl.coords == center_grid, axis=1))[0]
        expected_center = center_row[0] if len(center_row) else 0
        assert first_sl.center_index == expected_center
        
        # Check that searchlight contains voxels
        assert len(first_sl.coords) > 0
        assert len(first_sl.data) == len(first_sl.coords)
    
    def test_searchlight_eager(self):
        """Test eager searchlight generation."""
        sl = searchlight(self.mask, radius=4, eager=True, nonzero=True)
        
        # Should return list
        assert isinstance(sl, list)
        assert len(sl) == 64  # Number of nonzero voxels
        
        # Check all searchlights
        for s in sl:
            assert isinstance(s, ROIVolWindow)
            assert s.space == self.space
            assert len(s.coords) > 0
    
    def test_searchlight_all_voxels(self):
        """nonzero flag controls ROI membership, but centers stay nonzero."""
        sl = searchlight(self.mask, radius=4, eager=False, nonzero=False)
        
        # Nonzero centers are always used for iteration.
        assert isinstance(sl, LazyList)
        assert len(sl) == 64  # Number of nonzero voxels
        # Each center must be nonzero by construction.
        assert all(
            self.mask.data[tuple(sl[i].parent_grid)]
            for i in range(len(sl))
        )
    
    def test_searchlight_large_radius(self):
        """Test searchlight with large radius."""
        sl = searchlight(self.mask, radius=20, eager=True, nonzero=True)
        
        # Large radius should include many voxels
        for s in sl:
            assert len(s.coords) > 10  # Should have many neighbors

    def test_searchlight_respects_spacing(self):
        """Searchlight centers should use physical spacing for radius."""
        spacing = (2.0, 2.0, 3.0)
        space = NeuroSpace(dim=[11, 11, 11], spacing=spacing)
        mask_data = np.ones(space.dim, dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)

        sl = searchlight(mask, radius=4.0, eager=True, nonzero=True)

        centroid = np.array([5, 5, 5], dtype=int)
        center_idx = int(space.grid_to_index(np.array([centroid]))[0])
        center_roi = next(s for s in sl if s.parent_index == center_idx)

        grid = np.indices(space.dim).reshape(3, -1).T
        spacing_arr = np.array(spacing)
        dist2 = np.sum(((grid - centroid) * spacing_arr) ** 2, axis=1)
        expected_n = int(np.sum(dist2 <= 4.0**2))

        assert len(center_roi) == expected_n


class TestSearchlightCoords:
    """Test searchlight_coords function."""
    
    def setup_method(self):
        """Set up test data."""
        self.mask_data = np.zeros((8, 8, 8), dtype=bool)
        self.mask_data[2:6, 2:6, 2:6] = True
        self.space = NeuroSpace(dim=[8, 8, 8], spacing=[1, 1, 1])
        self.mask = LogicalNeuroVol(self.mask_data, self.space)
    
    def test_searchlight_coords_returns_coordinates(self):
        """Test that searchlight_coords returns coordinate matrices."""
        sl_coords = searchlight_coords(self.mask, radius=2, nonzero=True)
        
        # Should return LazyList
        assert isinstance(sl_coords, LazyList)
        assert len(sl_coords) == 64  # 4x4x4 nonzero voxels
        
        # Get first coordinate set
        first_coords = sl_coords[0]
        assert isinstance(first_coords, np.ndarray)
        assert first_coords.shape[1] == 3  # Should have 3 columns (i,j,k)
        assert len(first_coords) > 0  # Should have some voxels
    
    def test_searchlight_coords_all_voxels(self):
        """nonzero=False should include all voxels as searchlight centers."""
        sl_coords = searchlight_coords(self.mask, radius=2, nonzero=False)
        
        # For nonzero=False, centers include all voxels in column-major/F-order.
        assert len(sl_coords) == self.mask_data.size
        expected = self.space.index_to_grid(np.arange(5))
        for i in range(5):
            np.testing.assert_array_equal(sl_coords[i], expected[i])

    def test_searchlight_coords_center_order_matches_all_voxel_grid(self):
        """searchlight_coords should iterate centers in NeuroSpace index_to_grid order."""
        sl_coords = searchlight_coords(self.mask, radius=1, nonzero=False)
        flat_idx = np.array([0, 3, 9, 18, 27], dtype=int)
        expected = self.space.index_to_grid(flat_idx)

        for idx, expected_coord in zip(flat_idx, expected):
            np.testing.assert_array_equal(sl_coords[idx], expected_coord)

    def test_searchlight_nonzero_controls_roi_content(self):
        """searchlight should keep zero-valued voxels only when nonzero=False."""
        mask = LogicalNeuroVol(np.zeros((3, 3, 3), dtype=bool), NeuroSpace((3, 3, 3)))
        mask.data[1, 1, 1] = True
        radius = 1.1

        coords_false = searchlight_coords(mask, radius=radius, nonzero=False)
        coords_true = searchlight_coords(mask, radius=radius, nonzero=True)
        assert len(coords_false) == 27
        assert len(coords_true) == 1

        assert len(coords_false[0]) > len(coords_true[0])
        assert len(coords_true[0]) == 1
        assert np.any(np.all(coords_true[0] == np.array([1, 1, 1]), axis=1))

        sl_false = searchlight(mask, radius=radius, eager=True, nonzero=False)
        sl_true = searchlight(mask, radius=radius, eager=True, nonzero=True)
        assert len(sl_false[0].coords) > len(sl_true[0].coords)
        assert len(sl_true[0].coords) == 1
        assert np.any(np.all(sl_true[0].coords == np.array([1, 1, 1]), axis=1))


class TestSearchlightCenterIndex:
    """Tests for `searchlight` exhaustive windows."""

    def setup_method(self):
        self.mask_data = np.zeros((10, 10, 10), dtype=bool)
        self.mask_data[3:7, 3:7, 3:7] = True
        self.space = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2])
        self.mask = LogicalNeuroVol(self.mask_data, self.space)

    def test_searchlight_center_index_tracks_center_voxel(self):
        """`center_index` should always locate the center grid row."""
        sl = searchlight(self.mask, radius=2, eager=True, nonzero=True)

        for i in range(0, len(sl), 7):
            roi = sl[i]
            center_grid = roi.parent_grid
            center_rows = np.where(np.all(roi.coords == center_grid, axis=1))[0]
            expected = center_rows[0] if len(center_rows) else 0
            assert roi.center_index == int(expected)


class TestRandomSearchlight:
    """Test random searchlight function."""
    
    def setup_method(self):
        """Set up test data."""
        # Create larger mask for random searchlight
        self.mask_data = np.zeros((20, 20, 20), dtype=bool)
        self.mask_data[5:15, 5:15, 5:15] = True
        self.space = NeuroSpace(dim=[20, 20, 20], spacing=[1, 1, 1])
        self.mask = LogicalNeuroVol(self.mask_data, self.space)
    
    def test_random_searchlight_non_overlapping(self):
        """Test that random searchlight creates non-overlapping regions."""
        np.random.seed(42)  # For reproducibility
        sl_list = random_searchlight(self.mask, radius=3)
        
        # Should return list
        assert isinstance(sl_list, list)
        assert len(sl_list) > 0
        
        # Check that centers are unique
        center_indices = [sl.parent_index for sl in sl_list]
        assert len(set(center_indices)) == len(center_indices)
        
        # Collect all covered voxels
        all_covered = set()
        for sl in sl_list:
            assert isinstance(sl, ROIVolWindow)
            # Get indices for this searchlight
            indices = self.space.grid_to_index(sl.coords)
            all_covered.update(indices)
        
        # Should cover a significant portion of the mask
        total_mask_voxels = np.sum(self.mask_data)
        assert len(all_covered) > total_mask_voxels * 0.3  # Lower threshold due to non-overlapping algorithm
    
    def test_random_searchlight_properties(self):
        """Test properties of random searchlights."""
        np.random.seed(42)
        sl_list = random_searchlight(self.mask, radius=5)
        
        for sl in sl_list:
            # Check ROIVolWindow properties
            assert isinstance(sl, ROIVolWindow)
            assert sl.parent_index is not None
            center_grid = sl.parent_grid
            center_rows = np.where(np.all(sl.coords == center_grid, axis=1))[0]
            expected_center = center_rows[0] if len(center_rows) else 0
            assert sl.center_index == expected_center
            
            # Check that center is within mask
            assert self.mask.data[tuple(center_grid)]

    def test_random_searchlight_respects_nonzero(self):
        """random_searchlight should keep zeros when nonzero=False."""
        mask_data = np.zeros((3, 3, 3), dtype=bool)
        mask_data[1, 1, 1] = True
        space = NeuroSpace((3, 3, 3))
        mask = LogicalNeuroVol(mask_data, space)

        sl_true = random_searchlight(mask, radius=1.5, nonzero=True)
        sl_false = random_searchlight(mask, radius=1.5, nonzero=False)

        assert len(sl_true) == 1
        assert len(sl_false) == 1
        assert len(sl_true[0].coords) < len(sl_false[0].coords)


class TestBootstrapSearchlight:
    """Test bootstrap searchlight function."""
    
    def setup_method(self):
        """Set up test data."""
        self.mask_data = np.ones((10, 10, 10), dtype=bool)
        self.space = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2])
        self.mask = LogicalNeuroVol(self.mask_data, self.space)
    
    def test_bootstrap_searchlight_iterations(self):
        """Test bootstrap searchlight with specified iterations."""
        np.random.seed(42)
        n_iter = 50
        sl_list = bootstrap_searchlight(self.mask, radius=6, iter=n_iter)
        
        # Should return exact number of iterations
        assert isinstance(sl_list, list)
        assert len(sl_list) == n_iter
        
        # Each should be ROIVolWindow
        for sl in sl_list:
            assert isinstance(sl, ROIVolWindow)
            assert len(sl.coords) > 0
    
    def test_bootstrap_searchlight_default_params(self):
        """Test bootstrap searchlight with default parameters."""
        np.random.seed(42)
        sl_list = bootstrap_searchlight(self.mask)
        
        # Default is 100 iterations with radius 8
        assert len(sl_list) == 100
        
        # Check properties
        for sl in sl_list:
            assert isinstance(sl, ROIVolWindow)
            assert sl.parent_index is not None
            center_grid = sl.parent_grid
            center_rows = np.where(np.all(sl.coords == center_grid, axis=1))[0]
            expected_center = center_rows[0] if len(center_rows) else 0
            assert sl.center_index == expected_center
    
    def test_bootstrap_searchlight_sampling(self):
        """Test that bootstrap samples with replacement."""
        np.random.seed(42)
        sl_list = bootstrap_searchlight(self.mask, radius=4, iter=200)
        
        # With 200 iterations from 1000 voxels, we should have duplicates
        parent_indices = [sl.parent_index for sl in sl_list]
        assert len(set(parent_indices)) < 200  # Some duplicates


class TestClusteredSearchlight:
    """Test clustered searchlight function."""
    
    def setup_method(self):
        """Set up test data."""
        self.mask_data = np.zeros((20, 20, 20), dtype=bool)
        self.mask_data[2:18, 2:18, 2:18] = True
        self.space = NeuroSpace(dim=[20, 20, 20], spacing=[1, 1, 1])
        self.mask = LogicalNeuroVol(self.mask_data, self.space)
    
    def test_clustered_searchlight_kmeans(self):
        """Test clustered searchlight with k-means."""
        n_clusters = 5
        clusters = list(clustered_searchlight(self.mask, radius=10, csize=n_clusters))
        
        # Should return specified number of clusters
        assert len(clusters) == n_clusters
        
        # Each cluster should be ROIVol
        total_voxels = 0
        for cluster in clusters:
            assert isinstance(cluster, ROIVol)
            assert len(cluster.coords) > 0
            assert len(cluster.data) == len(cluster.coords)
            total_voxels += len(cluster.coords)
        
        # Total voxels should equal mask voxels
        assert total_voxels == np.sum(self.mask_data)
    
    def test_clustered_searchlight_with_cvol(self):
        """Test clustered searchlight with pre-defined clusters."""
        # Create cluster volume
        cluster_data = np.zeros_like(self.mask_data, dtype=int)
        # Create 3 regions
        cluster_data[2:8, 2:18, 2:18] = 1
        cluster_data[8:14, 2:18, 2:18] = 2
        cluster_data[14:18, 2:18, 2:18] = 3
        
        cvol = DenseNeuroVol(cluster_data, self.space)
        
        clusters = list(clustered_searchlight(self.mask, radius=10, cvol=cvol))
        
        # Should return 3 clusters (labels 1, 2, 3)
        assert len(clusters) == 3
        
        # Check cluster sizes match
        for i, cluster in enumerate(clusters):
            expected_size = np.sum((cluster_data == (i + 1)) & self.mask_data)
            assert len(cluster.coords) == expected_size

    def test_clustered_searchlight_cvol_uses_f_order_labels(self):
        """cluster label lookup must follow NeuroSpace/`index_to_grid` ordering."""
        shape = (4, 3, 2)
        space = NeuroSpace(dim=list(shape), spacing=[1, 1, 1])
        mask_data = np.ones(shape, dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)

        cvol_data = np.arange(np.prod(shape), dtype=int).reshape(shape, order='F') % 3
        cvol = DenseNeuroVol(cvol_data, space)

        cvol_clusters = list(clustered_searchlight(mask, cvol=cvol))
        assert len(cvol_clusters) == 3

        for cluster in cvol_clusters:
            labels = cvol_data[tuple(cluster.coords.T)]
            assert np.all(labels == labels[0])
    
    def test_clustered_searchlight_error(self):
        """Test error when neither cvol nor csize provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            list(clustered_searchlight(self.mask, radius=10))


class TestSearchlightIntegration:
    """Integration tests for searchlight functionality."""
    
    def test_searchlight_with_dense_neurovol(self):
        """Test searchlight with DenseNeuroVol mask."""
        # Create continuous mask
        mask_data = np.random.rand(10, 10, 10)
        mask_data[mask_data < 0.5] = 0
        space = NeuroSpace(dim=[10, 10, 10])
        mask = DenseNeuroVol(mask_data, space)
        
        # Should convert to LogicalNeuroVol internally
        sl = searchlight(mask, radius=3, eager=True, nonzero=True)
        
        assert isinstance(sl, list)
        assert len(sl) == np.sum(mask_data > 0)
    
    def test_searchlight_coords_consistency(self):
        """Test consistency between searchlight and searchlight_coords."""
        mask_data = np.ones((8, 8, 8), dtype=bool)
        space = NeuroSpace(dim=[8, 8, 8])
        mask = LogicalNeuroVol(mask_data, space)
        
        # Get searchlights and coords
        sl_list = searchlight(mask, radius=3, eager=True, nonzero=True)
        sl_coords = searchlight_coords(mask, radius=3, nonzero=True)
        
        # Should have same length
        assert len(sl_list) == len(sl_coords)
        
        # Coordinates should match
        for i in range(len(sl_list)):
            roi_coords = sl_list[i].coords
            coord_coords = sl_coords[i]
            assert np.array_equal(roi_coords, coord_coords)
    
    def test_searchlight_memory_efficiency(self):
        """Test that lazy evaluation is memory efficient."""
        # Create large mask
        mask_data = np.ones((50, 50, 50), dtype=bool)
        space = NeuroSpace(dim=[50, 50, 50])
        mask = LogicalNeuroVol(mask_data, space)
        
        # Lazy searchlight should not consume much memory
        sl_lazy = searchlight(mask, radius=5, eager=False)
        
        # Should be able to access individual elements efficiently
        assert isinstance(sl_lazy, LazyList)
        assert len(sl_lazy) == 125000  # 50^3
        
        # Access a few searchlights
        for i in [0, 1000, 50000, 124999]:
            sl = sl_lazy[i]
            assert isinstance(sl, ROIVolWindow)
