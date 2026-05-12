"""Comprehensive tests for ROI classes based on R neuroim2 tests."""

import pytest
import numpy as np
from neuroim import (
    NeuroSpace, DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol,
    ROICoords, ROIVol, ROIVec, ROIVolWindow,
    roicoords, roivol, square_roi, cuboid_roi, spherical_roi, spherical_roi_set
)


class TestROICoords:
    """Test ROICoords functionality."""
    
    def test_roi_coords_construction(self):
        """Test basic ROICoords construction."""
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        roi = ROICoords(coords)
        
        assert len(roi) == 3
        assert roi.dim == (3, 3)
        assert np.array_equal(roi.coords, coords)
    
    def test_roi_coords_with_space(self):
        """Test ROICoords with explicit space."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        space = NeuroSpace(dim=[10, 10, 10])
        roi = ROICoords(coords, space)
        
        assert roi.space.dim[0] == 10
        assert len(roi) == 2
    
    def test_roi_coords_indexing(self):
        """Test ROICoords indexing."""
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        roi = ROICoords(coords)
        
        # Single coordinate
        sub_roi = roi[0]
        assert isinstance(sub_roi, ROICoords)
        assert len(sub_roi) == 1
        assert np.array_equal(sub_roi.coords, [[1, 2, 3]])
        
        # Multiple coordinates
        sub_roi = roi[0:2]
        assert len(sub_roi) == 2
        assert np.array_equal(sub_roi.coords, coords[0:2])
    
    def test_roi_coords_indices(self):
        """Test linear index calculation."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        space = NeuroSpace(dim=[10, 10, 10])
        roi = ROICoords(coords, space)
        
        indices = roi.indices()
        expected = np.array([0, 1, 10, 100])  # Column-major order
        np.testing.assert_array_equal(indices, expected)


class TestROIVol:
    """Test ROIVol functionality."""
    
    def test_roi_vol_construction(self):
        """Test basic ROIVol construction."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([10.0, 20.0])
        space = NeuroSpace(dim=[10, 10, 10])
        
        roi = ROIVol(data, space, coords)
        
        assert len(roi) == 2
        assert np.array_equal(roi.data, data)
        assert np.array_equal(roi.coords, coords)
    
    def test_roi_vol_indexing(self):
        """Test ROIVol indexing."""
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data = np.array([10.0, 20.0, 30.0])
        space = NeuroSpace(dim=[10, 10, 10])
        roi = ROIVol(data, space, coords)
        
        # Extract subset
        sub_roi = roi[0:2]
        assert isinstance(sub_roi, ROIVol)
        assert len(sub_roi) == 2
        assert np.array_equal(sub_roi.data, [10.0, 20.0])
        
        # Extract single value
        sub_roi = roi[1]
        assert len(sub_roi) == 1
        assert sub_roi.data[0] == 20.0
    
    def test_roi_vol_conversions(self):
        """Test ROIVol conversions."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([10.0, 20.0])
        space = NeuroSpace(dim=[10, 10, 10])
        roi = ROIVol(data, space, coords)
        
        # Convert to sparse volume
        sparse_vol = roi.as_sparse()
        assert isinstance(sparse_vol, SparseNeuroVol)
        assert sparse_vol[1, 2, 3] == 10.0
        assert sparse_vol[4, 5, 6] == 20.0
        
        # Convert to logical volume
        logical_vol = roi.as_logical()
        assert isinstance(logical_vol, LogicalNeuroVol)
        assert logical_vol[1, 2, 3] == True
        assert logical_vol[4, 5, 6] == True
        assert logical_vol[0, 0, 0] == False
    
    def test_roi_vol_as_numeric(self):
        """Test as_numeric method."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([10.0, 20.0])
        space = NeuroSpace(dim=[10, 10, 10])
        roi = ROIVol(data, space, coords)
        
        numeric = roi.as_numeric()
        assert isinstance(numeric, np.ndarray)
        np.testing.assert_array_equal(numeric, data)
    
    def test_roi_vol_get_coords(self):
        """Test coordinate extraction."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([10.0, 20.0])
        space = NeuroSpace(dim=[10, 10, 10], 
                          spacing=[2, 2, 2],
                          origin=[10, 20, 30])
        roi = ROIVol(data, space, coords)
        
        # Grid coordinates
        grid_coords = roi.get_coords(real=False)
        np.testing.assert_array_equal(grid_coords, coords)
        
        # Real-world coordinates
        real_coords = roi.get_coords(real=True)
        # Should be transformed by spacing and origin
        expected = (coords - 0.5) * 2 + [10, 20, 30]
        np.testing.assert_array_almost_equal(real_coords, expected)


class TestROICreation:
    """Test ROI creation functions."""
    
    def setup_method(self):
        """Create test volume."""
        self.vol = DenseNeuroVol(
            np.random.randn(20, 20, 20),
            NeuroSpace(dim=[20, 20, 20])
        )
    
    def test_square_roi(self):
        """Test square ROI creation."""
        # Create square ROI in z=10 plane
        roi = square_roi(self.vol, centroid=[10, 10, 10], surround=3, fixdim=2)
        
        assert isinstance(roi, ROIVol)
        # Should be 7x7 square (center +/- 3)
        assert len(roi) == 49
        
        # Check all z-coordinates are 10
        assert np.all(roi.coords[:, 2] == 10)
        
        # Check bounds
        assert np.min(roi.coords[:, 0]) == 7
        assert np.max(roi.coords[:, 0]) == 13
        assert np.min(roi.coords[:, 1]) == 7
        assert np.max(roi.coords[:, 1]) == 13
    
    def test_square_roi_with_fill(self):
        """Test square ROI with fill value."""
        roi = square_roi(self.vol, centroid=[10, 10, 10], 
                        surround=2, fill=99.0, fixdim=1)
        
        assert np.all(roi.data == 99.0)
    
    def test_square_roi_nonzero(self):
        """Test square ROI with nonzero filter."""
        # Create volume with some zeros
        vol = DenseNeuroVol(
            np.zeros((20, 20, 20)),
            NeuroSpace(dim=[20, 20, 20])
        )
        vol.data[8:13, 8:13, 10] = 1.0
        
        roi = square_roi(vol, centroid=[10, 10, 10], 
                        surround=3, nonzero=True, fixdim=2)
        
        # Should only include non-zero voxels
        assert np.all(roi.data != 0)
        assert len(roi) < 49  # Less than full square
    
    def test_cuboid_roi(self):
        """Test cuboid ROI creation."""
        roi = cuboid_roi(self.vol, centroid=[10, 10, 10], surround=2)
        
        assert isinstance(roi, ROIVol)
        # Should be 5x5x5 cube (center +/- 2)
        assert len(roi) == 125
        
        # Check bounds
        assert np.min(roi.coords[:, 0]) == 8
        assert np.max(roi.coords[:, 0]) == 12
    
    def test_cuboid_roi_asymmetric(self):
        """Test cuboid ROI with different surrounds."""
        roi = cuboid_roi(self.vol, centroid=[10, 10, 10], 
                        surround=[1, 2, 3])
        
        # Should be 3x5x7
        assert len(roi) == 3 * 5 * 7
        
        # Check bounds
        assert np.min(roi.coords[:, 0]) == 9
        assert np.max(roi.coords[:, 0]) == 11
        assert np.min(roi.coords[:, 1]) == 8
        assert np.max(roi.coords[:, 1]) == 12
        assert np.min(roi.coords[:, 2]) == 7
        assert np.max(roi.coords[:, 2]) == 13
    
    def test_spherical_roi(self):
        """Test spherical ROI creation."""
        roi = spherical_roi(self.vol, centroid=[10, 10, 10], radius=3.0)
        
        assert isinstance(roi, ROIVol)
        
        # Check all points are within radius
        center = np.array([10, 10, 10])
        distances = np.sqrt(np.sum((roi.coords - center)**2, axis=1))
        assert np.all(distances <= 3.0)
        
        # Should include center
        center_idx = np.where((roi.coords == center).all(axis=1))[0]
        assert len(center_idx) == 1
    
    def test_spherical_roi_with_fill(self):
        """Test spherical ROI with fill value."""
        roi = spherical_roi(self.vol, centroid=[10, 10, 10], 
                           radius=2.5, fill=42.0)
        
        assert np.all(roi.data == 42.0)
    
    def test_spherical_roi_set(self):
        """Test creating multiple spherical ROIs."""
        centroids = np.array([[5, 5, 5], [10, 10, 10], [15, 15, 15]])
        roi_list = spherical_roi_set(self.vol, centroids, radius=2.0)
        
        assert len(roi_list) == 3
        assert all(isinstance(r, ROIVol) for r in roi_list)
        
        # Check each ROI is centered correctly
        for i, roi in enumerate(roi_list):
            center = centroids[i]
            distances = np.sqrt(np.sum((roi.coords - center)**2, axis=1))
            assert np.all(distances <= 2.0)
    
    def test_spherical_roi_set_with_different_fills(self):
        """Test spherical ROI set with different fill values."""
        centroids = np.array([[5, 5, 5], [10, 10, 10]])
        fills = [100.0, 200.0]
        roi_list = spherical_roi_set(self.vol, centroids, 
                                    radius=1.5, fill=fills)
        
        assert roi_list[0].data[0] == 100.0
        assert roi_list[1].data[0] == 200.0


class TestROIVolWindow:
    """Test ROIVolWindow functionality."""
    
    def test_roi_vol_window_construction(self):
        """Test ROIVolWindow construction."""
        coords = np.array([[9, 9, 9], [10, 10, 10], [11, 11, 11]])
        data = np.array([1.0, 2.0, 3.0])
        space = NeuroSpace(dim=[20, 20, 20])
        
        # Parent index is the linear index in parent space
        parent_idx = np.ravel_multi_index((10, 10, 10), (20, 20, 20), order='F')
        center_idx = 1  # Index in coords where center is
        
        roi_win = ROIVolWindow(data, space, coords, parent_idx, center_idx)
        
        assert isinstance(roi_win, ROIVolWindow)
        assert roi_win.parent_index == parent_idx
        assert roi_win.center_index == center_idx
        assert len(roi_win) == 3


class TestROIVec:
    """Test ROIVec functionality."""
    
    def test_roi_vec_construction(self):
        """Test ROIVec construction."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([[1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0],
                        [4.0, 40.0]])  # time x voxels
        space = NeuroSpace(dim=[10, 10, 10])
        
        roi_vec = ROIVec(data, space, coords)
        
        assert roi_vec.data.shape == (4, 2)
        assert len(roi_vec) == 2  # Number of voxels
    
    def test_roi_vec_indexing(self):
        """Test ROIVec data access."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([[1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0]])
        space = NeuroSpace(dim=[10, 10, 10])
        roi_vec = ROIVec(data, space, coords)
        
        # Extract time series for first voxel
        ts = roi_vec[:, 0]
        np.testing.assert_array_equal(ts, [1.0, 2.0, 3.0])
        
        # Extract time series for second voxel
        ts = roi_vec[:, 1]
        np.testing.assert_array_equal(ts, [10.0, 20.0, 30.0])
    
    def test_roi_vec_setitem(self):
        """Test ROIVec data modification."""
        coords = np.array([[1, 2, 3]])
        data = np.array([[1.0], [2.0], [3.0]])
        space = NeuroSpace(dim=[10, 10, 10])
        roi_vec = ROIVec(data, space, coords)
        
        # Modify data
        roi_vec[:, 0] = [10.0, 20.0, 30.0]
        np.testing.assert_array_equal(roi_vec.data[:, 0], [10.0, 20.0, 30.0])


class TestROIFactoryFunctions:
    """Test ROI factory functions."""
    
    def test_roicoords_factory(self):
        """Test roicoords factory function."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        roi = roicoords(coords)
        
        assert isinstance(roi, ROICoords)
        assert len(roi) == 2
        
        # Test validation
        with pytest.raises(ValueError):
            roicoords(np.array([[1, 2]]))  # Wrong number of columns
    
    def test_roivol_factory(self):
        """Test roivol factory function."""
        space = NeuroSpace(dim=[10, 10, 10])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([10.0, 20.0])
        
        roi = roivol(space, coords, data)
        
        assert isinstance(roi, ROIVol)
        assert len(roi) == 2
        
        # Test validation
        with pytest.raises(ValueError):
            roivol(space, coords, [1, 2, 3])  # Wrong data length


class TestROIEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_roi(self):
        """Test empty ROI handling."""
        coords = np.array([]).reshape(0, 3)
        roi = ROICoords(coords)
        assert len(roi) == 0
        
        # Empty ROIVol
        data = np.array([])
        space = NeuroSpace(dim=[10, 10, 10])
        roi_vol = ROIVol(data, space, coords)
        assert len(roi_vol) == 0
    
    def test_roi_boundary_conditions(self):
        """Test ROI creation at volume boundaries."""
        vol = DenseNeuroVol(
            np.ones((10, 10, 10)),
            NeuroSpace(dim=[10, 10, 10])
        )
        
        # ROI at corner
        roi = cuboid_roi(vol, centroid=[0, 0, 0], surround=2)
        assert len(roi) > 0
        assert np.min(roi.coords) == 0
        
        # ROI at edge
        roi = spherical_roi(vol, centroid=[9, 9, 9], radius=3.0)
        assert len(roi) > 0
        assert np.max(roi.coords[:, 0]) == 9


if __name__ == "__main__":
    pytest.main([__file__])