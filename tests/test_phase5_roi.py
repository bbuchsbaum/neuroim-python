"""Tests for Phase 5: ROI Classes.

Tests ROI, ROICoords, ROIVol, ROIVec, ROIVolWindow and construction functions.
"""

import pytest
import numpy as np
from neuroimpy import (
    NeuroSpace, DenseNeuroVol,
    ROI, ROICoords, ROIVol, ROIVec, ROIVolWindow,
    roicoords, roivol,
    square_roi, cuboid_roi, spherical_roi, spherical_roi_set
)


class TestROICoords:
    """Test ROICoords class."""
    
    def test_roicoords_creation(self):
        """Test basic ROICoords creation."""
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        roi = ROICoords(coords)
        
        assert isinstance(roi, ROI)  # Check inheritance
        assert np.array_equal(roi.coords, coords)
        assert len(roi) == 3
        assert roi.dim == (3, 3)
    
    def test_roicoords_factory(self):
        """Test roicoords factory function."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        roi = roicoords(coords)
        
        assert isinstance(roi, ROICoords)
        assert np.array_equal(roi.coords, coords)
    
    def test_roicoords_invalid(self):
        """Test invalid coordinates."""
        # Wrong number of columns
        with pytest.raises(ValueError, match="3 columns"):
            ROICoords(np.array([[1, 2], [3, 4]]))
    
    def test_roicoords_indexing(self):
        """Test ROICoords indexing."""
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        roi = ROICoords(coords)
        
        # Single row
        sub = roi[1]
        assert isinstance(sub, ROICoords)
        assert len(sub) == 1
        assert np.array_equal(sub.coords, [[4, 5, 6]])
        
        # Multiple rows
        sub = roi[0:2]
        assert len(sub) == 2
        assert np.array_equal(sub.coords, coords[0:2])
    
    def test_roicoords_indices(self):
        """Test linear index calculation."""
        space = NeuroSpace([10, 10, 10])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        roi = ROICoords(coords, space)
        
        indices = roi.indices()
        # Check indices match expected values
        # In Fortran order: idx = i + j*dim[0] + k*dim[0]*dim[1]
        expected = [1 + 2*10 + 3*100, 4 + 5*10 + 6*100]
        assert np.array_equal(indices, expected)


class TestROIVol:
    """Test ROIVol class."""
    
    def test_roivol_creation(self):
        """Test basic ROIVol creation."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([1.5, 2.5])
        
        roi = ROIVol(data, space, coords)
        
        assert isinstance(roi, ROICoords)  # Check inheritance
        assert np.array_equal(roi.data, data)
        assert np.array_equal(roi.coords, coords)
        assert len(roi) == 2
    
    def test_roivol_factory(self):
        """Test roivol factory function."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([1.5, 2.5])
        
        roi = roivol(space, coords, data)
        
        assert isinstance(roi, ROIVol)
        assert np.array_equal(roi.data, data)
    
    def test_roivol_invalid(self):
        """Test invalid ROIVol creation."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Mismatched data length
        with pytest.raises(ValueError, match="length of data"):
            ROIVol(np.array([1.0]), space, coords)
    
    def test_roivol_indexing(self):
        """Test ROIVol indexing."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data = np.array([1.0, 2.0, 3.0])
        roi = ROIVol(data, space, coords)
        
        # Extract single value
        sub = roi[1]
        assert isinstance(sub, ROIVol)
        assert len(sub) == 1
        assert sub.data[0] == 2.0
        
        # Extract coordinates
        assert roi[1, 2] == 6  # coords[1, 2]
        
        # Extract all data
        assert np.array_equal(roi[None], data)
    
    def test_roivol_as_numeric(self):
        """Test as_numeric conversion."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([1.5, 2.5])
        roi = ROIVol(data, space, coords)
        
        numeric = roi.as_numeric()
        assert np.array_equal(numeric, data)
    
    def test_roivol_as_sparse(self):
        """Test conversion to SparseNeuroVol."""
        space = NeuroSpace([10, 10, 10])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([1.5, 2.5])
        roi = ROIVol(data, space, coords)
        
        sparse = roi.as_sparse()
        assert sparse.shape == (10, 10, 10)
        assert sparse[1, 2, 3] == 1.5
        assert sparse[4, 5, 6] == 2.5
    
    def test_roivol_as_logical(self):
        """Test conversion to LogicalNeuroVol."""
        space = NeuroSpace([10, 10, 10])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([1.5, 2.5])
        roi = ROIVol(data, space, coords)
        
        logical = roi.as_logical()
        assert logical.shape == (10, 10, 10)
        assert logical[1, 2, 3] == True
        assert logical[4, 5, 6] == True
        assert logical[0, 0, 0] == False
    
    def test_roivol_coords_real(self):
        """Test real coordinate conversion."""
        space = NeuroSpace([10, 10, 10])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([1.5, 2.5])
        roi = ROIVol(data, space, coords)
        
        # Grid coordinates
        grid_coords = roi.get_coords(real=False)
        assert np.array_equal(grid_coords, coords)
        
        # Real coordinates (with transformation)
        real_coords = roi.get_coords(real=True)
        assert real_coords.shape == (2, 3)
        # Should apply transformation matrix


class TestROIVec:
    """Test ROIVec class."""
    
    def test_roivec_creation(self):
        """Test basic ROIVec creation."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 timepoints, 2 voxels
        
        roi = ROIVec(data, space, coords)
        
        assert isinstance(roi, ROICoords)
        assert np.array_equal(roi.data, data)
        assert np.array_equal(roi.coords, coords)
    
    def test_roivec_invalid(self):
        """Test invalid ROIVec creation."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Wrong data shape
        with pytest.raises(ValueError, match="ncol.*nrow"):
            ROIVec(np.array([[1.0, 2.0, 3.0]]), space, coords)
    
    def test_roivec_indexing(self):
        """Test ROIVec column access."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        roi = ROIVec(data, space, coords)
        
        # Get column
        col = roi[0]
        assert np.array_equal(col, [1.0, 3.0, 5.0])
        
        # Set column
        roi[1] = [7.0, 8.0, 9.0]
        assert np.array_equal(roi[1], [7.0, 8.0, 9.0])


class TestROIVolWindow:
    """Test ROIVolWindow class."""
    
    def test_roivolwindow_creation(self):
        """Test ROIVolWindow creation."""
        space = NeuroSpace([64, 64, 64])
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data = np.array([1.0, 2.0, 3.0])
        
        roi = ROIVolWindow(data, space, coords, parent_index=100, center_index=1)
        
        assert isinstance(roi, ROIVol)
        assert roi.parent_index == 100
        np.testing.assert_array_equal(
            roi.parent_grid,
            space.index_to_grid(np.array([100], dtype=int))[0],
        )
        assert roi.center_index == 1
        assert len(roi) == 3


class TestROIConstructionFunctions:
    """Test ROI construction functions."""
    
    def test_square_roi(self):
        """Test square_roi function."""
        # Create test volume
        vol = DenseNeuroVol(np.random.randn(20, 20, 20), NeuroSpace([20, 20, 20]))
        
        # Create square ROI in z=10 plane
        roi = square_roi(vol, centroid=[10, 10, 10], surround=3, fixdim=2)
        
        assert isinstance(roi, ROIVol)
        # Should be 7x7 square (center ± 3)
        assert len(roi) == 49
        
        # Check all z-coordinates are 10
        assert np.all(roi.coords[:, 2] == 10)
        
        # Test with fill value
        roi_fill = square_roi(vol, centroid=[10, 10, 10], surround=2, 
                              fill=5.0, fixdim=2)
        assert np.all(roi_fill.data == 5.0)
    
    def test_cuboid_roi(self):
        """Test cuboid_roi function."""
        vol = DenseNeuroVol(np.random.randn(20, 20, 20), NeuroSpace([20, 20, 20]))
        
        # Uniform surround
        roi = cuboid_roi(vol, centroid=[10, 10, 10], surround=2)
        assert isinstance(roi, ROIVol)
        # Should be 5x5x5 cube
        assert len(roi) == 125
        
        # Non-uniform surround
        roi2 = cuboid_roi(vol, centroid=[10, 10, 10], surround=[1, 2, 3])
        # Should be 3x5x7 box
        assert len(roi2) == 3 * 5 * 7
        
        # Test with fill
        roi_fill = cuboid_roi(vol, centroid=[10, 10, 10], surround=1, fill=3.0)
        assert np.all(roi_fill.data == 3.0)
    
    def test_spherical_roi(self):
        """Test spherical_roi function."""
        vol = DenseNeuroVol(np.random.randn(20, 20, 20), NeuroSpace([20, 20, 20]))
        
        # Create spherical ROI
        roi = spherical_roi(vol, centroid=[10, 10, 10], radius=3.5)
        
        assert isinstance(roi, ROIVol)
        
        # Check all points are within radius
        distances = np.sqrt(np.sum((roi.coords - [10, 10, 10])**2, axis=1))
        assert np.all(distances <= 3.5)
        
        # Test with fill
        roi_fill = spherical_roi(vol, centroid=[10, 10, 10], radius=2.0, fill=7.0)
        assert np.all(roi_fill.data == 7.0)

    def test_spherical_roi_radius_min_spacing(self):
        """radius smaller than minimum spacing should raise."""
        vol = DenseNeuroVol(np.ones((10, 10, 10)), NeuroSpace([10, 10, 10], spacing=(2.0, 2.0, 2.0)))
        with pytest.raises(ValueError, match="radius is too small"):
            spherical_roi(vol, centroid=[5, 5, 5], radius=1.0)

    def test_spherical_roi_spacing(self):
        """spherical_roi should account for anisotropic voxel spacing."""
        spacing = (2.0, 2.0, 3.0)
        vol = DenseNeuroVol(
            np.random.randn(11, 11, 11),
            NeuroSpace([11, 11, 11], spacing=spacing),
        )
        centroid = np.array([5, 5, 5], dtype=float)
        radius = 4.0

        roi = spherical_roi(vol, centroid=centroid, radius=radius)

        grid = np.indices((11, 11, 11)).reshape(3, -1).T
        spacing_arr = np.array(spacing)
        dist2 = np.sum(((grid - centroid) * spacing_arr) ** 2, axis=1)
        expected_n = int(np.sum(dist2 <= radius**2))

        assert len(roi) == expected_n
    
    def test_spherical_roi_nonzero(self):
        """Test spherical_roi with nonzero option."""
        # Create volume with some zeros
        data = np.ones((20, 20, 20))
        data[8:13, 8:13, 8:13] = 0  # Zero out center region
        vol = DenseNeuroVol(data, NeuroSpace([20, 20, 20]))
        
        # Create ROI including zeros
        roi_all = spherical_roi(vol, centroid=[10, 10, 10], radius=3.0, nonzero=False)
        
        # Create ROI excluding zeros
        roi_nonzero = spherical_roi(vol, centroid=[10, 10, 10], radius=3.0, nonzero=True)
        
        # Should have fewer voxels when excluding zeros
        assert len(roi_nonzero) < len(roi_all)
        assert np.all(roi_nonzero.data != 0)
    
    def test_spherical_roi_set(self):
        """Test spherical_roi_set function."""
        vol = DenseNeuroVol(np.random.randn(30, 30, 30), NeuroSpace([30, 30, 30]))
        
        # Multiple centroids
        centroids = np.array([[10, 10, 10], [20, 20, 20], [15, 15, 15]])
        
        # Create ROI set
        roi_list = spherical_roi_set(vol, centroids, radius=2.5)
        
        assert len(roi_list) == 3
        assert all(isinstance(roi, ROIVol) for roi in roi_list)
        
        # Test with different fill values
        roi_list_fill = spherical_roi_set(vol, centroids, radius=2.0, 
                                          fill=[1.0, 2.0, 3.0])
        assert np.all(roi_list_fill[0].data == 1.0)
        assert np.all(roi_list_fill[1].data == 2.0)
        assert np.all(roi_list_fill[2].data == 3.0)
    
    def test_roi_edge_cases(self):
        """Test ROI construction at volume edges."""
        vol = DenseNeuroVol(np.ones((10, 10, 10)), NeuroSpace([10, 10, 10]))
        
        # ROI at corner
        roi = cuboid_roi(vol, centroid=[0, 0, 0], surround=2)
        assert len(roi) > 0
        assert np.all(roi.coords >= 0)
        assert np.all(roi.coords < 10)
        
        # ROI at edge  
        roi2 = spherical_roi(vol, centroid=[9, 9, 9], radius=3.0)
        assert len(roi2) > 0
        assert np.all(roi2.coords >= 0)
        assert np.all(roi2.coords < 10)


class TestROIIntegration:
    """Integration tests with other components."""
    
    def test_roi_with_sparse_vol(self):
        """Test ROI extraction from sparse volume."""
        from neuroimpy import SparseNeuroVol
        
        space = NeuroSpace([20, 20, 20])
        indices = np.array([100, 200, 300, 400, 500])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sparse_vol = SparseNeuroVol(data, space, indices)
        
        # Extract ROI
        roi = cuboid_roi(sparse_vol, centroid=[5, 0, 0], surround=1)
        
        assert isinstance(roi, ROIVol)
        # Check we got the right voxel if index 100 corresponds to [5,0,0]
        # (depends on the actual coordinate)
    
    def test_roi_repr(self):
        """Test string representations."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        roi_coords = ROICoords(coords)
        
        repr_str = repr(roi_coords)
        assert "ROICoords" in repr_str
        assert "2 x 3" in repr_str
        
        # ROIVol repr
        space = NeuroSpace([64, 64, 64])
        data = np.array([1.5, 2.5])
        roi_vol = ROIVol(data, space, coords)
        
        repr_str = repr(roi_vol)
        assert "ROIVol Object" in repr_str
        assert "ROI Points" in repr_str
        assert "Value Range" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
