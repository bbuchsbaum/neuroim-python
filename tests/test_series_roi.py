"""Tests for series_roi functionality."""

import pytest
import numpy as np
from neuroimpy import (
    NeuroSpace, DenseNeuroVec, SparseNeuroVec, 
    ROICoords, ROIVol, DenseNeuroVol, LogicalNeuroVol
)


class TestSeriesROI:
    """Test series_roi method for NeuroVec classes."""
    
    def setup_method(self):
        """Set up test data."""
        # Create 4D space
        self.space_4d = NeuroSpace(dim=[10, 10, 10, 5])
        
        # Create test data with known pattern
        # DenseNeuroVec expects data in shape (x, y, z, time)
        self.data = np.zeros((10, 10, 10, 5))
        # Fill specific voxels with distinct patterns
        for t in range(5):
            self.data[2, 3, 4, t] = t + 1  # Linear pattern
            self.data[5, 6, 7, t] = (t + 1) * 2  # Double pattern
            self.data[8, 8, 8, t] = (t + 1) ** 2  # Square pattern
        
        # Create NeuroVec
        self.vec = DenseNeuroVec(self.data, self.space_4d)
        
        # Create ROI with the three voxels
        roi_coords = np.array([[2, 3, 4], [5, 6, 7], [8, 8, 8]])
        self.roi_coords = ROICoords(roi_coords, NeuroSpace(dim=[10, 10, 10]))
        
        # Create ROIVol with same coordinates
        roi_data = np.array([1.0, 2.0, 3.0])  # Values don't matter for series extraction
        self.roi_vol = ROIVol(roi_data, NeuroSpace(dim=[10, 10, 10]), roi_coords)
    
    def test_series_roi_with_roi_coords(self):
        """Test series_roi with ROICoords input."""
        series = self.vec.series_roi(self.roi_coords)
        
        # Should return time x voxels matrix
        assert series.shape == (5, 3)
        
        # Check the patterns
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])  # Linear
        np.testing.assert_array_equal(series[:, 1], [2, 4, 6, 8, 10])  # Double
        np.testing.assert_array_equal(series[:, 2], [1, 4, 9, 16, 25])  # Square
    
    def test_series_roi_with_roi_vol(self):
        """Test series_roi with ROIVol input."""
        series = self.vec.series_roi(self.roi_vol)
        
        # Should return same result as with ROICoords
        assert series.shape == (5, 3)
        
        # Check the patterns
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])  # Linear
        np.testing.assert_array_equal(series[:, 1], [2, 4, 6, 8, 10])  # Double
        np.testing.assert_array_equal(series[:, 2], [1, 4, 9, 16, 25])  # Square
    
    def test_series_roi_single_voxel(self):
        """Test series_roi with single voxel ROI."""
        # Create single voxel ROI
        single_coords = np.array([[2, 3, 4]])
        single_roi = ROICoords(single_coords, NeuroSpace(dim=[10, 10, 10]))
        
        series = self.vec.series_roi(single_roi)
        
        assert series.shape == (5, 1)
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])
    
    def test_series_roi_sparse_neurovec(self):
        """Test series_roi with SparseNeuroVec."""
        # Create mask that includes our ROI voxels
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[2, 3, 4] = True
        mask_data[5, 6, 7] = True
        mask_data[8, 8, 8] = True
        mask_data[0, 0, 0] = True  # Add extra voxel
        
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=[10, 10, 10]))
        
        # Convert to sparse
        sparse_vec = self.vec.as_sparse(mask)
        
        # Extract series for ROI
        series = sparse_vec.series_roi(self.roi_coords)
        
        assert series.shape == (5, 3)
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])  # Linear
        np.testing.assert_array_equal(series[:, 1], [2, 4, 6, 8, 10])  # Double
        np.testing.assert_array_equal(series[:, 2], [1, 4, 9, 16, 25])  # Square
    
    def test_series_roi_empty_roi(self):
        """Test series_roi with empty ROI."""
        # Create empty ROI
        empty_coords = np.array([]).reshape(0, 3)
        empty_roi = ROICoords(empty_coords, NeuroSpace(dim=[10, 10, 10]))
        
        series = self.vec.series_roi(empty_roi)
        
        assert series.shape == (5, 0)
    
    def test_series_roi_invalid_input(self):
        """Test series_roi with invalid input."""
        with pytest.raises(TypeError):
            self.vec.series_roi("not an roi")
        
        with pytest.raises(TypeError):
            self.vec.series_roi(np.array([[1, 2, 3]]))  # Raw array not allowed
    
    def test_series_roi_outside_bounds(self):
        """Test series_roi with coordinates outside volume bounds."""
        # Create ROI with out-of-bounds coordinates
        bad_coords = np.array([[100, 100, 100], [2, 3, 4]])
        bad_roi = ROICoords(bad_coords, NeuroSpace(dim=[10, 10, 10]))
        
        # This should work but return zeros for out-of-bounds voxels
        series = self.vec.series_roi(bad_roi)
        
        assert series.shape == (5, 2)
        np.testing.assert_array_equal(series[:, 0], [0, 0, 0, 0, 0])  # Out of bounds
        np.testing.assert_array_equal(series[:, 1], [1, 2, 3, 4, 5])  # Valid voxel