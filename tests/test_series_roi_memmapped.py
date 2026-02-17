"""Tests for series_roi with memory-mapped NeuroVec classes."""

import pytest
import numpy as np
import tempfile
import os
from neuroimpy import (
    NeuroSpace, ROICoords, ROIVol,
    BigNeuroVec, FileBackedNeuroVec, MappedNeuroVec,
    DenseNeuroVec, DenseNeuroVol, write_vol
)


class TestSeriesROIMemMapped:
    """Test series_roi method for memory-mapped NeuroVec classes."""
    
    def setup_method(self):
        """Set up test data."""
        # Create 4D space
        self.space_4d = NeuroSpace(dim=[10, 10, 10, 5])
        
        # Create test data with known pattern
        self.data = np.zeros((10, 10, 10, 5))
        # Fill specific voxels with distinct patterns
        for t in range(5):
            self.data[2, 3, 4, t] = t + 1  # Linear pattern
            self.data[5, 6, 7, t] = (t + 1) * 2  # Double pattern
            self.data[8, 8, 8, t] = (t + 1) ** 2  # Square pattern
        
        # Create ROI with the three voxels
        roi_coords = np.array([[2, 3, 4], [5, 6, 7], [8, 8, 8]])
        self.roi_coords = ROICoords(roi_coords, NeuroSpace(dim=[10, 10, 10]))
        
        # Create ROIVol with same coordinates
        roi_data = np.array([1.0, 2.0, 3.0])
        self.roi_vol = ROIVol(roi_data, NeuroSpace(dim=[10, 10, 10]), roi_coords)
    
    def test_series_roi_big_neurovec(self):
        """Test series_roi with BigNeuroVec."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp:
            tmp_name = tmp.name
        
        try:
            # Create BigNeuroVec
            vec = BigNeuroVec(self.data, self.space_4d, filename=tmp_name)
            
            # Test with ROICoords
            series = vec.series_roi(self.roi_coords)
            assert series.shape == (5, 3)
            np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])
            np.testing.assert_array_equal(series[:, 1], [2, 4, 6, 8, 10])
            np.testing.assert_array_equal(series[:, 2], [1, 4, 9, 16, 25])
            
            # Test with ROIVol
            series = vec.series_roi(self.roi_vol)
            assert series.shape == (5, 3)
            np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])
            
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    
    def test_series_roi_file_backed_neurovec(self):
        """Test series_roi with FileBackedNeuroVec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save volumes to NIfTI files
            filenames = []
            for t in range(5):
                vol_data = self.data[..., t]
                vol_space = NeuroSpace(dim=[10, 10, 10])
                vol = DenseNeuroVol(vol_data, vol_space)
                filename = os.path.join(tmpdir, f"vol_{t:03d}.nii")
                write_vol(vol, filename)
                filenames.append(filename)

            vec = FileBackedNeuroVec(filenames)

            # Test with ROICoords
            series = vec.series_roi(self.roi_coords)
            assert series.shape == (5, 3)
            np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])
            np.testing.assert_array_equal(series[:, 1], [2, 4, 6, 8, 10])
            np.testing.assert_array_equal(series[:, 2], [1, 4, 9, 16, 25])
            
            # Test with ROIVol
            series = vec.series_roi(self.roi_vol)
            assert series.shape == (5, 3)
            np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])
    
    def test_series_roi_mapped_neurovec(self):
        """Test series_roi with MappedNeuroVec."""
        # Create DenseNeuroVec first
        base_vec = DenseNeuroVec(self.data, self.space_4d)
        
        # Create MappedNeuroVec with identity transform
        def transform(vol):
            return vol  # Identity transform
        
        vec = MappedNeuroVec(base_vec, transform)
        
        # Test with ROICoords
        series = vec.series_roi(self.roi_coords)
        assert series.shape == (5, 3)
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(series[:, 1], [2, 4, 6, 8, 10])
        np.testing.assert_array_equal(series[:, 2], [1, 4, 9, 16, 25])
        
        # Test with ROIVol
        series = vec.series_roi(self.roi_vol)
        assert series.shape == (5, 3)
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 5])
    
    def test_series_roi_mapped_neurovec_with_transform(self):
        """Test series_roi with MappedNeuroVec with a transform."""
        # Create DenseNeuroVec
        base_vec = DenseNeuroVec(self.data, self.space_4d)
        
        # Create MappedNeuroVec with scaling transform
        def scale_by_2(vol):
            return vol * 2
        
        vec = MappedNeuroVec(base_vec, scale_by_2)
        
        # Test with ROICoords - values should be doubled
        series = vec.series_roi(self.roi_coords)
        assert series.shape == (5, 3)
        np.testing.assert_array_equal(series[:, 0], [2, 4, 6, 8, 10])
        np.testing.assert_array_equal(series[:, 1], [4, 8, 12, 16, 20])
        np.testing.assert_array_equal(series[:, 2], [2, 8, 18, 32, 50])
