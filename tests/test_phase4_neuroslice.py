"""Phase 4 Tests: 2D Slices (NeuroSlice class)

These tests are direct translations from R's neuroim2 tests to ensure
complete compatibility between the R and Python implementations.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from neuroimpy.neuro_space import NeuroSpace
from neuroimpy.neuro_vol import DenseNeuroVol
from neuroimpy.neuro_slice import NeuroSlice, neuroslice, slice, slices
from neuroimpy.axis import AxisSet2D, NamedAxis


class TestNeuroSliceConstruction:
    """Test NeuroSlice construction."""
    
    def test_neuroslice_constructor(self):
        """Test NeuroSlice constructor works correctly.
        
        R Equivalent
        ------------
        test_that("NeuroSlice constructor works correctly", {
            data <- matrix(rnorm(64 * 64), 64, 64)
            space <- NeuroSpace(c(64, 64), spacing = c(1, 1))
            nslice <- NeuroSlice(data, space)
            
            expect_s4_class(nslice, "NeuroSlice")
            expect_equal(nslice@space, space)
            expect_equal(nslice@.Data, data)
        })
        """
        data = np.random.randn(64, 64)
        space = NeuroSpace((64, 64), spacing=(1, 1))
        nslice = neuroslice(data, space)
        
        assert isinstance(nslice, NeuroSlice)
        assert nslice.space == space
        assert_array_equal(nslice.data, data)
    
    def test_neuroslice_from_vector(self):
        """Test NeuroSlice construction from 1D vector."""
        space = NeuroSpace((10, 10), spacing=(1, 1))
        data = np.random.randn(100)
        nslice = neuroslice(data, space)
        
        assert isinstance(nslice, NeuroSlice)
        assert nslice.shape == (10, 10)
        assert nslice.data.size == 100
    
    def test_neuroslice_sparse_construction(self):
        """Test sparse NeuroSlice construction."""
        space = NeuroSpace((10, 10), spacing=(1, 1))
        
        # Create sparse data
        n_points = 20
        sparse_data = np.random.randn(n_points)
        sparse_indices = np.random.choice(100, n_points, replace=False)
        
        nslice = neuroslice(sparse_data, space, indices=sparse_indices)
        
        assert isinstance(nslice, NeuroSlice)
        # Check that values were placed at correct indices
        flat_data = nslice.data.ravel(order='F')
        assert_array_almost_equal(flat_data[sparse_indices], sparse_data)


class TestNeuroSliceCoordinates:
    """Test coordinate transformations."""
    
    def test_grid_to_index(self):
        """Test grid_to_index methods work correctly.
        
        R Equivalent
        ------------
        test_that("grid_to_index methods work correctly", {
            data <- matrix(rnorm(64 * 64), 64, 64)
            space <- NeuroSpace(c(64, 64), spacing = c(1, 1))
            nslice <- NeuroSlice(data, space)
            
            coords <- matrix(c(1, 1, 64, 64), nrow=2, byrow=TRUE)
            idx <- grid_to_index(nslice, coords)
            
            expect_equal(idx, c(1, 4096))
            
            coords <- c(1, 1)
            idx <- grid_to_index(nslice, coords)
            
            expect_equal(idx, 1)
        })
        """
        data = np.random.randn(64, 64)
        space = NeuroSpace((64, 64), spacing=(1, 1))
        nslice = neuroslice(data, space)
        
        # Test matrix of coordinates
        coords = np.array([[0, 0], [63, 63]])  # Python 0-based
        idx = nslice.grid_to_index(coords)
        assert_array_equal(idx, [0, 4095])  # Python 0-based
        
        # Test single coordinate
        coords = np.array([0, 0])  # Python 0-based
        idx = nslice.grid_to_index(coords)
        assert idx == 0  # Python 0-based
    
    def test_index_to_grid(self):
        """Test index_to_grid method works correctly.
        
        R Equivalent
        ------------
        test_that("index_to_grid method works correctly", {
            data <- matrix(rnorm(64 * 64), 64, 64)
            space <- NeuroSpace(c(64, 64), spacing = c(1, 1))
            nslice <- NeuroSlice(data, space)
            
            idx <- c(1, 4096)
            coords <- index_to_grid(nslice, idx)
            
            expect_equal(coords, matrix(c(1, 1, 64, 64), nrow = 2, byrow = TRUE))
        })
        """
        data = np.random.randn(64, 64)
        space = NeuroSpace((64, 64), spacing=(1, 1))
        nslice = neuroslice(data, space)
        
        idx = np.array([0, 4095])  # Python 0-based
        coords = nslice.index_to_grid(idx)
        
        expected = np.array([[0, 0], [63, 63]])  # Python 0-based
        assert_array_equal(coords, expected)


class TestNeuroSliceExtraction:
    """Test slice extraction from volumes."""
    
    def test_slice_from_volume(self):
        """Test extracting slices from a volume."""
        # Create a test volume
        vol_data = np.random.randn(10, 12, 14)
        vol_space = NeuroSpace((10, 12, 14), spacing=(1, 2, 3))
        vol = DenseNeuroVol(vol_data, vol_space)
        
        # Extract slice along axis 2 (z-axis)
        z_slice = slice(vol, 5, 2)  # Python 0-based
        assert isinstance(z_slice, NeuroSlice)
        assert z_slice.shape == (10, 12)
        assert_array_equal(z_slice.data, vol_data[:, :, 5])
        assert_array_equal(z_slice.spacing, [1, 2])
        
        # Extract slice along axis 1 (y-axis)
        y_slice = slice(vol, 6, 1)  # Python 0-based
        assert isinstance(y_slice, NeuroSlice)
        assert y_slice.shape == (10, 14)
        assert_array_equal(y_slice.data, vol_data[:, 6, :])
        assert_array_equal(y_slice.spacing, [1, 3])
        
        # Extract slice along axis 0 (x-axis)
        x_slice = slice(vol, 3, 0)  # Python 0-based
        assert isinstance(x_slice, NeuroSlice)
        assert x_slice.shape == (12, 14)
        assert_array_equal(x_slice.data, vol_data[3, :, :])
        assert_array_equal(x_slice.spacing, [2, 3])
    
    def test_slices_method(self):
        """Test extracting all slices from a volume."""
        vol_data = np.random.randn(8, 10, 12)
        vol_space = NeuroSpace((8, 10, 12))
        vol = DenseNeuroVol(vol_data, vol_space)
        
        # Get all z-slices
        all_slices = slices(vol)
        
        assert len(all_slices) == 12
        for i, s in enumerate(all_slices):
            assert isinstance(s, NeuroSlice)
            assert s.shape == (8, 10)
            assert_array_equal(s.data, vol_data[:, :, i])


class TestNeuroSliceOperations:
    """Test operations on NeuroSlice."""
    
    def test_arithmetic_operations(self):
        """Test arithmetic operations on slices."""
        data1 = np.random.randn(10, 10)
        data2 = np.random.randn(10, 10)
        space = NeuroSpace((10, 10))
        
        slice1 = neuroslice(data1, space)
        slice2 = neuroslice(data2, space)
        
        # Slice + Slice
        result = slice1 + slice2
        assert isinstance(result, NeuroSlice)
        assert_array_almost_equal(result.data, data1 + data2)
        
        # Slice - Slice
        result = slice1 - slice2
        assert_array_almost_equal(result.data, data1 - data2)
        
        # Slice * scalar
        result = slice1 * 2
        assert_array_almost_equal(result.data, data1 * 2)
        
        # Slice / scalar
        result = slice1 / 2
        assert_array_almost_equal(result.data, data1 / 2)

    def test_reverse_scalar_arithmetic(self):
        """Test reverse arithmetic for scalar-on-left operations."""
        data = np.random.randn(10, 10)
        space = NeuroSpace((10, 10))
        nslice = neuroslice(data, space)

        assert_array_almost_equal((2 + nslice).data, 2 + data)
        assert_array_almost_equal((3 * nslice).data, 3 * data)
        assert_array_almost_equal((5 - nslice).data, 5 - data)
        assert_array_almost_equal((100 / nslice).data, 100 / data)
    
    def test_indexing(self):
        """Test indexing operations."""
        data = np.arange(100).reshape(10, 10, order='F')
        space = NeuroSpace((10, 10))
        nslice = neuroslice(data, space)
        
        # Single element
        assert nslice[0, 0] == 0
        assert nslice[9, 9] == 99
        
        # Row
        assert_array_equal(nslice[0, :], data[0, :])
        
        # Column
        assert_array_equal(nslice[:, 0], data[:, 0])
        
        # Assignment
        nslice[5, 5] = 999
        assert nslice[5, 5] == 999
    
    def test_values_method(self):
        """Test values method returns flattened array."""
        data = np.arange(100).reshape(10, 10, order='F')
        space = NeuroSpace((10, 10))
        nslice = neuroslice(data, space)
        
        vals = nslice.values()
        assert vals.shape == (100,)
        assert_array_equal(vals, data.ravel(order='F'))


class TestNeuroSliceProperties:
    """Test NeuroSlice properties."""
    
    def test_properties(self):
        """Test slice properties."""
        data = np.random.randn(20, 30)
        space = NeuroSpace((20, 30), spacing=(2, 3), origin=(10, 20))
        nslice = neuroslice(data, space)
        
        # Test shape
        assert nslice.shape == (20, 30)
        
        # Test dim
        assert_array_equal(nslice.dim, [20, 30])
        
        # Test spacing
        assert_array_equal(nslice.spacing, [2, 3])
        
        # Test origin
        assert_array_equal(nslice.origin, [10, 20])
        
        # Test axes
        assert nslice.axes.ndim == 2


# R-Python compatibility notes
def test_phase4_compatibility_notes():
    """Document key differences for Phase 4."""
    notes = """
    Phase 4 R-Python Compatibility Notes:
    
    1. Indexing remains 0-based in Python vs 1-based in R
    2. slice() function uses 0-based indices in Python
    3. Fortran-order (column-major) array storage maintained
    4. grid_to_index and index_to_grid use 0-based indices
    5. Factory function is neuroslice() in Python vs NeuroSlice() in R
    """
    print(notes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
