"""
R-Python Equivalence Tests using NumPy fixtures

This module tests that neuroimpy produces identical results to R neuroim2
using pre-converted NumPy fixtures.
"""

import pytest
import numpy as np
import neuroimpy as pn
from neuroimpy.neuro_vol import DenseNeuroVol, LogicalNeuroVol
from neuroimpy.neuro_vec import DenseNeuroVec
from neuroimpy import SparseNeuroVec
from neuroimpy.roi import spherical_roi, cuboid_roi
from neuroimpy.stats import partition
from neuroimpy.connected_components import conn_comp
from pathlib import Path

# Check if fixtures exist
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "numpy_fixtures"

# Create fixtures if they don't exist
if not FIXTURES_DIR.exists():
    import subprocess
    fixtures_script = Path(__file__).parent / "fixtures" / "convert_r_to_numpy.py"
    subprocess.run(["python", str(fixtures_script)], check=True)


class TestNeuroVolEquivalence:
    """Test NeuroVol operations match R exactly"""
    
    def test_arithmetic_operations(self):
        """Test that arithmetic operations produce identical results to R"""
        # Create test volume matching R's 1:1000
        space = pn.NeuroSpace(dim=(10, 10, 10), 
                             spacing=(2, 2, 2),
                             origin=(0, 0, 0))
        
        # R's 1:1000 creates array with values 1,2,3,...,1000
        # Using column-major (Fortran) order to match R
        test_data = np.arange(1, 1001).reshape(10, 10, 10, order='F')
        vol = DenseNeuroVol(test_data, space)
        
        # Test addition: vol + 10
        vol_add = vol + 10
        expected = np.load(FIXTURES_DIR / "vol_add_result.npy")
        np.testing.assert_array_equal(vol_add.data, expected,
                                     "Addition result differs from R")
        
        # Test multiplication: vol * 2
        vol_mult = vol * 2
        expected = np.load(FIXTURES_DIR / "vol_mult_result.npy")
        np.testing.assert_array_equal(vol_mult.data, expected,
                                     "Multiplication result differs from R")
        
        # Test division: vol / 2
        vol_div = vol / 2
        expected = np.load(FIXTURES_DIR / "vol_div_result.npy")
        np.testing.assert_allclose(vol_div.data, expected, rtol=1e-10,
                                  err_msg="Division result differs from R")
    
    def test_indexing_equivalence(self):
        """Test that indexing behavior matches R (with 0-based adjustment)"""
        space = pn.NeuroSpace(dim=(10, 10, 10))
        test_data = np.arange(1, 1001).reshape(10, 10, 10, order='F')
        vol = DenseNeuroVol(test_data, space)
        
        # R: vol[5, 5, 5] = 445 (in 1-based indexing)
        # Python: vol[4, 4, 4] should give same value
        assert vol[4, 4, 4] == 445, "Single voxel indexing differs from R"
        
        # R: vol[1, 1, 1] = 1 (first element)
        # Python: vol[0, 0, 0] = 1
        assert vol[0, 0, 0] == 1, "First element differs from R"
        
        # R: vol[10, 10, 10] = 1000 (last element)
        # Python: vol[9, 9, 9] = 1000
        assert vol[9, 9, 9] == 1000, "Last element differs from R"


class TestNeuroVecEquivalence:
    """Test NeuroVec operations match R exactly"""
    
    @pytest.fixture
    def test_vec(self):
        """Create a test NeuroVec with known structure"""
        vec_space = pn.NeuroSpace(dim=(10, 10, 10, 20))
        # Create predictable test data
        np.random.seed(42)
        vec_data = np.random.randn(10, 10, 10, 20)
        return DenseNeuroVec(vec_data, vec_space)
    
    def test_series_extraction_single(self, test_vec):
        """Test single voxel time series extraction"""
        # R: series(vec, c(5, 5, 5)) with 1-based indexing
        # Python: series_3d(4, 4, 4) with 0-based indexing
        ts = test_vec.series_3d(4, 4, 4)
        
        if (FIXTURES_DIR / "vec_series_single.npy").exists():
            expected = np.load(FIXTURES_DIR / "vec_series_single.npy")
            np.testing.assert_allclose(ts, expected, rtol=1e-10,
                                      err_msg="Single voxel series differs from R")
        else:
            # Verify shape at minimum
            assert ts.shape == (20,), "Time series shape incorrect"
    
    def test_series_extraction_multiple(self, test_vec):
        """Test multiple voxel time series extraction"""
        # R coords (1-based): (3,3,3), (7,7,7), (5,6,7)
        # Python coords (0-based): (2,2,2), (6,6,6), (4,5,6)
        coords = np.array([[2, 2, 2], [6, 6, 6], [4, 5, 6]])
        ts = test_vec.series(coords)
        
        if (FIXTURES_DIR / "vec_series_multi.npy").exists():
            expected = np.load(FIXTURES_DIR / "vec_series_multi.npy")
            np.testing.assert_allclose(ts, expected, rtol=1e-10,
                                      err_msg="Multiple voxel series differs from R")
        else:
            # Verify shape: should be (n_timepoints, n_voxels)
            assert ts.shape == (20, 3), "Multi-voxel series shape incorrect"


class TestROIEquivalence:
    """Test ROI operations match R exactly"""
    
    def test_spherical_roi_structure(self):
        """Test spherical ROI produces correct structure"""
        space = pn.NeuroSpace(dim=(10, 10, 10))
        
        # Create a test volume for ROI extraction
        vol = DenseNeuroVol(np.ones((10, 10, 10)), space)
        
        # R: spherical_roi(c(5, 5, 5), radius = 3, space)
        # Python: spherical_roi(vol, [4, 4, 4], radius=3)
        roi = spherical_roi(vol, centroid=[4, 4, 4], radius=3)
        
        if (FIXTURES_DIR / "roi_sphere_coords.npy").exists():
            expected = np.load(FIXTURES_DIR / "roi_sphere_coords.npy")
            # Note: R returns 1-based coords, fixture should be 0-based
            np.testing.assert_array_equal(roi.coords, expected,
                                         "Spherical ROI coords differ from R")
        else:
            # At minimum, check reasonable number of voxels
            n_voxels = len(roi)
            assert 50 < n_voxels < 200, f"Unexpected number of voxels: {n_voxels}"
    
    def test_cubic_roi_structure(self):
        """Test cubic ROI produces correct structure"""
        space = pn.NeuroSpace(dim=(10, 10, 10))
        
        # Create a test volume for ROI extraction
        vol = DenseNeuroVol(np.ones((10, 10, 10)), space)
        
        # R: cubic_roi(c(5, 5, 5), surround = 2, space)
        # Python: cuboid_roi(vol, [4, 4, 4], surround=2)
        roi = cuboid_roi(vol, centroid=[4, 4, 4], surround=2)
        
        if (FIXTURES_DIR / "roi_cube_coords.npy").exists():
            expected = np.load(FIXTURES_DIR / "roi_cube_coords.npy")
            np.testing.assert_array_equal(roi.coords, expected,
                                         "Cubic ROI coords differ from R")
        else:
            # Should be a 5x5x5 cube = 125 voxels
            assert len(roi) == 125, "Cubic ROI size incorrect"


class TestSparseOperations:
    """Test sparse data structures match R behavior"""
    
    def test_sparse_neurovec_creation(self):
        """Test creating sparse neurovec matches R structure"""
        # Create mask (30% of voxels active)
        np.random.seed(42)
        mask_data = np.random.rand(10, 10, 10) > 0.7
        space = pn.NeuroSpace(dim=(10, 10, 10))
        mask = LogicalNeuroVol(mask_data, space)
        
        # Create sparse data (time x voxels)
        n_active = mask_data.sum()
        sparse_data = np.random.randn(20, n_active)  # time x voxels
        vec_space = pn.NeuroSpace(dim=(10, 10, 10, 20))
        
        sparse_vec = SparseNeuroVec(data=sparse_data, 
                                      mask=mask,
                                      space=vec_space)
        
        # Test basic properties
        assert sparse_vec.data.shape == (20, n_active)  # time x voxels
        assert np.array_equal(sparse_vec.mask.data, mask_data)
    
    def test_sparse_series_extraction(self):
        """Test series extraction from sparse neurovec"""
        # This would compare against R sparse series extraction
        pass  # Implement when R fixtures available


class TestStatisticalOperations:
    """Test statistical operations match R"""
    
    def test_partition_structure(self):
        """Test partition operation structure"""
        space = pn.NeuroSpace(dim=(10, 10, 10))
        test_data = np.arange(1, 1001).reshape(10, 10, 10, order='F')
        vol = DenseNeuroVol(test_data, space)
        
        # Create mask
        mask_data = np.ones((10, 10, 10), dtype=bool)
        mask_data[0:2, :, :] = False  # Exclude some voxels
        mask = LogicalNeuroVol(mask_data, space)
        
        # Partition into 3 clusters
        result = partition(vol, k=3, mask=mask)
        
        # Verify structure
        assert isinstance(result, pn.ClusteredNeuroVol)
        assert result.num_clusters() == 3
        assert result.mask.data.sum() == mask_data.sum()


def test_column_major_consistency():
    """Test that column-major ordering is handled correctly"""
    # R uses column-major, Python uses row-major by default
    # This test ensures we handle the conversion properly
    
    # Create data both ways
    row_major = np.arange(8).reshape(2, 2, 2, order='C')
    col_major = np.arange(8).reshape(2, 2, 2, order='F')
    
    # In column-major (R style), linear index 0,1,2,3... fills:
    # [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]
    assert col_major[0, 0, 0] == 0
    assert col_major[1, 0, 0] == 1
    assert col_major[0, 1, 0] == 2
    assert col_major[1, 1, 0] == 3
    
    # This is different from row-major
    assert row_major[0, 0, 0] == 0
    assert row_major[0, 0, 1] == 1
    assert row_major[0, 1, 0] == 2
    assert row_major[0, 1, 1] == 3


# Test utilities for R equivalence

def r_to_python_indices(r_indices):
    """Convert R 1-based indices to Python 0-based"""
    return np.asarray(r_indices) - 1


def python_to_r_indices(py_indices):
    """Convert Python 0-based indices to R 1-based"""
    return np.asarray(py_indices) + 1


def assert_equivalent_with_indexing(py_result, r_result, index_cols=None):
    """
    Compare Python and R results, adjusting for indexing differences.
    
    Parameters
    ----------
    py_result : array-like
        Python result
    r_result : array-like
        R result (may have 1-based indices)
    index_cols : list of int, optional
        Column indices that contain indices needing adjustment
    """
    py_result = np.asarray(py_result)
    r_result = np.asarray(r_result)
    
    if index_cols is not None:
        # Adjust specified columns from 1-based to 0-based
        r_adjusted = r_result.copy()
        for col in index_cols:
            r_adjusted[:, col] = r_result[:, col] - 1
        np.testing.assert_array_equal(py_result, r_adjusted)
    else:
        np.testing.assert_allclose(py_result, r_result, rtol=1e-10)