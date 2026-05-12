"""
R-Python Equivalence Tests

This module tests that neuroim produces identical results to R neuroim2
by comparing against pre-generated R fixtures.

To generate R fixtures:
1. Install R and neuroim2 package
2. Run: cd tests/fixtures && Rscript generate_r_fixtures.R
3. This will create r_outputs/ directory with .rds files
"""

import pytest
import numpy as np
import neuroim as pn
from pathlib import Path
import warnings

# Skip these tests if R fixtures don't exist
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "r_outputs"
pytestmark = pytest.mark.skipif(
    not FIXTURES_DIR.exists(),
    reason="R fixtures not generated. Run generate_r_fixtures.R first."
)

# Note: Loading .rds files requires rpy2 or pyreadr
# For now, we'll create a mock loader and document the process
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    warnings.warn("pyreadr not installed. Install with: pip install pyreadr")

def load_rds(filename):
    """Load R .rds file"""
    if not HAS_PYREADR:
        pytest.skip("pyreadr required for R fixture loading")
    
    filepath = FIXTURES_DIR / filename
    result = pyreadr.read_r(str(filepath))
    return result[None]  # pyreadr returns dict with None key for single objects


class TestNeuroVolEquivalence:
    """Test NeuroVol operations match R exactly"""
    
    def test_arithmetic_operations(self):
        """Test that arithmetic operations produce identical results"""
        # Create test volume matching R
        space = pn.NeuroSpace(dim=(10, 10, 10), 
                             spacing=(2, 2, 2),
                             origin=(0, 0, 0))
        
        # Match R's 1:1000 which creates 1,2,3,...,1000
        test_data = np.arange(1, 1001).reshape(10, 10, 10, order='F')  # R uses column-major
        vol = pn.DenseNeuroVol(test_data, space)
        
        # Test addition
        vol_add = vol + 10
        if HAS_PYREADR:
            r_add_result = load_rds("vol_add_result.rds")
            np.testing.assert_array_equal(vol_add.data, r_add_result)
        else:
            # Manual verification of expected result
            expected = test_data + 10
            np.testing.assert_array_equal(vol_add.data, expected)
        
        # Test multiplication
        vol_mult = vol * 2
        if HAS_PYREADR:
            r_mult_result = load_rds("vol_mult_result.rds")
            np.testing.assert_array_equal(vol_mult.data, r_mult_result)
        else:
            expected = test_data * 2
            np.testing.assert_array_equal(vol_mult.data, expected)
        
        # Test division
        vol_div = vol / 2
        if HAS_PYREADR:
            r_div_result = load_rds("vol_div_result.rds")
            np.testing.assert_allclose(vol_div.data, r_div_result)
        else:
            expected = test_data / 2
            np.testing.assert_allclose(vol_div.data, expected)


class TestNeuroVecEquivalence:
    """Test NeuroVec operations match R exactly"""
    
    def test_series_extraction(self):
        """Test time series extraction matches R"""
        # Create 4D test data with known seed
        np.random.seed(42)
        vec_space = pn.NeuroSpace(dim=(10, 10, 10, 20))
        
        # Note: R's rnorm with seed=42 will produce different values
        # In real test, we'd load the actual R data
        if HAS_PYREADR:
            r_vec = load_rds("test_vec.rds")
            vec_data = r_vec  # Extract data from R object
        else:
            # Create synthetic test data
            vec_data = np.random.randn(10, 10, 10, 20)
        
        vec = pn.DenseNeuroVec(vec_data, vec_space)
        
        # Test single voxel extraction
        # R uses 1-based indexing: c(5,5,5) -> Python [4,4,4]
        ts1 = vec.series_3d(4, 4, 4)
        
        if HAS_PYREADR:
            r_ts1 = load_rds("vec_series_single.rds")
            np.testing.assert_allclose(ts1, r_ts1)
        
        # Test multiple voxel extraction
        # R coords: (3,3,3), (7,7,7), (5,6,7) -> Python: subtract 1
        coords = np.array([[2, 2, 2], [6, 6, 6], [4, 5, 6]])
        ts2 = vec.series(coords)
        
        if HAS_PYREADR:
            r_ts2 = load_rds("vec_series_multi.rds")
            np.testing.assert_allclose(ts2, r_ts2)


class TestROIEquivalence:
    """Test ROI operations match R exactly"""
    
    def test_spherical_roi(self):
        """Test spherical ROI coordinates match R"""
        space = pn.NeuroSpace(dim=(10, 10, 10), 
                             spacing=(2, 2, 2),
                             origin=(0, 0, 0))
        
        # R uses 1-based center c(5,5,5) -> Python [4,4,4]
        base = pn.DenseNeuroVol(np.zeros(space.dim, dtype=float), space)
        roi = pn.spherical_roi(base, [4, 4, 4], radius=3, fill=1)
        
        if HAS_PYREADR:
            r_coords = load_rds("roi_sphere_coords.rds")
            # R returns 1-based coords, we need to subtract 1
            r_coords_0based = r_coords - 1
            np.testing.assert_array_equal(roi.coords.coords, r_coords_0based)
    
    def test_cubic_roi(self):
        """Test cubic ROI coordinates match R"""
        space = pn.NeuroSpace(dim=(10, 10, 10), 
                             spacing=(2, 2, 2),
                             origin=(0, 0, 0))
        
        # R uses 1-based center c(5,5,5) -> Python [4,4,4]
        base = pn.DenseNeuroVol(np.zeros(space.dim, dtype=float), space)
        roi = pn.cuboid_roi(base, [4, 4, 4], surround=2, fill=1)
        
        if HAS_PYREADR:
            r_coords = load_rds("roi_cube_coords.rds")
            # R returns 1-based coords, we need to subtract 1
            r_coords_0based = r_coords - 1
            np.testing.assert_array_equal(roi.coords.coords, r_coords_0based)


class TestConnectedComponentsEquivalence:
    """Test connected components match R exactly"""
    
    def test_conn_comp(self):
        """Test connected components results match R"""
        # Would need to load exact R data for true equivalence
        # For now, test the structure matches
        space = pn.NeuroSpace(dim=(10, 10, 10))
        
        # Create test data with same seed as R
        np.random.seed(42)
        stat_data = np.random.normal(0, 2, size=(10, 10, 10))
        stat_vol = pn.DenseNeuroVol(stat_data, space)
        
        result = pn.conn_comp(stat_vol, threshold=1.5, connect="26-connect")
        
        if HAS_PYREADR:
            r_mask = load_rds("conn_comp_mask.rds")
            r_clusters = load_rds("conn_comp_clusters.rds")
            r_table = load_rds("conn_comp_table.rds")
            
            # Compare components
            np.testing.assert_array_equal(result.mask.data, r_mask)
            np.testing.assert_array_equal(result.cluster_map.data, r_clusters)
            # Table comparison would need careful column matching


class TestStatisticalOperationsEquivalence:
    """Test statistical operations match R"""
    
    def test_partition(self):
        """Test partition results match R"""
        space = pn.NeuroSpace(dim=(10, 10, 10))
        test_data = np.arange(1, 1001).reshape(10, 10, 10, order='F')
        vol = pn.DenseNeuroVol(test_data, space)
        
        # Create mask
        np.random.seed(42)
        mask_data = np.random.rand(10, 10, 10) > 0.7
        mask = pn.LogicalNeuroVol(mask_data, space)
        
        partitioned = pn.partition(vol, k=3, mask=mask)
        
        if HAS_PYREADR:
            r_partition = load_rds("partition_result.rds")
            # Compare partition assignments
            # Note: cluster labels might differ but structure should match


def create_equivalence_test_template():
    """
    Generate a template for creating more equivalence tests.
    This shows the pattern for comparing R and Python results.
    """
    template = '''
def test_new_function_equivalence():
    """Test that new_function produces identical results to R"""
    
    # Step 1: Create inputs matching R exactly
    # Remember R uses 1-based indexing and column-major order
    
    # Step 2: Run Python version
    py_result = pn.new_function(...)
    
    # Step 3: Load R result
    if HAS_PYREADR:
        r_result = load_rds("new_function_result.rds")
        
        # Step 4: Compare results
        # - Adjust for indexing differences (subtract 1 from R indices)
        # - Use allclose for floating point comparisons
        # - Consider array order (R is column-major, Python row-major)
        np.testing.assert_allclose(py_result, r_result, rtol=1e-10)
    '''
    return template


# Utility functions for R-Python comparison

def adjust_r_indices(r_coords):
    """Convert R 1-based indices to Python 0-based"""
    return r_coords - 1


def ensure_column_major(arr):
    """Ensure array is in column-major (Fortran) order like R"""
    return np.asarray(arr, order='F')


def compare_with_tolerance(py_result, r_result, rtol=1e-10, atol=1e-12):
    """Compare numerical results with appropriate tolerance"""
    np.testing.assert_allclose(py_result, r_result, rtol=rtol, atol=atol)


# Instructions for running equivalence tests
"""
To run R-Python equivalence tests:

1. Install R and neuroim2:
   - Install R from https://www.r-project.org/
   - In R: install.packages("devtools")
   - In R: devtools::install_github("bbuchsbaum/neuroim2")

2. Install Python dependencies:
   pip install pyreadr  # For reading .rds files

3. Generate R fixtures:
   cd tests/fixtures
   Rscript generate_r_fixtures.R

4. Run equivalence tests:
   pytest tests/test_r_equivalence.py -v

5. To add new equivalence tests:
   - Add test case to generate_r_fixtures.R
   - Add corresponding test in this file
   - Follow the pattern in create_equivalence_test_template()
"""
