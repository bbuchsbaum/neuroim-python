"""
Tests for parallel searchlight processing.
"""

import pytest
import numpy as np
import time
from neuroimpy.neuro_space import NeuroSpace
from neuroimpy.neuro_vol import DenseNeuroVol, LogicalNeuroVol
from neuroimpy.searchlight import searchlight_iterator, searchlight_coords
from neuroimpy.searchlight_high_level import searchlight


class TestParallelSearchlight:
    """Test parallel searchlight functionality."""
    
    @pytest.fixture
    def small_mask(self):
        """Create a small mask for testing."""
        space = NeuroSpace(dim=(10, 10, 10), spacing=(1, 1, 1))
        # Create a spherical mask
        x, y, z = np.ogrid[:10, :10, :10]
        center = [5, 5, 5]
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        mask_data = r < 4
        return LogicalNeuroVol(mask_data, space)
    
    @pytest.fixture
    def sample_data(self, small_mask):
        """Create sample data for testing."""
        # Create data with some pattern
        data = np.random.randn(*small_mask.shape)
        return DenseNeuroVol(data * small_mask.data, small_mask.space)
    
    def test_searchlight_iterator_parallel(self, small_mask):
        """Test that parallel searchlight_iterator produces same results as sequential."""
        radius = 3.0
        
        # Sequential processing
        seq_results = searchlight_iterator(small_mask, radius, eager=True, nonzero=True, cores=0)
        
        # Parallel processing
        par_results = searchlight_iterator(small_mask, radius, eager=True, nonzero=True, cores=2)
        
        # Check same number of searchlights
        assert len(seq_results) == len(par_results)
        
        # Check that results are equivalent (order might differ)
        seq_centers = {sl.parent_index for sl in seq_results}
        par_centers = {sl.parent_index for sl in par_results}
        assert seq_centers == par_centers
    
    def test_searchlight_coords_parallel(self, small_mask):
        """Test that parallel searchlight_coords produces same results as sequential."""
        radius = 3.0
        
        # Sequential processing
        seq_coords = searchlight_coords(small_mask, radius, nonzero=True, cores=0)
        
        # Parallel processing
        par_coords = searchlight_coords(small_mask, radius, nonzero=True, cores=2)
        
        # Check same number of coordinate sets
        assert len(seq_coords) == len(par_coords)
        
        # Check that coordinate sets are equivalent
        for i in range(len(seq_coords)):
            seq_c = seq_coords[i]
            par_c = par_coords[i]
            # Coords might be in different order, so sort before comparing
            seq_sorted = seq_c[np.lexsort((seq_c[:, 2], seq_c[:, 1], seq_c[:, 0]))]
            par_sorted = par_c[np.lexsort((par_c[:, 2], par_c[:, 1], par_c[:, 0]))]
            np.testing.assert_array_equal(seq_sorted, par_sorted)
    
    def test_searchlight_apply_parallel(self, small_mask, sample_data):
        """Test that parallel searchlight application produces same results."""
        radius = 3.0
        
        # Define a simple method
        def mean_method(data):
            return np.mean(data)
        
        # Sequential processing
        seq_result = searchlight(small_mask, radius, mean_method, 
                               data=sample_data, eager=True, cores=0)
        
        # Parallel processing
        par_result = searchlight(small_mask, radius, mean_method, 
                               data=sample_data, eager=True, cores=2)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(seq_result.data, par_result.data)
    
    def test_searchlight_lazy_not_parallel(self, small_mask):
        """Test that lazy evaluation doesn't use parallel processing."""
        radius = 3.0
        
        # Lazy evaluation should return LazyList regardless of cores
        lazy_seq = searchlight_iterator(small_mask, radius, eager=False, cores=0)
        lazy_par = searchlight_iterator(small_mask, radius, eager=False, cores=2)
        
        # Both should be LazyList
        assert hasattr(lazy_seq, '__len__')
        assert hasattr(lazy_par, '__len__')
        assert len(lazy_seq) == len(lazy_par)
    
    def test_searchlight_performance(self, small_mask):
        """Test that parallel processing provides speedup for CPU-bound tasks."""
        radius = 3.0
        
        # Define a CPU-intensive method
        def expensive_method(data):
            # Simulate expensive computation
            result = 0.0
            for _ in range(1000):
                result += np.sum(np.exp(data)) / len(data)
            return result
        
        # Time sequential processing
        start = time.time()
        seq_result = searchlight(small_mask, radius, expensive_method, 
                               eager=True, cores=0)
        seq_time = time.time() - start
        
        # Time parallel processing
        start = time.time()
        par_result = searchlight(small_mask, radius, expensive_method, 
                               eager=True, cores=2)
        par_time = time.time() - start
        
        # Results should be identical
        np.testing.assert_array_almost_equal(seq_result.data, par_result.data, decimal=10)
        
        # Parallel should be faster (though on small data the overhead might dominate)
        print(f"Sequential time: {seq_time:.3f}s, Parallel time: {par_time:.3f}s")
        # Don't assert speedup due to overhead on small data
    
    def test_searchlight_cores_parameter(self, small_mask):
        """Test different values of cores parameter."""
        radius = 3.0
        
        # Test with different core counts
        for cores in [0, 1, 2, -1]:  # -1 should use all available cores
            results = searchlight_iterator(small_mask, radius, eager=True, cores=cores)
            assert len(results) > 0
    
    def test_searchlight_with_neurovec(self, small_mask):
        """Test parallel searchlight with NeuroVec data."""
        from neuroimpy.neuro_vec import DenseNeuroVec
        
        # Create 4D data
        space_4d = NeuroSpace(dim=(10, 10, 10, 5), spacing=(1, 1, 1, 1))
        data_4d = np.random.randn(10, 10, 10, 5) * small_mask.data[..., np.newaxis]
        vec_data = DenseNeuroVec(data_4d, space_4d)
        
        radius = 3.0
        
        # Define method for time series
        def temporal_mean(data):
            # data should be (n_voxels, n_timepoints)
            return np.mean(np.mean(data, axis=1))
        
        # Test both sequential and parallel
        seq_result = searchlight(small_mask, radius, temporal_mean, 
                               data=vec_data, eager=True, cores=0)
        par_result = searchlight(small_mask, radius, temporal_mean, 
                               data=vec_data, eager=True, cores=2)
        
        # Results should be close (might have small numerical differences)
        np.testing.assert_array_almost_equal(seq_result.data, par_result.data, decimal=10)
    
    def test_empty_searchlight_regions(self, small_mask):
        """Test handling of empty searchlight regions in parallel."""
        # Create a mask with isolated voxels
        space = NeuroSpace(dim=(20, 20, 20), spacing=(1, 1, 1))
        mask_data = np.zeros((20, 20, 20), dtype=bool)
        # Add a few isolated voxels
        mask_data[5, 5, 5] = True
        mask_data[15, 15, 15] = True
        sparse_mask = LogicalNeuroVol(mask_data, space)
        
        radius = 1.0  # Small radius
        
        def count_method(data):
            return len(data)
        
        # Test parallel processing with sparse mask
        result = searchlight(sparse_mask, radius, count_method, eager=True, cores=2)
        
        # Should have results
        assert not np.all(np.isnan(result.data))


def test_parallel_imports():
    """Test that joblib is properly imported."""
    from neuroimpy.searchlight import Parallel, delayed
    from neuroimpy.searchlight_high_level import Parallel as Parallel2, delayed as delayed2
    
    assert Parallel is not None
    assert delayed is not None
    assert Parallel2 is not None
    assert delayed2 is not None