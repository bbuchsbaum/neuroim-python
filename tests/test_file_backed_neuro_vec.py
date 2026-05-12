"""Comprehensive tests for FileBackedNeuroVec."""

import pytest
import numpy as np
import tempfile
import os
from neuroim import NeuroSpace, DenseNeuroVol, FileBackedNeuroVec
from neuroim.file_backed_neuro_vec import file_backed_neurovec
from neuroim.io import write_vol


@pytest.fixture
def temp_vol_files():
    """Create temporary volume files for testing."""
    temp_files = []
    temp_dir = tempfile.mkdtemp()
    
    # Create 5 volumes with different patterns
    shape = (10, 10, 10)
    space = NeuroSpace(dim=shape)
    
    for i in range(5):
        # Create volume with a unique pattern
        data = np.ones(shape) * (i + 1)
        # Add some variation
        data[i:i+2, i:i+2, i:i+2] = (i + 1) * 10
        
        vol = DenseNeuroVol(data, space)  # Let it use default float64
        
        # Save to temporary file
        filename = os.path.join(temp_dir, f"vol_{i:03d}.nii.gz")
        write_vol(vol, filename)
        temp_files.append(filename)
    
    yield temp_files, shape, space
    
    # Cleanup
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    os.rmdir(temp_dir)


class TestFileBackedNeuroVecCreation:
    """Test FileBackedNeuroVec creation and initialization."""
    
    def test_creation_basic(self, temp_vol_files):
        """Test basic creation."""
        files, shape, space = temp_vol_files
        
        fb_vec = FileBackedNeuroVec(files)
        
        assert fb_vec.n_volumes == 5
        assert fb_vec.vol_shape == shape
        assert fb_vec.shape == shape + (5,)
        assert fb_vec.cache_size == 1
        assert fb_vec.dtype == np.float64  # Default numpy dtype
    
    def test_creation_with_space(self, temp_vol_files):
        """Test creation with explicit space."""
        files, shape, vol_space = temp_vol_files
        
        # Create 4D space
        from neuroim.axis import AxisSet4D, NamedAxis
        time_axis = NamedAxis("t", 1)
        axes_4d = AxisSet4D(vol_space.axes.i, vol_space.axes.j, vol_space.axes.k, time_axis)
        
        space_4d = NeuroSpace(
            dim=[10, 10, 10, 5],
            spacing=[1.0, 1.0, 1.0, 1.0],
            origin=[0.0, 0.0, 0.0, 0.0],
            axes=axes_4d
        )
        
        fb_vec = FileBackedNeuroVec(files, space_4d)
        
        assert fb_vec.space == space_4d
        assert fb_vec.shape == (10, 10, 10, 5)
    
    def test_creation_with_cache_size(self, temp_vol_files):
        """Test creation with custom cache size."""
        files, _, _ = temp_vol_files
        
        fb_vec = FileBackedNeuroVec(files, cache_size=3)
        
        assert fb_vec.cache_size == 3
    
    def test_creation_empty_list_error(self):
        """Test that empty file list raises error."""
        with pytest.raises(ValueError, match="filenames list cannot be empty"):
            FileBackedNeuroVec([])
    
    def test_factory_function(self, temp_vol_files):
        """Test the factory function."""
        files, _, _ = temp_vol_files
        
        fb_vec = file_backed_neurovec(files)
        
        assert isinstance(fb_vec, FileBackedNeuroVec)
        assert fb_vec.n_volumes == 5


class TestFileBackedNeuroVecIndexing:
    """Test FileBackedNeuroVec indexing operations."""
    
    def test_getitem_single_volume(self, temp_vol_files):
        """Test accessing a single volume."""
        files, shape, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Get first volume
        vol0 = fb_vec[:, :, :, 0]
        assert vol0.shape == shape
        assert np.all(vol0[0:2, 0:2, 0:2] == 10)
        
        # Get last volume
        vol4 = fb_vec[:, :, :, 4]
        assert np.all(vol4[4:6, 4:6, 4:6] == 50)
    
    def test_getitem_slice(self, temp_vol_files):
        """Test slicing volumes."""
        files, shape, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Get slice of volumes
        vols = fb_vec[:, :, :, 1:4]
        assert vols.shape == shape + (3,)
        assert np.all(vols[1:3, 1:3, 1:3, 0] == 20)
        assert np.all(vols[3:5, 3:5, 3:5, 2] == 40)
    
    def test_getitem_4d_single_time(self, temp_vol_files):
        """Test 4D indexing with single time point."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Get a single voxel at time 0
        val = fb_vec[5, 5, 5, 0]
        assert val == 1.0
        
        # Get a slice at time 2
        slice_data = fb_vec[2:4, 2:4, 2:4, 2]
        assert np.all(slice_data == 30)
    
    def test_getitem_4d_time_slice(self, temp_vol_files):
        """Test 4D indexing with time slice."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Get multiple time points for a voxel
        vals = fb_vec[5, 5, 5, 1:4]
        assert vals.shape == (3,)
        np.testing.assert_array_equal(vals, [2, 3, 4])
        
        # Get multiple time points for a region
        region = fb_vec[0:2, 0:2, 0:2, 0:3]
        assert region.shape == (2, 2, 2, 3)
        # First time point has pattern at [0:2, 0:2, 0:2]
        assert np.all(region[:, :, :, 0] == 10)
        # Second time point has pattern at [1:3, 1:3, 1:3], so only partial overlap
        assert region[0, 0, 0, 1] == 2  # Outside pattern
        assert region[1, 1, 1, 1] == 20  # Inside pattern
    
    def test_getitem_invalid_index(self, temp_vol_files):
        """Test invalid indexing."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Index out of range
        with pytest.raises(IndexError, match="out of range"):
            fb_vec[10]  # Time index out of range
        
        # Wrong number of indices
        with pytest.raises(ValueError, match="Expected 4 indices"):
            fb_vec[0, 1, 2]
        
        # Unsupported index type
        with pytest.raises(TypeError, match="Unsupported index type"):
            fb_vec["invalid"]
    
    def test_setitem_not_supported(self, temp_vol_files):
        """Test that setting values is not supported."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        with pytest.raises(TypeError, match="Cannot modify"):
            fb_vec[:, :, :, 0] = np.zeros((10, 10, 10))


class TestFileBackedNeuroVecCache:
    """Test FileBackedNeuroVec caching behavior."""
    
    def test_cache_basic(self, temp_vol_files):
        """Test basic caching."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files, cache_size=2)
        
        # Access volumes
        vol0 = fb_vec[:, :, :, 0]
        assert 0 in fb_vec._cache
        
        vol1 = fb_vec[:, :, :, 1]
        assert 1 in fb_vec._cache
        assert len(fb_vec._cache) == 2
        
        # Access third volume - should evict first
        vol2 = fb_vec[:, :, :, 2]
        assert 2 in fb_vec._cache
        assert 0 not in fb_vec._cache
        assert len(fb_vec._cache) == 2
    
    def test_cache_lru(self, temp_vol_files):
        """Test LRU cache behavior."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files, cache_size=2)
        
        # Access volumes 0 and 1
        fb_vec[:, :, :, 0]
        fb_vec[:, :, :, 1]
        
        # Access 0 again (makes it most recent)
        fb_vec[:, :, :, 0]
        
        # Access 2 - should evict 1 (least recent)
        fb_vec[:, :, :, 2]
        assert 0 in fb_vec._cache
        assert 2 in fb_vec._cache
        assert 1 not in fb_vec._cache
    
    def test_clear_cache(self, temp_vol_files):
        """Test cache clearing."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files, cache_size=3)
        
        # Fill cache
        fb_vec[:, :, :, 0]
        fb_vec[:, :, :, 1]
        fb_vec[:, :, :, 2]
        assert len(fb_vec._cache) == 3
        
        # Clear cache
        fb_vec.clear_cache()
        assert len(fb_vec._cache) == 0
        assert len(fb_vec._cache_order) == 0


class TestFileBackedNeuroVecProperties:
    """Test FileBackedNeuroVec properties and data access."""
    
    def test_data_property(self, temp_vol_files):
        """Test data property loads all volumes."""
        files, shape, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        data = fb_vec.data
        assert data.shape == shape + (5,)
        
        # Check patterns
        for i in range(5):
            assert np.all(data[i:i+2, i:i+2, i:i+2, i] == (i + 1) * 10)
    
    def test_values_property(self, temp_vol_files):
        """Test values property."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        values = fb_vec.values
        data = fb_vec.data
        
        np.testing.assert_array_equal(values, data)
    
    def test_shape_property(self, temp_vol_files):
        """Test shape property."""
        files, shape, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        assert fb_vec.shape == shape + (5,)


class TestFileBackedNeuroVecSeries:
    """Test FileBackedNeuroVec series extraction."""
    
    def test_series_single_voxel(self, temp_vol_files):
        """Test series extraction for single voxel."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Get series for voxel (5, 5, 5)
        series = fb_vec.series(5, 5, 5)
        assert series.shape == (5,)
        # Volume 4 has pattern at [4:6, 4:6, 4:6] which includes (5,5,5)
        np.testing.assert_array_equal(series, [1, 2, 3, 4, 50])
        
        # Get series for voxel with pattern
        series = fb_vec.series(0, 0, 0)
        assert series[0] == 10  # Pattern at volume 0
        assert series[1] == 2   # No pattern at volume 1 for (0,0,0)
        
        # Get series for voxel (1,1,1) which has patterns in volumes 0 and 1
        series = fb_vec.series(1, 1, 1)
        assert series[0] == 10
        assert series[1] == 20
    
    def test_series_coordinates(self, temp_vol_files):
        """Test series extraction with coordinate array."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Multiple voxels
        coords = np.array([[5, 5, 5], [0, 0, 0], [9, 9, 9]])
        series = fb_vec.series(coords)
        
        assert series.shape == (5, 3)
        # First voxel (5,5,5) - has pattern at time 4
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 50])
        # Second voxel (0,0,0) has pattern at time 0
        assert series[0, 1] == 10
        assert series[1, 1] == 2  # No pattern at time 1
    
    def test_series_out_of_bounds(self, temp_vol_files):
        """Test series extraction with out-of-bounds coordinates."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Coordinates with some out of bounds
        coords = np.array([[5, 5, 5], [20, 20, 20], [-1, -1, -1]])
        series = fb_vec.series(coords)
        
        assert series.shape == (5, 3)
        # Valid coordinate (5,5,5) has pattern at time 4
        np.testing.assert_array_equal(series[:, 0], [1, 2, 3, 4, 50])
        # Out of bounds should be zeros
        np.testing.assert_array_equal(series[:, 1], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(series[:, 2], [0, 0, 0, 0, 0])
    
    def test_series_linear_indices(self, temp_vol_files):
        """Test series extraction with linear indices (now implemented)."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)

        # Linear indices now work
        result = fb_vec.series(np.array([0, 1, 2]))
        assert result.shape[1] == 3
    
    def test_series_invalid_input(self, temp_vol_files):
        """Test series extraction with invalid input."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        with pytest.raises(ValueError, match="Invalid input"):
            fb_vec.series("invalid")


class TestFileBackedNeuroVecOperations:
    """Test FileBackedNeuroVec operations."""
    
    def test_sub_vector_slice(self, temp_vol_files):
        """Test sub_vector with slice."""
        files, shape, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Get subset
        sub_vec = fb_vec.sub_vector(slice(1, 4))
        
        assert isinstance(sub_vec, FileBackedNeuroVec)
        assert sub_vec.n_volumes == 3
        assert sub_vec.shape == shape + (3,)
        
        # Check data
        assert np.all(sub_vec[1:3, 1:3, 1:3, 0] == 20)
    
    def test_sub_vector_indices(self, temp_vol_files):
        """Test sub_vector with indices array."""
        files, shape, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Get subset
        sub_vec = fb_vec.sub_vector(np.array([0, 2, 4]))
        
        assert sub_vec.n_volumes == 3
        assert np.all(sub_vec[2:4, 2:4, 2:4, 1] == 30)
    
    def test_vols_all(self, temp_vol_files):
        """Test vols() to get all volumes."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        vol_list = fb_vec.vols()
        
        assert len(vol_list) == 5
        for i, vol in enumerate(vol_list):
            assert isinstance(vol, DenseNeuroVol)
            assert np.all(vol.data[i:i+2, i:i+2, i:i+2] == (i + 1) * 10)
    
    def test_vols_subset(self, temp_vol_files):
        """Test vols() with indices."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        vol_list = fb_vec.vols([0, 2, 4])
        
        assert len(vol_list) == 3
        assert np.all(vol_list[0].data[0:2, 0:2, 0:2] == 10)
        assert np.all(vol_list[1].data[2:4, 2:4, 2:4] == 30)
    
    def test_as_matrix(self, temp_vol_files):
        """Test conversion to matrix."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        mat = fb_vec.as_matrix()
        
        assert mat.shape == (5, 1000)  # 5 volumes, 10*10*10 voxels
        # Check first voxel (0,0,0) across time
        # Volume 0: has pattern at [0:2, 0:2, 0:2] so (0,0,0) = 10
        # Volume 1: has pattern at [1:3, 1:3, 1:3] so (0,0,0) = 2
        # Volume 2-4: no pattern at (0,0,0) so values are 3, 4, 5
        assert np.all(mat[:, 0] == [10, 2, 3, 4, 5])
    
    def test_as_dense(self, temp_vol_files):
        """Test conversion to DenseNeuroVec."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        dense_vec = fb_vec.as_dense()
        
        from neuroim.neuro_vec import DenseNeuroVec
        assert isinstance(dense_vec, DenseNeuroVec)
        assert dense_vec.shape == fb_vec.shape
        np.testing.assert_array_equal(dense_vec.data, fb_vec.data)
    
    def test_as_sparse_with_mask(self, temp_vol_files):
        """Test conversion to SparseNeuroVec with mask."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Create mask
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[0:5, 0:5, 0:5] = True
        
        sparse_vec = fb_vec.as_sparse(mask)
        
        from neuroim import SparseNeuroVec
        assert isinstance(sparse_vec, SparseNeuroVec)
    
    def test_as_sparse_auto_mask(self, temp_vol_files):
        """Test conversion to SparseNeuroVec with automatic mask."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        sparse_vec = fb_vec.as_sparse()
        
        from neuroim import SparseNeuroVec
        assert isinstance(sparse_vec, SparseNeuroVec)


class TestFileBackedNeuroVecArithmetic:
    """Test FileBackedNeuroVec arithmetic operations."""
    
    def test_add_scalar(self, temp_vol_files):
        """Test addition with scalar."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        result = fb_vec._arithmetic_op(10, np.add)
        
        from neuroim.neuro_vec import DenseNeuroVec
        assert isinstance(result, DenseNeuroVec)
        assert np.all(result.data[0:2, 0:2, 0:2, 0] == 20)
    
    def test_multiply_neurovec(self, temp_vol_files):
        """Test multiplication with another NeuroVec."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Create another vector
        from neuroim.neuro_vec import DenseNeuroVec
        other = DenseNeuroVec(np.ones(fb_vec.shape) * 2, fb_vec.space)
        
        result = fb_vec._arithmetic_op(other, np.multiply)
        
        assert isinstance(result, DenseNeuroVec)
        assert np.all(result.data[0:2, 0:2, 0:2, 0] == 20)
    
    def test_arithmetic_incompatible_shape(self, temp_vol_files):
        """Test arithmetic with incompatible shapes."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Create vector with different shape
        from neuroim.neuro_vec import DenseNeuroVec
        other_space = NeuroSpace(dim=[10, 10, 10, 3])
        other = DenseNeuroVec(np.ones((10, 10, 10, 3)), other_space)
        
        with pytest.raises(ValueError, match="Incompatible shapes"):
            fb_vec._arithmetic_op(other, np.add)
    
    def test_arithmetic_unsupported_type(self, temp_vol_files):
        """Test arithmetic with unsupported type."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        with pytest.raises(TypeError, match="Unsupported operand"):
            fb_vec._arithmetic_op("invalid", np.add)


class TestFileBackedNeuroVecComparison:
    """Test FileBackedNeuroVec comparison operations."""
    
    def test_compare_scalar(self, temp_vol_files):
        """Test comparison with scalar."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        result = fb_vec._comparison_op(5, np.greater)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == fb_vec.shape
        # Volumes 0-3 have values <= 5, volume 4 has values > 5 in pattern area
        assert np.all(result[4:6, 4:6, 4:6, 4] == True)
    
    def test_compare_neurovec(self, temp_vol_files):
        """Test comparison with another NeuroVec."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Create comparison vector
        from neuroim.neuro_vec import DenseNeuroVec
        other = DenseNeuroVec(np.ones(fb_vec.shape) * 15, fb_vec.space)
        
        result = fb_vec._comparison_op(other, np.less)
        
        assert result.shape == fb_vec.shape
        # Most values should be less than 15
        assert np.sum(result) > 0
    
    def test_compare_file_backed(self, temp_vol_files):
        """Test comparison with another FileBackedNeuroVec."""
        files, _, _ = temp_vol_files
        fb_vec1 = FileBackedNeuroVec(files)
        fb_vec2 = FileBackedNeuroVec(files)
        
        result = fb_vec1._comparison_op(fb_vec2, np.equal)
        
        assert np.all(result)  # Should be all equal
    
    def test_compare_incompatible_shape(self, temp_vol_files):
        """Test comparison with incompatible shapes."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        # Create vector with different shape
        from neuroim.neuro_vec import DenseNeuroVec
        other_space = NeuroSpace(dim=[10, 10, 10, 3])
        other = DenseNeuroVec(np.ones((10, 10, 10, 3)), other_space)
        
        with pytest.raises(ValueError, match="Incompatible shapes"):
            fb_vec._comparison_op(other, np.equal)
    
    def test_compare_unsupported_type(self, temp_vol_files):
        """Test comparison with unsupported type."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files)
        
        with pytest.raises(TypeError, match="Unsupported operand"):
            fb_vec._comparison_op([1, 2, 3], np.equal)


class TestFileBackedNeuroVecRepr:
    """Test FileBackedNeuroVec string representation."""
    
    def test_repr(self, temp_vol_files):
        """Test string representation."""
        files, _, _ = temp_vol_files
        fb_vec = FileBackedNeuroVec(files, cache_size=3)
        
        repr_str = repr(fb_vec)
        
        assert "FileBackedNeuroVec" in repr_str
        assert "10 X 10 X 10 X 5" in repr_str
        assert "N Files   : 5" in repr_str
        assert "Cache Size: 3" in repr_str


class TestFileBackedNeuroVecInconsistentVolumes:
    """Test FileBackedNeuroVec with inconsistent volumes."""
    
    def test_inconsistent_volume_shape(self):
        """Test error when volumes have different shapes."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create volumes with different shapes
            files = []
            
            # First volume
            vol1 = DenseNeuroVol(np.ones((10, 10, 10)), NeuroSpace(dim=[10, 10, 10]))
            file1 = os.path.join(temp_dir, "vol1.nii.gz")
            write_vol(vol1, file1)
            files.append(file1)
            
            # Second volume with different shape
            vol2 = DenseNeuroVol(np.ones((8, 8, 8)), NeuroSpace(dim=[8, 8, 8]))
            file2 = os.path.join(temp_dir, "vol2.nii.gz")
            write_vol(vol2, file2)
            files.append(file2)
            
            # Create FileBackedNeuroVec
            fb_vec = FileBackedNeuroVec(files)
            
            # Accessing first volume should work
            fb_vec[:, :, :, 0]
            
            # Accessing second volume should raise error
            with pytest.raises(ValueError, match="inconsistent shape"):
                fb_vec[:, :, :, 1]
                
        finally:
            # Cleanup
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
            os.rmdir(temp_dir)