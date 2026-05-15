"""Comprehensive tests for NeuroVec classes based on R neuroim2 tests."""

import pytest
import numpy as np
import tempfile
import os
from neuroim import (NeuroSpace, DenseNeuroVec, SparseNeuroVec, NeuroVec, DenseNeuroVol, LogicalNeuroVol, BigNeuroVec, FileBackedNeuroVec, MappedNeuroVec)
from neuroim.neuro_vec import neurovec, neurovecseq
from neuroim.neuro_vol import neurovol


def gen_dat(d1=12, d2=12, d3=12, d4=4, rand=False):
    """Generate test NeuroVec data (matches R helper function)."""
    if rand:
        dat = np.random.randn(d1, d2, d3, d4)
    else:
        dat = np.zeros((d1, d2, d3, d4))
    spc = NeuroSpace(dim=[d1, d2, d3, d4])
    return DenseNeuroVec(dat, spc)


class TestNeuroVecConstruction:
    """Test NeuroVec construction from various data sources."""
    
    def test_construct_dense_neurovec(self):
        """Test basic DenseNeuroVec construction."""
        bv = gen_dat(12, 12, 12, 4)
        assert bv is not None
        assert bv[0, 0, 0, 0] == 0
        assert bv[11, 11, 11, 3] == 0
        assert bv.shape == (12, 12, 12, 4)
    
    def test_construct_from_matrix_time_x_voxels(self):
        """Test construction from matrix (time x voxels)."""
        spc = NeuroSpace(dim=[10, 10, 10, 4])
        mat = np.random.randn(4, 1000)  # time x voxels
        vec = neurovec(mat, spc)
        
        assert vec.shape == (10, 10, 10, 4)
        assert isinstance(vec, DenseNeuroVec)
    
    def test_construct_from_matrix_voxels_x_time(self):
        """Test construction from matrix (voxels x time)."""
        spc = NeuroSpace(dim=[10, 10, 10, 4])
        mat = np.random.randn(1000, 4)  # voxels x time
        vec = neurovec(mat.T, spc)  # Transpose to time x voxels
        
        assert vec.shape == (10, 10, 10, 4)
        assert isinstance(vec, DenseNeuroVec)
    
    def test_construct_sparse_neurovec_with_mask(self):
        """Test SparseNeuroVec construction with mask."""
        # Create mask
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[2:8, 2:8, 2:8] = True
        mask_space = NeuroSpace(dim=[10, 10, 10])
        mask = LogicalNeuroVol(mask_data, mask_space)
        
        # Create sparse data
        n_voxels = mask.sum
        n_time = 5
        mat = np.random.randn(n_time, n_voxels)
        
        # Create 4D space
        space_4d = NeuroSpace(dim=[10, 10, 10, n_time])
        
        # Create sparse vector
        vec = SparseNeuroVec(mat, space_4d, mask)
        
        assert vec.shape == (10, 10, 10, 5)
        assert isinstance(vec, SparseNeuroVec)
    
    def test_construct_from_list_of_volumes(self):
        """Test construction from list of volumes."""
        spc = NeuroSpace(dim=[10, 10, 10])
        vol = neurovol(np.random.randn(10, 10, 10), spc)
        vlist = [vol, vol, vol]
        
        vec = neurovecseq(vlist)
        assert vec.shape == (10, 10, 10, 3)
        assert isinstance(vec, DenseNeuroVec)
    
    def test_construct_from_4d_array(self):
        """Test construction from 4D array."""
        dat = np.random.randn(10, 10, 10, 5)
        vec = neurovec(dat)
        
        assert vec.shape == (10, 10, 10, 5)
        assert isinstance(vec, DenseNeuroVec)


class TestNeuroVecIndexing:
    """Test NeuroVec indexing operations."""
    
    def setup_method(self):
        """Create test vector."""
        self.vec = gen_dat(10, 10, 10, 5, rand=True)
    
    def test_single_volume_extraction(self):
        """Test extracting single volume."""
        vol = self.vec[..., 0]
        assert isinstance(vol, DenseNeuroVol)
        assert vol.shape == (10, 10, 10)
        np.testing.assert_array_equal(vol.data, self.vec.data[..., 0])
    
    def test_single_voxel_time_series(self):
        """Test extracting time series for single voxel."""
        ts = self.vec[5, 5, 5, :]
        assert ts.shape == (5,)
        np.testing.assert_array_equal(ts, self.vec.data[5, 5, 5, :])
    
    def test_slab_extraction(self):
        """Test extracting spatial slab."""
        slab = self.vec[2:8, 2:8, 2:8, :]
        assert slab.shape == (6, 6, 6, 5)
    
    def test_time_range_extraction(self):
        """Test extracting time range."""
        sub = self.vec[..., 1:3]
        assert sub.shape == (10, 10, 10, 2)


class TestNeuroVecSeries:
    """Test series extraction methods."""
    
    def setup_method(self):
        """Create test vector with known pattern."""
        dat = np.zeros((10, 10, 10, 5))
        # Set specific voxels with patterns
        for t in range(5):
            dat[5, 5, 5, t] = t + 1  # Linear
            dat[7, 7, 7, t] = (t + 1) * 2  # Double
        
        self.vec = DenseNeuroVec(dat, NeuroSpace(dim=[10, 10, 10, 5]))
    
    def test_series_single_voxel(self):
        """Test series extraction for single voxel."""
        ts = self.vec.series(5, 5, 5)
        assert ts.shape == (5,)
        np.testing.assert_array_equal(ts, [1, 2, 3, 4, 5])
        
        ts = self.vec.series(7, 7, 7)
        np.testing.assert_array_equal(ts, [2, 4, 6, 8, 10])
    
    def test_series_coordinate_matrix(self):
        """Test series extraction with coordinate matrix."""
        coords = np.array([[5, 5, 5], [7, 7, 7]])
        series_mat = self.vec.series(coords)
        
        assert series_mat.shape == (5, 2)  # time x voxels
        np.testing.assert_array_equal(series_mat[:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(series_mat[:, 1], [2, 4, 6, 8, 10])
    
    def test_series_linear_indices(self):
        """Test series extraction with linear indices."""
        # Calculate linear indices
        idx1 = np.ravel_multi_index((5, 5, 5), (10, 10, 10), order='F')
        idx2 = np.ravel_multi_index((7, 7, 7), (10, 10, 10), order='F')
        
        series_mat = self.vec.series(np.array([idx1, idx2]))
        assert series_mat.shape == (5, 2)

    def test_pythonic_series_methods_match_legacy_dispatcher(self):
        """Explicit series methods match the deprecated dispatcher."""
        coords = np.array([[5, 5, 5], [7, 7, 7]])
        idx = np.array([
            np.ravel_multi_index((5, 5, 5), (10, 10, 10), order="F"),
            np.ravel_multi_index((7, 7, 7), (10, 10, 10), order="F"),
        ])

        np.testing.assert_array_equal(
            self.vec.series_at(5, 5, 5),
            self.vec.data[5, 5, 5, :],
        )
        np.testing.assert_array_equal(
            self.vec.series_at_coords(coords),
            self.vec.series(coords),
        )
        np.testing.assert_array_equal(
            self.vec.series_at_indices(idx),
            self.vec.series(idx),
        )

        with pytest.warns(DeprecationWarning, match="series_at"):
            self.vec.series(coords)

    def test_explicit_series_methods_raise_on_out_of_bounds(self):
        """Explicit Pythonic series methods reject silent OOB indexing."""
        with pytest.raises(IndexError, match="coordinate out of bounds"):
            self.vec.series_at(-1, 0, 0)

        coords = np.array([[5, 5, 5], [10, 0, 0]])
        with pytest.raises(IndexError, match="coordinates out of bounds"):
            self.vec.series_at_coords(coords)

        bad_idx = np.array([0, np.prod(self.vec.shape[:3])])
        with pytest.raises(IndexError, match="linear indices out of bounds"):
            self.vec.series_at_indices(bad_idx)

    def test_explicit_series_methods_zero_fill_when_requested(self):
        """Zero-fill OOB behavior is opt-in for explicit series methods."""
        coords = np.array([[5, 5, 5], [10, 0, 0]])
        coord_result = self.vec.series_at_coords(coords, out_of_bounds="zero")
        assert coord_result.shape == (5, 2)
        np.testing.assert_array_equal(coord_result[:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(coord_result[:, 1], np.zeros(5))

        idx = np.array([
            np.ravel_multi_index((5, 5, 5), self.vec.shape[:3], order="F"),
            np.prod(self.vec.shape[:3]),
        ])
        index_result = self.vec.series_at_indices(idx, out_of_bounds="zero")
        assert index_result.shape == (5, 2)
        np.testing.assert_array_equal(index_result[:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(index_result[:, 1], np.zeros(5))

        np.testing.assert_array_equal(
            self.vec.series_at(-1, 0, 0, out_of_bounds="zero"),
            np.zeros(5),
        )

    def test_series_at_world_matches_voxel_lookup(self):
        """World-coordinate shortcut routes through the spatial contract."""
        np.testing.assert_array_equal(
            self.vec.series_at_world(np.array([5.0, 5.0, 5.0])),
            self.vec.series_at(5, 5, 5),
        )

        with pytest.raises(ValueError, match="outside the image grid"):
            self.vec.series_at_world(np.array([-1.0, 0.0, 0.0]))

    def test_series_roi_world_returns_typed_single_voxel_result(self):
        """World-coordinate ROI shortcut returns the typed ROI result."""
        result = self.vec.series_roi_world(np.array([5.0, 5.0, 5.0]))
        assert result.values.shape == (5, 1)
        np.testing.assert_array_equal(result.values[:, 0], self.vec.series_at(5, 5, 5))
        np.testing.assert_array_equal(result.coords, np.array([[5, 5, 5]]))
        assert result.provenance.method_name == "series_roi_world"
        assert result.provenance.radius == 0.0

    def test_series_roi_world_records_radius_in_receipt(self):
        """Spherical world-coordinate ROI provenance records the caller radius."""
        result = self.vec.series_roi_world(np.array([5.0, 5.0, 5.0]), radius=2.0)
        assert result.provenance.method_name == "series_roi_world"
        assert result.provenance.radius == 2.0


class TestNeuroVecArithmetic:
    """Test NeuroVec arithmetic operations."""
    
    def setup_method(self):
        """Create test vectors."""
        self.vec1 = gen_dat(10, 10, 10, 4, rand=False)
        self.vec1.data[:] = 2
        self.vec2 = gen_dat(10, 10, 10, 4, rand=False)
        self.vec2.data[:] = 3
    
    def test_vector_addition(self):
        """Test vector addition."""
        result = self.vec1 + self.vec2
        assert isinstance(result, DenseNeuroVec)
        assert np.all(result.data == 5)
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        result = self.vec1 * 3
        assert isinstance(result, DenseNeuroVec)
        assert np.all(result.data == 6)
    
    def test_volume_vector_operation(self):
        """Test operation between volume and vector."""
        vol = DenseNeuroVol(np.ones((10, 10, 10)) * 2, 
                           NeuroSpace(dim=[10, 10, 10]))
        result = self.vec1 * vol
        
        assert isinstance(result, DenseNeuroVec)
        assert np.all(result.data == 4)


class TestNeuroVecConversions:
    """Test conversions between dense and sparse."""
    
    def setup_method(self):
        """Create test vector."""
        dat = np.random.randn(10, 10, 10, 4)
        dat[dat < 0] = 0  # Make sparse
        self.dense_vec = DenseNeuroVec(dat, NeuroSpace(dim=[10, 10, 10, 4]))
    
    def test_dense_to_sparse_conversion(self):
        """Test converting dense to sparse."""
        # Create mask
        mask = self.dense_vec.data[..., 0] != 0
        mask_vol = LogicalNeuroVol(mask, NeuroSpace(dim=[10, 10, 10]))
        
        sparse_vec = self.dense_vec.to_sparse(mask_vol)
        assert isinstance(sparse_vec, SparseNeuroVec)
        with pytest.warns(DeprecationWarning, match="to_sparse"):
            self.dense_vec.as_sparse(mask_vol)
        
        # Check a few values
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if mask[i, j, k]:
                        ts_dense = self.dense_vec.series_at(i, j, k)
                        ts_sparse = sparse_vec.series_at(i, j, k)
                        np.testing.assert_array_almost_equal(ts_dense, ts_sparse)
    
    def test_sparse_to_dense_conversion(self):
        """Test converting sparse to dense."""
        # Create sparse vector
        mask = self.dense_vec.data[..., 0] != 0
        mask_vol = LogicalNeuroVol(mask, NeuroSpace(dim=[10, 10, 10]))
        sparse_vec = self.dense_vec.to_sparse(mask_vol)
        
        # Convert back to dense
        dense_vec2 = sparse_vec.to_dense()
        assert isinstance(dense_vec2, DenseNeuroVec)
        with pytest.warns(DeprecationWarning, match="to_dense"):
            sparse_vec.as_dense()
        
        # Check values match where mask is true
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if mask[i, j, k]:
                        ts1 = self.dense_vec.series_at(i, j, k)
                        ts2 = dense_vec2.series_at(i, j, k)
                        np.testing.assert_array_almost_equal(ts1, ts2)


class TestNeuroVecMethods:
    """Test various NeuroVec methods."""
    
    def setup_method(self):
        """Create test vector."""
        self.vec = gen_dat(10, 10, 10, 4, rand=True)
    
    def test_vols_method(self):
        """Test vols() method returns list of volumes."""
        vols = self.vec.vols()
        assert len(vols) == 4
        assert all(isinstance(v, DenseNeuroVol) for v in vols)
        assert all(v.shape == (10, 10, 10) for v in vols)
        
        # Test with indices
        vols = self.vec.vols([0, 2])
        assert len(vols) == 2
        np.testing.assert_array_equal(vols[0].data, self.vec.data[..., 0])
        np.testing.assert_array_equal(vols[1].data, self.vec.data[..., 2])
    
    def test_sub_vector(self):
        """Test sub_vector extraction."""
        sub = self.vec.subvolumes([1, 2])
        assert sub.shape == (10, 10, 10, 2)
        np.testing.assert_array_equal(sub.data[..., 0], self.vec.data[..., 1])
        np.testing.assert_array_equal(sub.data[..., 1], self.vec.data[..., 2])
        with pytest.warns(DeprecationWarning, match="subvolumes"):
            self.vec.sub_vector([1, 2])
    
    def test_concat(self):
        """Test vector concatenation."""
        vec1 = gen_dat(10, 10, 10, 3, rand=True)
        vec2 = gen_dat(10, 10, 10, 2, rand=True)
        
        concat_vec = vec1.concat(vec2)
        assert concat_vec.shape == (10, 10, 10, 5)
        
        # Check data
        np.testing.assert_array_equal(concat_vec.data[..., :3], vec1.data)
        np.testing.assert_array_equal(concat_vec.data[..., 3:], vec2.data)
    
    def test_as_matrix(self):
        """Test conversion to matrix."""
        mat = self.vec.as_matrix()
        assert mat.shape == (1000, 4)  # voxels x time
        
        # Check a specific voxel
        idx = np.ravel_multi_index((5, 5, 5), (10, 10, 10), order='F')
        np.testing.assert_array_equal(mat[idx, :], self.vec.data[5, 5, 5, :])
    
    def test_scale_series(self):
        """Test time series scaling."""
        # Create vector with known mean/std
        vec = gen_dat(10, 10, 10, 10, rand=True)
        
        # Scale series
        scaled = vec.scale_series(center=True, scale=True)
        
        # Check that each voxel's time series has mean ~0 and std ~1
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    ts = scaled.series(i, j, k)
                    if np.std(vec.series(i, j, k)) > 0:  # Skip constant series
                        assert np.abs(np.mean(ts)) < 1e-10
                        assert np.abs(np.std(ts) - 1.0) < 1e-10


class TestSparseNeuroVec:
    """Test SparseNeuroVec specific functionality."""
    
    def test_sparse_construction(self):
        """Test sparse vector construction."""
        # Create mask
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[3:7, 3:7, 3:7] = True
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=[10, 10, 10]))
        
        # Create data
        n_voxels = mask.sum
        n_time = 5
        data = np.random.randn(n_time, n_voxels)
        
        # Create sparse vector
        space_4d = NeuroSpace(dim=[10, 10, 10, n_time])
        vec = SparseNeuroVec(data, space_4d, mask)
        
        assert vec.shape == (10, 10, 10, 5)
        assert vec.mask.sum == n_voxels
    
    def test_sparse_series_extraction(self):
        """Test series extraction from sparse vector."""
        # Create simple mask
        mask_data = np.zeros((5, 5, 5), dtype=bool)
        mask_data[2, 2, 2] = True
        mask_data[3, 3, 3] = True
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=[5, 5, 5]))
        
        # Create data with known pattern
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])  # time x voxels
        
        space_4d = NeuroSpace(dim=[5, 5, 5, 4])
        vec = SparseNeuroVec(data, space_4d, mask)
        
        # Test series extraction
        ts1 = vec.series(2, 2, 2)
        np.testing.assert_array_equal(ts1, [1, 2, 3, 4])
        
        ts2 = vec.series(3, 3, 3)
        np.testing.assert_array_equal(ts2, [10, 20, 30, 40])
        
        # Non-mask location should return zeros
        ts3 = vec.series(0, 0, 0)
        np.testing.assert_array_equal(ts3, [0, 0, 0, 0])


class TestMemoryMappedVec:
    """Test memory-mapped NeuroVec variants."""
    
    def test_big_neurovec(self):
        """Test BigNeuroVec construction and operations."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp:
            tmp_name = tmp.name
        
        try:
            # Create BigNeuroVec
            data = np.random.randn(10, 10, 10, 4)
            space = NeuroSpace(dim=[10, 10, 10, 4])
            vec = BigNeuroVec(data, space, filename=tmp_name)
            
            assert vec.shape == (10, 10, 10, 4)
            
            # Test series extraction
            ts = vec.series(5, 5, 5)
            np.testing.assert_array_equal(ts, data[5, 5, 5, :])
            
            # Test arithmetic
            vec2 = vec + 1
            assert isinstance(vec2, BigNeuroVec)
            assert vec2[5, 5, 5, 0] == vec[5, 5, 5, 0] + 1
            
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    
    def test_mapped_neurovec(self):
        """Test MappedNeuroVec with transformation."""
        # Create base vector
        base = gen_dat(10, 10, 10, 4, rand=True)
        
        # Create mapped vector with scaling
        def scale_func(vol):
            return vol * 2
        
        mapped = MappedNeuroVec(base, scale_func)
        
        # Test that values are scaled
        assert mapped[5, 5, 5, 0] == base[5, 5, 5, 0] * 2
        
        # Test series extraction
        ts_base = base.series(5, 5, 5)
        ts_mapped = mapped.series(5, 5, 5)
        np.testing.assert_array_equal(ts_mapped, ts_base * 2)


if __name__ == "__main__":
    pytest.main([__file__])
