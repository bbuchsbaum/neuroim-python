"""Phase 3 Tests: Core NeuroVec functionality (no I/O dependencies)

These tests cover the core NeuroVec functionality without requiring nibabel.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vol import DenseNeuroVol, LogicalNeuroVol
from neuroim.neuro_vec import (
    NeuroVec, DenseNeuroVec, SparseNeuroVec,
    neurovec, neurovecseq
)


def _unwrap(x):
    """Explicitly cross the typed->ndarray boundary.

    A1a: typed containers no longer implicitly coerce via np.asarray.
    ``.data`` is the native-shaped ndarray for both DenseNeuroVol (3-D)
    and DenseNeuroVec (4-D); plain ndarrays pass through unchanged.
    """
    return x.data if hasattr(x, "space") and hasattr(x, "data") else np.asarray(x)


def gen_dat(d1=12, d2=12, d3=12, d4=4, rand=False):
    """Helper function to generate test data (matches R's gen_dat)."""
    if rand:
        dat = np.random.randn(d1, d2, d3, d4)
    else:
        dat = np.zeros((d1, d2, d3, d4))
    spc = NeuroSpace((d1, d2, d3, d4))
    return DenseNeuroVec(dat, spc)


class TestNeuroVecConstruction:
    """Test NeuroVec construction with various parameters."""
    
    def test_dense_neurovec_construction(self):
        """Test DenseNeuroVec construction."""
        bv = gen_dat(12, 12, 12, 4)
        assert bv is not None
        assert bv[0, 0, 0, 0] == 0  # Python 0-based
        assert bv[11, 11, 11, 3] == 0
        assert_array_equal(bv.dim, [12, 12, 12, 4])
    
    def test_dense_neurovec_from_matrix(self):
        """Test DenseNeuroVec construction from matrix."""
        spc = NeuroSpace((10, 10, 10, 4))
        mat = np.random.randn(4, 1000)
        vec1 = neurovec(mat.T, spc)  # voxels x time
        vec2 = neurovec(mat, spc)    # time x voxels
        
        assert isinstance(vec1, DenseNeuroVec)
        assert isinstance(vec2, DenseNeuroVec)
        assert_array_almost_equal(vec1.data, vec2.data)
        assert vec1.data.size == vec2.data.size
    
    def test_sparse_neurovec_construction(self):
        """Test SparseNeuroVec construction."""
        # Create a simple mask
        mask_space = NeuroSpace((10, 10, 10))
        mask_data = np.random.rand(10, 10, 10) > 0.5
        mask = LogicalNeuroVol(mask_data, mask_space)
        
        idx = np.where(mask.data)
        n_masked = mask.sum
        mat = np.random.randn(n_masked, 5)
        
        vec_space = mask_space.add_dim(1, 5)  # Add 1 dimension of size 5
        vec = neurovec(mat.T, vec_space, mask=mask)
        
        assert isinstance(vec, SparseNeuroVec)
        assert_array_equal(vec.dim, [10, 10, 10, 5])
    
    def test_neurovec_from_vol_list(self):
        """Test NeuroVec construction from list of volumes."""
        spc = NeuroSpace((10, 10, 10))
        vol = DenseNeuroVol(np.random.randn(10, 10, 10), spc)
        vlist = [vol, vol, vol]
        vec = neurovec(vlist)
        
        assert isinstance(vec, DenseNeuroVec)
        assert_array_equal(vec.dim, [10, 10, 10, 3])


class TestNeuroVecOperations:
    """Test operations on NeuroVec."""
    
    def test_concatenate_neurovecs(self):
        """Test concatenating NeuroVecs."""
        bv1 = gen_dat(rand=True)
        bv2 = gen_dat(rand=True)
        
        bv3 = bv1.concat(bv2)
        assert isinstance(bv3, NeuroVec)
        assert_array_equal(bv3.dim, [12, 12, 12, 8])
        
        bv4 = bv1.concat(bv2, bv1, bv3)
        assert_array_equal(bv4.dim, [12, 12, 12, 20])
        assert isinstance(bv4, NeuroVec)
        assert bv4[0, 0, 0, 0] == bv1[0, 0, 0, 0]
    
    def test_extract_single_volume(self):
        """Test extracting single volume from NeuroVec."""
        bv1 = gen_dat()
        bv2 = gen_dat()
        
        bv3 = bv1.concat(bv2)
        
        vol1 = bv3[..., 0]  # Python 0-based
        assert isinstance(vol1, DenseNeuroVol)
        assert_array_equal(vol1.dim, [12, 12, 12])
    
    def test_sub_vector(self):
        """Test extracting sub-vector from NeuroVec."""
        bv1 = gen_dat()
        vec1 = bv1.sub_vector([0, 1])  # Python 0-based
        assert isinstance(vec1, NeuroVec)
        assert_array_equal(vec1.dim, [12, 12, 12, 2])
    
    def test_vols_method(self):
        """Test vols method to extract volumes."""
        bv1 = gen_dat(rand=True)
        vols = bv1.vols()
        
        assert len(vols) == 4
        assert all(isinstance(v, DenseNeuroVol) for v in vols)
        
        # Extract specific volumes
        vols2 = bv1.vols([1, 2])
        assert len(vols2) == 2
        assert_array_equal(vols2[0].data, _unwrap(bv1[..., 1]))
        assert_array_equal(vols2[1].data, _unwrap(bv1[..., 2]))


class TestNeuroVecSeries:
    """Test series extraction from NeuroVec."""
    
    def test_extract_single_series(self):
        """Test extracting time series for single voxel."""
        bv1 = gen_dat()
        series = bv1.series(0, 0, 0)  # Python 0-based
        assert_array_equal(series, bv1[0, 0, 0, :])
    
    def test_extract_multiple_series(self):
        """Test extracting time series for multiple voxels."""
        bv1 = gen_dat(rand=True)
        mat = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])  # Python 0-based
        
        # Extract series individually
        r1 = np.column_stack([bv1.series(coord[0], coord[1], coord[2]) for coord in mat])
        
        # Extract series in batch
        r2 = bv1.series(mat)
        
        assert_array_equal(r1, r2)
    
    def test_series_with_linear_indices(self):
        """Test series extraction with linear indices."""
        bv1 = gen_dat(10, 10, 10, 5, rand=True)
        
        # Test single linear index
        series1 = bv1.series(0)
        assert series1.shape == (5,)
        assert_array_equal(series1, bv1[0, 0, 0, :])
        
        # Test multiple linear indices
        indices = [0, 10, 100, 500]
        series_multi = bv1.series(np.array(indices))
        assert series_multi.shape == (5, 4)


class TestNeuroVecArithmetic:
    """Test arithmetic operations on NeuroVec."""
    
    def test_neurovec_arithmetic(self):
        """Test arithmetic operations between NeuroVecs."""
        bv1 = gen_dat(5, 5, 5, 5, rand=True)
        bv2 = gen_dat(5, 5, 5, 5, rand=True)
        
        bv3 = bv1 + bv2
        bv4 = bv1 * bv2
        bv5 = bv2 - bv1
        
        assert isinstance(bv3, DenseNeuroVec)
        assert isinstance(bv4, DenseNeuroVec)
        assert isinstance(bv5, DenseNeuroVec)
        
        assert_array_almost_equal(bv3.data, bv1.data + bv2.data)
        assert_array_almost_equal(bv4.data, bv1.data * bv2.data)
        assert_array_almost_equal(bv5.data, bv2.data - bv1.data)
    
    def test_neurovec_scalar_arithmetic(self):
        """Test arithmetic with scalars."""
        bv1 = gen_dat(5, 5, 5, 5, rand=True)
        
        bv2 = bv1 + 10
        bv3 = bv1 * 2
        bv4 = bv1 - 5
        bv5 = bv1 / 2
        
        assert_array_almost_equal(bv2.data, bv1.data + 10)
        assert_array_almost_equal(bv3.data, bv1.data * 2)
        assert_array_almost_equal(bv4.data, bv1.data - 5)
        assert_array_almost_equal(bv5.data, bv1.data / 2)

    def test_neurovec_reverse_scalar_arithmetic(self):
        """Test reverse arithmetic with scalar on the left."""
        bv1 = gen_dat(5, 5, 5, 5, rand=True)

        assert_array_almost_equal((10 + bv1).data, 10 + bv1.data)
        assert_array_almost_equal((2 * bv1).data, 2 * bv1.data)
        assert_array_almost_equal((20 - bv1).data, 20 - bv1.data)
        assert_array_almost_equal((100 / bv1).data, 100 / bv1.data)
    
    def test_neurovec_neurovol_arithmetic(self):
        """Test arithmetic between NeuroVec and NeuroVol."""
        spc_3d = NeuroSpace((5, 5, 5))
        vol = DenseNeuroVol(np.random.randn(5, 5, 5), spc_3d)
        
        spc_4d = spc_3d.add_dim(1, 3)  # Add 1 dimension of size 3
        bv1 = DenseNeuroVec(np.random.randn(5, 5, 5, 3), spc_4d)
        
        bv2 = bv1 + vol
        bv3 = bv2 - vol
        bv4 = bv1 * vol
        
        # Check that adding and subtracting vol cancels out
        assert_array_almost_equal(bv3.data, bv1.data)


class TestNeuroVecConversions:
    """Test conversions between dense and sparse."""
    
    def test_dense_to_sparse(self):
        """Test converting dense NeuroVec to sparse."""
        bv1 = gen_dat(12, 12, 12, 4, rand=True)
        
        # Create mask with specific indices
        mask_data = np.zeros((12, 12, 12), dtype=bool)
        mask_data.flat[[0, 99, 999]] = True
        mask_space = NeuroSpace((12, 12, 12))
        mask = LogicalNeuroVol(mask_data, mask_space)
        
        svec = bv1.as_sparse(mask)
        assert isinstance(svec, SparseNeuroVec)
        assert_array_equal(svec.dim, [12, 12, 12, 4])
        assert svec.mask.sum == 3
    
    def test_sparse_to_dense(self):
        """Test converting sparse NeuroVec to dense."""
        # Create sparse vector
        mask_space = NeuroSpace((10, 10, 10))
        mask_data = np.random.rand(10, 10, 10) > 0.7
        mask = LogicalNeuroVol(mask_data, mask_space)
        
        vec_space = mask_space.add_dim(1, 5)  # Add 1 dimension of size 5
        sparse_data = np.random.randn(5, mask.sum)
        svec = SparseNeuroVec(sparse_data, vec_space, mask)
        
        # Convert to dense
        dvec = svec.as_dense()
        assert isinstance(dvec, DenseNeuroVec)
        assert_array_equal(dvec.dim, svec.dim)
        
        # Check that values match at mask locations
        mask_indices = np.where(mask.data.ravel(order='F'))[0]
        for i, idx in enumerate(mask_indices):
            coords = np.unravel_index(idx, (10, 10, 10), order='F')
            assert_array_almost_equal(dvec[coords[0], coords[1], coords[2], :], 
                                    sparse_data[:, i])


class TestSparseNeuroVec:
    """Test SparseNeuroVec specific functionality."""
    
    def test_sparse_construction_and_indexing(self):
        """Test SparseNeuroVec construction and indexing."""
        dat = np.random.randn(12, 12, 12, 4)
        spc = NeuroSpace((12, 12, 12, 4))
        tmp = np.random.randn(12, 12, 12)
        mask = tmp > -1000000  # All true
        mask_vol = LogicalNeuroVol(mask, NeuroSpace((12, 12, 12)))
        
        # Create sparse from dense data
        bvec = neurovec(dat, spc, mask=mask_vol)
        
        assert_array_equal(bvec.dim, dat.shape)
        assert bvec[0, 0, 0, 0] == dat[0, 0, 0, 0]
        assert_array_equal(_unwrap(bvec[0, 0, 0, :]), dat[0, 0, 0, :])
        assert_array_equal(_unwrap(bvec[0, 0, :, :]), dat[0, 0, :, :])
        assert_array_equal(_unwrap(bvec[0, :, :, :]), dat[0, :, :, :])
        assert_array_equal(_unwrap(bvec[0, 1:3, :, :]), dat[0, 1:3, :, :])
        assert_array_equal(_unwrap(bvec[0, 1:3, :, 1:3]), dat[0, 1:3, :, 1:3])
        assert_array_equal(_unwrap(bvec[0:3, 1:3, :, :]), dat[0:3, 1:3, :, :])
        assert_array_equal(_unwrap(bvec[0, 1:3, 1:3, :]), dat[0, 1:3, 1:3, :])
    
    def test_sparse_arithmetic(self):
        """Test arithmetic on SparseNeuroVec."""
        dat = np.random.randn(12, 12, 12, 4)
        spc = NeuroSpace((12, 12, 12, 4))
        mask = np.random.rand(12, 12, 12) > 0.8
        mask_vol = LogicalNeuroVol(mask, NeuroSpace((12, 12, 12)))
        
        bv1 = neurovec(dat, spc, mask=mask_vol)
        bv2 = bv1 + bv1
        bv3 = bv1 * bv2
        bv4 = bv3 - bv1
        
        assert isinstance(bv2, SparseNeuroVec)
        assert isinstance(bv3, SparseNeuroVec)
        assert isinstance(bv4, SparseNeuroVec)


class TestNeuroVecMatrix:
    """Test matrix conversions."""
    
    def test_as_matrix(self):
        """Test converting NeuroVec to matrix."""
        bv1 = gen_dat(5, 5, 5, 5, rand=True)
        mat = bv1.as_matrix()
        
        # Verify shape (voxels x time)
        assert mat.shape == (125, 5)
        
        # Verify content matches series extraction
        for i in range(125):
            series = bv1.series(i)
            assert_array_equal(mat[i, :], series)


class TestNeuroVecScaling:
    """Test scaling operations."""
    
    def test_scale_series(self):
        """Test scaling time series."""
        bv1 = gen_dat(5, 5, 5, 10, rand=True)
        
        # Center only
        bv2 = bv1.scale_series(center=True, scale=False)
        # Check mean is ~0 for each voxel
        means = np.mean(bv2.data, axis=3)
        assert_array_almost_equal(means, np.zeros_like(means), decimal=10)
        
        # Scale only
        bv3 = bv1.scale_series(center=False, scale=True)
        # Check std is ~1 for each voxel (where original std != 0)
        stds = np.std(bv3.data, axis=3)
        mask = np.std(bv1.data, axis=3) != 0
        assert_array_almost_equal(stds[mask], np.ones_like(stds[mask]), decimal=10)
        
        # Center and scale
        bv4 = bv1.scale_series(center=True, scale=True)
        means = np.mean(bv4.data, axis=3)
        stds = np.std(bv4.data, axis=3)
        assert_array_almost_equal(means, np.zeros_like(means), decimal=10)
        assert_array_almost_equal(stds[mask], np.ones_like(stds[mask]), decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
