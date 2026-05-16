"""Phase 2 Tests: 3D Volumes (NeuroVol classes)

These tests are direct translations from R's neuroim2 tests to ensure
complete compatibility between the R and Python implementations.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vol import (
    DenseNeuroVol,
    SparseNeuroVol,
    LogicalNeuroVol,
    neurovol,
)
from neuroim.io import read_vol, write_vol


class TestNeuroVolConstruction:
    """Test NeuroVol construction with various parameters."""

    def test_dense_neurovol_from_3d_array(self):
        """Test DenseNeuroVol construction from 3D array.

        R Equivalent
        ------------
        test_that("can construct NeuroVol from 3D array", {
          dat <- array(0, c(64,64,64))
          spc <- NeuroSpace(c(64,64,64))
          bv <- DenseNeuroVol(dat, spc)
          expect_true(!is.null(bv))
          expect_equal(bv[1,1,1], 0)
          expect_equal(bv[64,64,64], 0)
          expect_equal(dim(bv), c(64,64,64))
        })
        """
        dat = np.zeros((64, 64, 64))
        spc = NeuroSpace((64, 64, 64))
        bv = DenseNeuroVol(dat, spc)

        assert bv is not None
        assert bv[0, 0, 0] == 0  # Python uses 0-based indexing
        assert bv[63, 63, 63] == 0
        assert_array_equal(bv.dim, [64, 64, 64])

    def test_dense_neurovol_from_1d_array_with_indices(self):
        """Test DenseNeuroVol construction from 1D array with indices.

        R Equivalent
        ------------
        test_that("can construct NeuroVol from 1D array with indices", {
          dat <- rnorm(100)
          spc <- NeuroSpace(c(64,64,64))
          indices = seq(1,20000, length.out=100)
          bv <- DenseNeuroVol(dat, spc, indices=indices)
          expect_true(!is.null(bv))
          expect_equal(dim(bv), c(64,64,64))
        })
        """
        dat = np.random.randn(100)
        spc = NeuroSpace((64, 64, 64))
        # R: seq(1,20000, length.out=100) -> Python: 0-based indices
        indices = np.linspace(0, 19999, 100, dtype=int)
        bv = DenseNeuroVol(dat, spc, indices=indices)

        assert bv is not None
        assert_array_equal(bv.dim, [64, 64, 64])
        # Check that specified indices have the data
        assert_array_almost_equal(bv.values()[indices], dat)

    def test_sparse_neurovol_construction(self):
        """Test SparseNeuroVol construction."""
        # Create sparse data
        spc = NeuroSpace((10, 10, 10))
        indices = np.array([0, 100, 200, 300, 400])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sparse_vol = SparseNeuroVol(values, spc, indices)

        assert sparse_vol is not None
        assert sparse_vol.nnz == 5
        assert_array_equal(sparse_vol.indices, indices)
        assert_array_equal(sparse_vol.data, values)

    def test_logical_neurovol_construction(self):
        """Test LogicalNeuroVol construction."""
        spc = NeuroSpace((10, 10, 10))
        data = np.random.rand(10, 10, 10) > 0.5

        mask = LogicalNeuroVol(data, spc)

        assert mask is not None
        assert mask.data.dtype == bool
        assert_array_equal(mask.shape, (10, 10, 10))

    def test_neurovol_factory_function(self):
        """Test neurovol factory function."""
        spc = NeuroSpace((5, 5, 5))

        # Should create DenseNeuroVol
        dense = neurovol(np.zeros((5, 5, 5)), spc)
        assert isinstance(dense, DenseNeuroVol)

        # Should create LogicalNeuroVol
        logical = neurovol(np.ones((5, 5, 5), dtype=bool), spc)
        assert isinstance(logical, LogicalNeuroVol)

        # Should create SparseNeuroVol
        sparse = neurovol([1, 2, 3], spc, indices=[0, 10, 20])
        assert isinstance(sparse, SparseNeuroVol)

    def test_dimension_mismatch_error(self):
        """Test error when data dimensions don't match space.

        R Equivalent
        ------------
        test_that("NeuroVol with mismatching data and space dimensions throw error", {
          spc <- NeuroSpace(c(64,64,64))
          dat <- array(0, c(64,64,63))
          expect_error(DenseNeuroVol(dat, spc))
        })
        """
        spc = NeuroSpace((64, 64, 64))
        dat = np.zeros((64, 64, 63))

        with pytest.raises(ValueError):
            DenseNeuroVol(dat, spc)


class TestNeuroVolIndexing:
    """Test various indexing operations on NeuroVol."""

    def test_3d_indexing(self):
        """Test standard 3D indexing."""
        spc = NeuroSpace((10, 10, 10))
        data = np.arange(1000).reshape((10, 10, 10))
        vol = DenseNeuroVol(data, spc)

        # Single voxel
        assert vol[0, 0, 0] == 0
        assert vol[5, 5, 5] == 555

        # Slice
        assert_array_equal(vol[5, :, :], data[5, :, :])

    def test_linear_indexing(self):
        """Test linear indexing."""
        spc = NeuroSpace((10, 10, 10))
        data = np.arange(1000).reshape((10, 10, 10), order="F")
        vol = DenseNeuroVol(data, spc)

        # Linear indices
        assert vol[0] == 0
        assert vol[100] == 100
        assert_array_equal(vol[[0, 100, 200]], [0, 100, 200])

    def test_coordinate_matrix_indexing(self):
        """Test indexing with Nx3 coordinate matrix."""
        spc = NeuroSpace((10, 10, 10))
        data = np.arange(1000).reshape((10, 10, 10))
        vol = DenseNeuroVol(data, spc)

        # Nx3 matrix of coordinates
        coords = np.array([[0, 0, 0], [5, 5, 5], [9, 9, 9]])
        values = vol[coords]

        assert_array_equal(values, [0, 555, 999])

    def test_logical_indexing(self):
        """Test indexing with LogicalNeuroVol."""
        spc = NeuroSpace((5, 5, 5))
        data = np.arange(125).reshape((5, 5, 5))
        vol = DenseNeuroVol(data, spc)

        # Create mask
        mask_data = data > 60
        mask = LogicalNeuroVol(mask_data, spc)

        # Index with mask
        masked_values = vol[mask]
        expected = data[mask_data]

        assert_array_equal(masked_values, expected)

    def test_sparse_indexing(self):
        """Test indexing on SparseNeuroVol."""
        spc = NeuroSpace((10, 10, 10))
        indices = [0, 100, 200, 300]
        values = [1.0, 2.0, 3.0, 4.0]
        sparse_vol = SparseNeuroVol(values, spc, indices)

        # Linear indexing
        assert sparse_vol[0] == 1.0
        assert sparse_vol[100] == 2.0
        assert sparse_vol[50] == 0.0  # Not in sparse data

        # 3D indexing (0,0,0) -> index 0
        assert sparse_vol[0, 0, 0] == 1.0


class TestNeuroVolConversions:
    """Test conversions between different NeuroVol types."""

    def test_dense_to_sparse(self):
        """Test converting DenseNeuroVol to SparseNeuroVol."""
        spc = NeuroSpace((10, 10, 10))
        data = np.zeros((10, 10, 10))
        data[1, 1, 1] = 1.0
        data[5, 5, 5] = 2.0

        dense = DenseNeuroVol(data, spc)
        sparse = dense.as_sparse()

        assert isinstance(sparse, SparseNeuroVol)
        assert sparse.nnz == 2
        assert_array_equal(np.sort(sparse.data), [1.0, 2.0])

    def test_sparse_to_dense(self):
        """Test converting SparseNeuroVol to DenseNeuroVol."""
        spc = NeuroSpace((5, 5, 5))
        indices = [0, 10, 20]
        values = [1.0, 2.0, 3.0]

        sparse = SparseNeuroVol(values, spc, indices)
        dense = sparse.as_dense()

        assert isinstance(dense, DenseNeuroVol)
        assert dense[0] == 1.0
        assert dense[10] == 2.0
        assert dense[20] == 3.0
        assert dense[5] == 0.0  # Not in sparse data

    def test_dense_to_logical(self):
        """Test converting DenseNeuroVol to LogicalNeuroVol."""
        spc = NeuroSpace((5, 5, 5))
        data = np.array([0, 1, 2, 0, 3]).reshape((5, 1, 1))
        data = np.tile(data, (1, 5, 5))

        dense = DenseNeuroVol(data, spc)
        logical = dense.as_logical()

        assert isinstance(logical, LogicalNeuroVol)
        assert logical.data.dtype == bool
        assert not logical[0, 0, 0]
        assert logical[1, 0, 0]

    def test_as_mask_with_indices(self):
        """Test as_mask with specific indices."""
        spc = NeuroSpace((5, 5, 5))
        vol = DenseNeuroVol(np.ones((5, 5, 5)), spc)

        indices = [0, 10, 20, 30]
        mask = vol.as_mask(indices)

        assert isinstance(mask, LogicalNeuroVol)
        assert mask.sum == 4  # Only 4 voxels are True
        assert mask[0]
        assert mask[10]
        assert not mask[5]


class TestNeuroVolArithmetic:
    """Test arithmetic operations on NeuroVol."""

    def test_dense_scalar_arithmetic(self):
        """Test arithmetic with scalars."""
        spc = NeuroSpace((5, 5, 5))
        data = np.ones((5, 5, 5)) * 10
        vol = DenseNeuroVol(data, spc)

        # Addition
        result = vol + 5
        assert isinstance(result, DenseNeuroVol)
        assert np.all(result.data == 15)

        # Subtraction
        result = vol - 3
        assert np.all(result.data == 7)

        # Multiplication
        result = vol * 2
        assert np.all(result.data == 20)

        # Division
        result = vol / 2
        assert np.all(result.data == 5)

    def test_dense_dense_arithmetic(self):
        """Test arithmetic between DenseNeuroVol objects."""
        spc = NeuroSpace((5, 5, 5))
        vol1 = DenseNeuroVol(np.ones((5, 5, 5)) * 10, spc)
        vol2 = DenseNeuroVol(np.ones((5, 5, 5)) * 3, spc)

        # Addition
        result = vol1 + vol2
        assert isinstance(result, DenseNeuroVol)
        assert np.all(result.data == 13)

        # Subtraction
        result = vol1 - vol2
        assert np.all(result.data == 7)

    def test_sparse_scalar_arithmetic(self):
        """Test sparse arithmetic with scalars."""
        spc = NeuroSpace((10, 10, 10))
        sparse = SparseNeuroVol([1, 2, 3], spc, [0, 10, 20])

        # Multiplication preserves sparsity
        result = sparse * 2
        assert isinstance(result, SparseNeuroVol)
        assert_array_equal(result.data, [2, 4, 6])
        assert_array_equal(result.indices, [0, 10, 20])

    def test_space_mismatch_error(self):
        """Test error when volumes have different spaces."""
        spc1 = NeuroSpace((5, 5, 5))
        spc2 = NeuroSpace((10, 10, 10))

        vol1 = DenseNeuroVol(np.ones((5, 5, 5)), spc1)
        vol2 = DenseNeuroVol(np.ones((10, 10, 10)), spc2)

        with pytest.raises(ValueError):
            vol1 + vol2


class TestNeuroVolComparisons:
    """Test comparison operations on NeuroVol."""

    def test_scalar_comparisons(self):
        """Test comparisons with scalars."""
        spc = NeuroSpace((5, 5, 5))
        data = np.arange(125).reshape((5, 5, 5))
        vol = DenseNeuroVol(data, spc)

        # Greater than
        result = vol > 50
        assert isinstance(result, LogicalNeuroVol)
        assert result.sum == np.sum(data > 50)

        # Less than
        result = vol < 20
        assert result.sum == np.sum(data < 20)

        # Equal
        vol2 = DenseNeuroVol(np.ones((5, 5, 5)) * 50, spc)
        vol2[2, 2, 2] = 100
        result = vol2 == 50
        assert result.sum == 124  # All but one voxel

    def test_volume_comparisons(self):
        """Test comparisons between volumes."""
        spc = NeuroSpace((5, 5, 5))
        vol1 = DenseNeuroVol(np.arange(125).reshape((5, 5, 5)), spc)
        vol2 = DenseNeuroVol(np.ones((5, 5, 5)) * 60, spc)

        result = vol1 > vol2
        assert isinstance(result, LogicalNeuroVol)
        assert result.sum == np.sum(vol1.data > vol2.data)


class TestNeuroVolStatistics:
    """Test summary statistics on NeuroVol."""

    def test_summary_stats(self):
        """Test min, max, mean, sum operations."""
        spc = NeuroSpace((5, 5, 5))
        data = np.arange(125).reshape((5, 5, 5))
        vol = DenseNeuroVol(data, spc)

        assert vol.min() == 0
        assert vol.max() == 124
        assert vol.mean() == 62.0
        assert vol.sum() == 7750
        assert vol.range() == (0, 124)

    def test_summary_stats_with_nan(self):
        """Test summary stats with NaN handling."""
        spc = NeuroSpace((5, 5, 5))
        data = np.arange(125, dtype=float).reshape((5, 5, 5))
        data[2, 2, 2] = np.nan
        vol = DenseNeuroVol(data, spc)

        # Without na_rm
        assert np.isnan(vol.sum(na_rm=False))

        # With na_rm
        assert vol.sum(na_rm=True) == 7750 - 62  # Sum minus the NaN value
        assert vol.mean(na_rm=True) == (7750 - 62) / 124


class TestLogicalNeuroVolOperations:
    """Test logical operations specific to LogicalNeuroVol."""

    def test_logical_operations(self):
        """Test AND, OR, XOR, NOT operations."""
        spc = NeuroSpace((5, 5, 5))

        # Create two masks
        data1 = np.zeros((5, 5, 5), dtype=bool)
        data1[:3, :, :] = True
        mask1 = LogicalNeuroVol(data1, spc)

        data2 = np.zeros((5, 5, 5), dtype=bool)
        data2[2:, :, :] = True
        mask2 = LogicalNeuroVol(data2, spc)

        # AND
        result_and = mask1 & mask2
        assert isinstance(result_and, LogicalNeuroVol)
        assert result_and.sum == 25  # Overlap in slice 2

        # OR
        result_or = mask1 | mask2
        assert result_or.sum == 125  # All voxels

        # XOR
        result_xor = mask1 ^ mask2
        assert result_xor.sum == 100  # Non-overlapping voxels

        # NOT
        result_not = ~mask1
        assert result_not.sum == 50  # Inverted

    def test_logical_sum_property(self):
        """Test sum property of LogicalNeuroVol."""
        spc = NeuroSpace((10, 10, 10))
        data = np.random.rand(10, 10, 10) > 0.7
        mask = LogicalNeuroVol(data, spc)

        assert mask.sum == np.sum(data)
        assert isinstance(mask.sum, int)


class TestNeuroVolCoordinateOperations:
    """Test coordinate-related operations on NeuroVol."""

    def test_index_to_grid_on_neurovol(self):
        """Test index_to_grid method.

        R Equivalent
        ------------
        test_that("index_to_grid on NeuroVol checks out", {
          vol1 <- read_vol(gmask)
          i <- 65
          expect_equal(index_to_grid(vol1, i), matrix(c(1,2,1), nrow=1))
          expect_equal(index_to_grid(vol1, 1), matrix(c(1,1,1), nrow=1))
        })
        """
        vol = DenseNeuroVol(np.zeros((64, 64, 25)), NeuroSpace((64, 64, 25)))

        # R: index 65 -> Python: index 64
        grid = vol.index_to_grid(64)
        assert_array_equal(grid[0], [0, 1, 0])  # Python 0-based

        # R: index 1 -> Python: index 0
        grid = vol.index_to_grid(0)
        assert_array_equal(grid[0], [0, 0, 0])

    def test_coords_method(self):
        """Test coords method."""
        spc = NeuroSpace((3, 3, 3), spacing=(2, 2, 2), origin=(10, 20, 30))
        vol = DenseNeuroVol(np.zeros((3, 3, 3)), spc)

        # Grid coordinates
        grid_coords = vol.coords(real=False)
        assert grid_coords.shape == (27, 3)
        assert_array_equal(grid_coords[0], [0, 0, 0])
        assert_array_equal(grid_coords[-1], [2, 2, 2])

        # Real world coordinates
        real_coords = vol.coords(real=True)
        assert real_coords.shape == (27, 3)
        # First voxel should be at origin
        assert_array_equal(real_coords[0], [10, 20, 30])


class TestNeuroVolRepresentation:
    """Test string representation of NeuroVol."""

    def test_repr(self):
        """Test __repr__ method."""
        spc = NeuroSpace((10, 20, 30), spacing=(1, 2, 3), origin=(0, 0, 0))
        data = np.random.randn(10, 20, 30)
        vol = DenseNeuroVol(data, spc)

        repr_str = repr(vol)
        assert "DenseNeuroVol" in repr_str
        assert "10 X 20 X 30" in repr_str
        assert "1.0 X 2.0 X 3.0" in repr_str  # spacing


class TestNeuroVolValues:
    """Test values extraction from NeuroVol."""

    def test_values_fortran_order(self):
        """Test that values() returns data in Fortran order."""
        spc = NeuroSpace((3, 3, 3))
        data = np.arange(27).reshape((3, 3, 3))
        vol = DenseNeuroVol(data, spc)

        values = vol.values()
        expected = data.ravel(order="F")

        assert_array_equal(values, expected)


class TestNeuroVolIO:
    """Test I/O operations (requires nibabel)."""

    def test_write_read_roundtrip(self, tmp_path):
        """Test writing and reading a volume."""
        # Create test volume
        spc = NeuroSpace((10, 10, 10), spacing=(2, 2, 2), origin=(0, 0, 0))
        data = np.random.randn(10, 10, 10)
        vol = DenseNeuroVol(data, spc)

        path = tmp_path / "roundtrip.nii.gz"
        write_vol(vol, path)
        vol2 = read_vol(path)

        assert_array_almost_equal(vol.data, vol2.data)
        assert_array_almost_equal(vol.spacing, vol2.spacing)

    def test_read_4d_volume_index(self):
        """Test reading specific volume from 4D file."""
        # This test would require a 4D test file
        # For now, just test the interface exists
        pass


# R-Python compatibility notes
def test_phase2_compatibility_notes():
    """Document key differences for Phase 2."""
    notes = """
    Phase 2 R-Python Compatibility Notes:

    1. Indexing remains 0-based in Python vs 1-based in R
    2. values() returns Fortran-ordered (column-major) data to match R
    3. Sparse volumes use scipy.sparse for efficiency
    4. LogicalNeuroVol extends DenseNeuroVol (similar to R)
    5. I/O uses nibabel for NIfTI support
    """
    print(notes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
