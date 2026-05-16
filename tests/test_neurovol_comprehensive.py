"""Comprehensive tests for NeuroVol classes based on R neuroim2 tests."""

import pytest
import numpy as np
import tempfile
import os

from neuroim import (
    NeuroSpace,
    DenseNeuroVol,
    SparseNeuroVol,
    LogicalNeuroVol,
    write_vol,
)
from neuroim.io import read_vol
from neuroim.neuro_vol import neurovol


class TestNeuroVolConstruction:
    """Test NeuroVol construction from various data sources."""

    def test_construct_from_3d_array(self):
        """Test construction from 3D array."""
        dat = np.zeros((64, 64, 64))
        spc = NeuroSpace(dim=[64, 64, 64])
        bv = DenseNeuroVol(dat, spc)

        assert bv is not None
        assert bv[0, 0, 0] == 0
        assert bv[63, 63, 63] == 0
        assert bv.shape == (64, 64, 64)

    def test_construct_from_1d_array_with_indices(self):
        """Test construction from 1D array with indices."""
        dat = np.random.randn(100)
        spc = NeuroSpace(dim=[64, 64, 64])

        # Create indices
        indices = np.linspace(0, 20000, 100, dtype=int)
        bv = SparseNeuroVol(dat, spc, indices)

        assert bv is not None
        assert bv.shape == (64, 64, 64)

    def test_construct_from_1d_vector(self):
        """Test construction from 1D vector."""
        spc = NeuroSpace(dim=[64, 64, 64])
        dat = np.random.randn(np.prod(spc.dim))
        bv = DenseNeuroVol(dat, spc)

        assert bv is not None
        assert bv.shape == (64, 64, 64)

    def test_construct_from_matrix_single_row(self):
        """Test construction from matrix with 1 row."""
        spc = NeuroSpace(dim=[64, 64, 64])
        dat = np.random.randn(1, np.prod(spc.dim))
        bv = DenseNeuroVol(dat, spc)

        assert bv is not None
        assert bv.shape == (64, 64, 64)

    def test_construct_from_matrix_single_column(self):
        """Test construction from matrix with 1 column."""
        spc = NeuroSpace(dim=[64, 64, 64])
        dat = np.random.randn(np.prod(spc.dim), 1)
        bv = DenseNeuroVol(dat, spc)

        assert bv is not None
        assert bv.shape == (64, 64, 64)

    def test_construct_logical_neurovol(self):
        """Test construction of LogicalNeuroVol."""
        dat = np.zeros((10, 10, 10), dtype=bool)
        dat[5, 5, 5] = True
        spc = NeuroSpace(dim=[10, 10, 10])

        vol = LogicalNeuroVol(dat, spc)
        assert vol[5, 5, 5]
        assert not vol[0, 0, 0]
        assert vol.sum == 1


class TestNeuroVolIndexing:
    """Test NeuroVol indexing and slicing operations."""

    def setup_method(self):
        """Create test volume."""
        self.dat = np.arange(64**3).reshape(64, 64, 64)
        self.spc = NeuroSpace(dim=[64, 64, 64])
        self.vol = DenseNeuroVol(self.dat, self.spc)

    def test_single_voxel_indexing(self):
        """Test single voxel access."""
        assert self.vol[0, 0, 0] == self.dat[0, 0, 0]
        assert self.vol[63, 63, 63] == self.dat[63, 63, 63]
        assert self.vol[10, 20, 30] == self.dat[10, 20, 30]

    def test_slice_indexing(self):
        """Test slice indexing."""
        # Single slice
        slice_data = self.vol[:, :, 10]
        assert slice_data.shape == (64, 64)
        np.testing.assert_array_equal(slice_data, self.dat[:, :, 10])

        # Range slice
        range_data = self.vol[10:20, 10:20, 10:20]
        assert range_data.shape == (10, 10, 10)
        np.testing.assert_array_equal(range_data, self.dat[10:20, 10:20, 10:20])

    def test_boolean_indexing(self):
        """Test boolean indexing."""
        mask = self.dat > 1000
        masked_vals = self.vol[mask]
        # Check if anything was masked
        if np.sum(mask) > 0:
            assert len(masked_vals) == np.sum(mask)
            np.testing.assert_array_equal(masked_vals, self.dat[mask])
        else:
            # If no values > 1000, masked_vals should be empty
            assert masked_vals is not None
            assert len(masked_vals) == 0

    def test_coordinate_indexing(self):
        """Test indexing with coordinate arrays."""
        coords = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20]])
        vals = self.vol[coords[:, 0], coords[:, 1], coords[:, 2]]
        expected = np.array(
            [self.dat[0, 0, 0], self.dat[10, 10, 10], self.dat[20, 20, 20]]
        )
        np.testing.assert_array_equal(vals, expected)


class TestNeuroVolArithmetic:
    """Test NeuroVol arithmetic operations."""

    def setup_method(self):
        """Create test volumes."""
        self.spc = NeuroSpace(dim=[10, 10, 10])
        self.vol1 = DenseNeuroVol(np.ones((10, 10, 10)) * 2, self.spc)
        self.vol2 = DenseNeuroVol(np.ones((10, 10, 10)) * 3, self.spc)

    def test_addition(self):
        """Test volume addition."""
        result = self.vol1 + self.vol2
        assert isinstance(result, DenseNeuroVol)
        assert np.all(result.data == 5)

        # Scalar addition
        result = self.vol1 + 5
        assert np.all(result.data == 7)

    def test_subtraction(self):
        """Test volume subtraction."""
        result = self.vol2 - self.vol1
        assert isinstance(result, DenseNeuroVol)
        assert np.all(result.data == 1)

        # Scalar subtraction
        result = self.vol1 - 1
        assert np.all(result.data == 1)

    def test_multiplication(self):
        """Test volume multiplication."""
        result = self.vol1 * self.vol2
        assert isinstance(result, DenseNeuroVol)
        assert np.all(result.data == 6)

        # Scalar multiplication
        result = self.vol1 * 3
        assert np.all(result.data == 6)

    def test_division(self):
        """Test volume division."""
        result = self.vol2 / self.vol1
        assert isinstance(result, DenseNeuroVol)
        assert np.all(result.data == 1.5)

        # Scalar division
        result = self.vol2 / 3
        assert np.all(result.data == 1)

    def test_comparison(self):
        """Test volume comparison operations."""
        result = self.vol1 > 1
        assert isinstance(result, LogicalNeuroVol)
        assert np.all(result.data)

        result = self.vol1 < self.vol2
        assert isinstance(result, LogicalNeuroVol)
        assert np.all(result.data)

        result = self.vol1 == 2
        assert isinstance(result, LogicalNeuroVol)
        assert np.all(result.data)


class TestNeuroVolConversions:
    """Test conversions between NeuroVol types."""

    def setup_method(self):
        """Create test volume."""
        self.spc = NeuroSpace(dim=[10, 10, 10])
        dat = np.random.randn(10, 10, 10)
        dat[dat < 0] = 0  # Make sparse
        self.dense_vol = DenseNeuroVol(dat, self.spc)

    def test_dense_to_sparse(self):
        """Test conversion from dense to sparse."""
        sparse_vol = self.dense_vol.as_sparse()
        assert isinstance(sparse_vol, SparseNeuroVol)

        # Check values match
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    assert sparse_vol[i, j, k] == self.dense_vol[i, j, k]

    def test_sparse_to_dense(self):
        """Test conversion from sparse to dense."""
        sparse_vol = self.dense_vol.as_sparse()
        dense_vol = sparse_vol.as_dense()
        assert isinstance(dense_vol, DenseNeuroVol)

        # Check values match
        np.testing.assert_array_equal(dense_vol.data, self.dense_vol.data)

    def test_to_logical(self):
        """Test conversion to logical volume."""
        logical_vol = self.dense_vol.as_logical()
        assert isinstance(logical_vol, LogicalNeuroVol)
        assert logical_vol.data.dtype == bool

        # Check non-zero locations match
        expected = self.dense_vol.data != 0
        np.testing.assert_array_equal(logical_vol.data, expected)

    def test_logical_to_sparse(self):
        """Test conversion from logical to sparse."""
        logical_vol = self.dense_vol.as_logical()
        sparse_vol = logical_vol.as_sparse()
        assert isinstance(sparse_vol, SparseNeuroVol)

        # Check that sparse vol has 1s where logical is True
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if logical_vol[i, j, k]:
                        assert sparse_vol[i, j, k] == 1
                    else:
                        assert sparse_vol[i, j, k] == 0


class TestNeuroVolIO:
    """Test reading and writing NeuroVol objects."""

    def test_write_read_nifti(self, tmp_path):
        """Test writing and reading NIFTI files."""
        # Create test volume
        spc = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2])
        dat = np.random.randn(10, 10, 10)
        vol = DenseNeuroVol(dat, spc)

        path = tmp_path / "vol.nii"
        write_vol(vol, path)
        vol2 = read_vol(path)

        np.testing.assert_array_almost_equal(vol.data, vol2.data)
        np.testing.assert_array_equal(vol.shape, vol2.shape)
        np.testing.assert_array_almost_equal(vol.spacing, vol2.spacing)

    def test_write_read_nifti_gz(self):
        """Test writing and reading compressed NIFTI files."""
        # Create test volume
        spc = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2])
        dat = np.random.randn(10, 10, 10)
        vol = DenseNeuroVol(dat, spc)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_name = tmp.name

        try:
            # Write volume
            write_vol(vol, tmp_name)

            # Read it back
            vol2 = read_vol(tmp_name)

            # Check data matches
            np.testing.assert_array_almost_equal(vol.data, vol2.data)
            np.testing.assert_array_equal(vol.shape, vol2.shape)

        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


class TestNeuroVolCoordinates:
    """Test coordinate system transformations."""

    def setup_method(self):
        """Create test volume with custom transformation."""
        # Create space with non-identity transformation
        self.spc = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2], origin=[10, 20, 30])
        self.vol = DenseNeuroVol(np.zeros((10, 10, 10)), self.spc)

    def test_grid_to_world(self):
        """Test grid to world coordinate transformation."""
        # Test origin
        world_coords = self.spc.grid_to_coord(np.array([[0, 0, 0]]))
        expected = np.array([[10, 20, 30]])
        np.testing.assert_array_almost_equal(world_coords, expected)

        # Test other point
        world_coords = self.spc.grid_to_coord(np.array([[1, 1, 1]]))
        expected = np.array([[12, 22, 32]])  # origin + spacing
        np.testing.assert_array_almost_equal(world_coords, expected)

    def test_world_to_grid(self):
        """Test world to grid coordinate transformation."""
        # Test origin
        grid_coords = self.spc.coord_to_grid(np.array([[10, 20, 30]]))
        expected = np.array([[0, 0, 0]])
        np.testing.assert_array_almost_equal(grid_coords, expected)

        # Test other point
        grid_coords = self.spc.coord_to_grid(np.array([[12, 22, 32]]))
        expected = np.array([[1, 1, 1]])
        np.testing.assert_array_almost_equal(grid_coords, expected)

    def test_index_conversions(self):
        """Test linear index conversions."""
        # Grid to index
        coords = np.array([[0, 0, 0], [1, 1, 1], [9, 9, 9]])
        indices = self.spc.grid_to_index(coords)

        # Index to grid
        coords2 = self.spc.index_to_grid(indices)
        np.testing.assert_array_equal(coords, coords2)


class TestNeuroVolFactory:
    """Test the neurovol factory function."""

    def test_factory_dense(self):
        """Test factory creates DenseNeuroVol."""
        dat = np.random.randn(10, 10, 10)
        spc = NeuroSpace(dim=[10, 10, 10])
        vol = neurovol(dat, spc)
        assert isinstance(vol, DenseNeuroVol)

    def test_factory_sparse(self):
        """Test factory creates SparseNeuroVol with indices."""
        dat = np.random.randn(100)
        spc = NeuroSpace(dim=[10, 10, 10])
        indices = np.arange(100)
        vol = neurovol(dat, spc, indices=indices)
        assert isinstance(vol, SparseNeuroVol)

    def test_factory_logical(self):
        """Test factory creates LogicalNeuroVol from boolean data."""
        dat = np.zeros((10, 10, 10), dtype=bool)
        dat[5, 5, 5] = True
        spc = NeuroSpace(dim=[10, 10, 10])
        vol = neurovol(dat, spc)
        assert isinstance(vol, LogicalNeuroVol)

    def test_factory_infer_space(self):
        """Test factory infers space from data shape."""
        dat = np.random.randn(10, 10, 10)
        # Create space from data shape
        space = NeuroSpace(dim=dat.shape)
        vol = neurovol(dat, space)
        assert vol.shape == (10, 10, 10)
        assert isinstance(vol.space, NeuroSpace)


class TestNeuroVolMethods:
    """Test various NeuroVol methods."""

    def setup_method(self):
        """Create test volume."""
        self.spc = NeuroSpace(dim=[10, 10, 10])
        self.dat = np.random.randn(10, 10, 10)
        self.vol = DenseNeuroVol(self.dat, self.spc)

    def test_values_method(self):
        """Test values() method returns flattened array."""
        vals = self.vol.values()
        assert vals.ndim == 1
        assert len(vals) == 10 * 10 * 10
        np.testing.assert_array_equal(vals, self.dat.ravel(order="F"))

    def test_vol_sum(self):
        """Test sum() method."""
        expected = np.sum(self.dat)
        assert np.isclose(self.vol.sum(), expected)

    def test_vol_mean(self):
        """Test mean() method."""
        expected = np.mean(self.dat)
        assert np.isclose(self.vol.mean(), expected)

    def test_vol_mean_alias(self):
        """Test vol_mean() compatibility alias."""
        assert np.isclose(self.vol.vol_mean(), np.mean(self.dat))

    def test_vol_sd(self):
        """Test std() method."""
        expected = np.std(self.dat)
        assert np.isclose(np.std(self.vol.data), expected)

    def test_vol_sd_alias(self):
        """Test vol_sd() compatibility alias."""
        assert np.isclose(self.vol.vol_sd(), np.std(self.dat))

    def test_min_max(self):
        """Test min/max methods."""
        assert self.vol.min() == np.min(self.dat)
        assert self.vol.max() == np.max(self.dat)

    def test_which_min_max_aliases(self):
        """Test which_min()/which_max() compatibility aliases."""
        flat_data = self.vol.data.ravel(order="F")
        assert self.vol.which_min() == int(np.argmin(flat_data))
        assert self.vol.which_max() == int(np.argmax(flat_data))

    def test_which_min_max(self):
        """Test finding min/max indices."""
        # Get flattened data in Fortran order to match internal storage
        flat_data = self.vol.data.ravel(order="F")
        min_idx = np.argmin(flat_data)
        max_idx = np.argmax(flat_data)

        # Convert to grid coordinates
        min_coords = self.spc.index_to_grid(np.array([min_idx]))
        max_coords = self.spc.index_to_grid(np.array([max_idx]))

        # Check values
        assert (
            self.vol[min_coords[0, 0], min_coords[0, 1], min_coords[0, 2]]
            == self.vol.min()
        )
        assert (
            self.vol[max_coords[0, 0], max_coords[0, 1], max_coords[0, 2]]
            == self.vol.max()
        )


class TestSparseNeuroVol:
    """Test SparseNeuroVol specific functionality."""

    def test_sparse_construction(self):
        """Test sparse volume construction."""
        spc = NeuroSpace(dim=[10, 10, 10])

        # Create sparse data
        indices = np.array([0, 10, 20, 30, 40])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        vol = SparseNeuroVol(values, spc, indices)

        # Check sparsity
        assert vol.shape == (10, 10, 10)
        assert len(vol.indices) == 5

        # Check values at indices
        coords = spc.index_to_grid(indices)
        for i, coord in enumerate(coords):
            assert vol[coord[0], coord[1], coord[2]] == values[i]

    def test_sparse_arithmetic(self):
        """Test arithmetic on sparse volumes."""
        spc = NeuroSpace(dim=[10, 10, 10])

        # Create two sparse volumes
        indices1 = np.array([0, 10, 20])
        values1 = np.array([1.0, 2.0, 3.0])
        vol1 = SparseNeuroVol(values1, spc, indices1)

        indices2 = np.array([10, 20, 30])
        values2 = np.array([4.0, 5.0, 6.0])
        vol2 = SparseNeuroVol(values2, spc, indices2)

        # Test addition
        result = vol1 + vol2
        assert isinstance(result, DenseNeuroVol)  # Result is dense

        # Check specific values
        coords1 = spc.index_to_grid(np.array([0]))
        coords2 = spc.index_to_grid(np.array([10]))
        coords3 = spc.index_to_grid(np.array([30]))

        assert result[coords1[0, 0], coords1[0, 1], coords1[0, 2]] == 1.0
        assert result[coords2[0, 0], coords2[0, 1], coords2[0, 2]] == 6.0  # 2 + 4
        assert result[coords3[0, 0], coords3[0, 1], coords3[0, 2]] == 6.0


if __name__ == "__main__":
    pytest.main([__file__])
