"""Tests for indexing utilities."""

import pytest
import numpy as np
from neuroimpy import NeuroSpace, DenseNeuroVol, DenseNeuroVec
from neuroimpy.indexing import (
    linear_access, matricized_access, from_matvec, to_matvec, dot_reduce
)


class TestLinearAccess:
    """Test linear_access function."""

    def setup_method(self):
        self.vol_space = NeuroSpace(dim=[4, 5, 6])
        self.vol_data = np.arange(120, dtype=float).reshape(4, 5, 6, order='F')
        self.vol = DenseNeuroVol(self.vol_data, self.vol_space)

        self.vec_space = NeuroSpace(dim=[4, 5, 6, 3])
        self.vec_data = np.random.randn(4, 5, 6, 3)
        self.vec = DenseNeuroVec(self.vec_data, self.vec_space)

    def test_vol_linear(self):
        indices = np.array([0, 1, 10, 119])
        result = linear_access(self.vol, indices)
        expected = self.vol_data.ravel(order='F')[indices]
        np.testing.assert_array_equal(result, expected)

    def test_vec_linear(self):
        indices = np.array([0, 5, 20])
        result = linear_access(self.vec, indices)
        # Result should be (time, n_indices)
        assert result.shape == (3, 3)

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            linear_access("bad", np.array([0]))


class TestMatricizedAccess:
    """Test matricized_access function."""

    def setup_method(self):
        self.space = NeuroSpace(dim=[4, 5, 6, 3])
        self.data = np.arange(360, dtype=float).reshape(4, 5, 6, 3, order='F')
        self.vec = DenseNeuroVec(self.data, self.space)

    def test_basic(self):
        row_idx = np.array([0, 2])
        col_idx = np.array([0, 5, 10])
        result = matricized_access(self.vec, row_idx, col_idx)
        assert result.shape == (2, 3)

    def test_single_element(self):
        result = matricized_access(self.vec, np.array([1]), np.array([0]))
        assert result.shape == (1, 1)


class TestFromMatvec:
    """Test from_matvec function."""

    def setup_method(self):
        self.space = NeuroSpace(dim=[3, 4, 5, 2])
        self.n_vox = 3 * 4 * 5
        self.n_time = 2

    def test_roundtrip(self):
        """from_matvec(to_matvec(vec)) should reconstruct the original."""
        data = np.random.randn(3, 4, 5, 2)
        vec = DenseNeuroVec(data, self.space)
        mat, space = to_matvec(vec)
        reconstructed = from_matvec(mat, space)
        np.testing.assert_allclose(reconstructed.data, data)

    def test_shape(self):
        mat = np.random.randn(self.n_time, self.n_vox)
        result = from_matvec(mat, self.space)
        assert result.shape == (3, 4, 5, 2)

    def test_wrong_cols(self):
        mat = np.random.randn(self.n_time, self.n_vox + 1)
        with pytest.raises(ValueError, match="columns"):
            from_matvec(mat, self.space)

    def test_wrong_rows(self):
        mat = np.random.randn(self.n_time + 1, self.n_vox)
        with pytest.raises(ValueError, match="rows"):
            from_matvec(mat, self.space)

    def test_not_2d(self):
        mat = np.random.randn(self.n_time, self.n_vox, 1)
        with pytest.raises(ValueError, match="2D"):
            from_matvec(mat, self.space)


class TestToMatvec:
    """Test to_matvec function."""

    def setup_method(self):
        self.space = NeuroSpace(dim=[3, 4, 5, 2])
        self.data = np.random.randn(3, 4, 5, 2)
        self.vec = DenseNeuroVec(self.data, self.space)

    def test_shape(self):
        mat, space = to_matvec(self.vec)
        assert mat.shape == (2, 60)
        assert space is self.vec.space

    def test_fortran_order(self):
        """Voxels should be flattened in Fortran order."""
        mat, _ = to_matvec(self.vec)
        # First row is time=0, flattened in F order
        np.testing.assert_array_equal(mat[0], self.data[..., 0].ravel(order='F'))


class TestDotReduce:
    """Test dot_reduce function."""

    def setup_method(self):
        self.space = NeuroSpace(dim=[3, 4, 5, 4])
        self.data = np.ones((3, 4, 5, 4))
        self.vec = DenseNeuroVec(self.data, self.space)

    def test_uniform_weights(self):
        weights = np.ones(4)
        result = dot_reduce(self.vec, weights)
        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (3, 4, 5)
        np.testing.assert_allclose(result.data, 4.0)

    def test_specific_weights(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        result = dot_reduce(self.vec, weights)
        # sum(1*1 + 1*2 + 1*3 + 1*4) = 10
        np.testing.assert_allclose(result.data, 10.0)

    def test_zero_weights(self):
        weights = np.zeros(4)
        result = dot_reduce(self.vec, weights)
        np.testing.assert_allclose(result.data, 0.0)

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="weights length"):
            dot_reduce(self.vec, np.ones(3))

    def test_varying_data(self):
        """Non-uniform data with specific weights."""
        data = np.zeros((3, 4, 5, 4))
        data[0, 0, 0, :] = np.array([1.0, 2.0, 3.0, 4.0])
        vec = DenseNeuroVec(data, self.space)
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        result = dot_reduce(vec, weights)
        assert result.data[0, 0, 0] == 1.0
        assert result.data[1, 0, 0] == 0.0
