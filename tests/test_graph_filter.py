"""Tests for connectivity-graph-based spatial filtering."""

import pytest
import numpy as np
from scipy import sparse

from neuroim import (
    DenseNeuroVol, LogicalNeuroVol, DenseNeuroVec, NeuroSpace,
    cgb_make_graph, cgb_filter, cgb_smooth,
    cgb_smooth_loro, cgb_nuisance, laplace_enhance,
)


def _make_space_3d(nx=5, ny=5, nz=5):
    return NeuroSpace(dim=(nx, ny, nz))


def _make_space_4d(nx=5, ny=5, nz=5, nt=10):
    return NeuroSpace(dim=(nx, ny, nz, nt))


def _make_vol(nx=5, ny=5, nz=5, value=1.0):
    space = _make_space_3d(nx, ny, nz)
    data = np.full((nx, ny, nz), value, dtype=float)
    return DenseNeuroVol(data, space)


def _make_mask(nx=5, ny=5, nz=5):
    """Create a mask with the interior voxels set to True."""
    space = _make_space_3d(nx, ny, nz)
    mask_data = np.zeros((nx, ny, nz), dtype=bool)
    mask_data[1:-1, 1:-1, 1:-1] = True
    return LogicalNeuroVol(mask_data, space)


class TestCgbMakeGraph:
    """Tests for cgb_make_graph."""

    def test_full_volume_graph(self):
        """Graph from a 3x3x3 volume without mask."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        assert graph.shape == (27, 27)
        # Symmetric
        diff = graph - graph.T
        assert diff.nnz == 0

    def test_graph_symmetry(self):
        """Adjacency matrix must be symmetric."""
        vol = _make_vol(4, 4, 4)
        graph = cgb_make_graph(vol)
        diff = graph - graph.T
        assert diff.nnz == 0

    def test_center_voxel_neighbors(self):
        """Center voxel of 3x3x3 has 6 neighbors."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        # Center voxel index in row-major: (1,1,1)
        # But we need the linear index in the flattened mask
        center_idx = np.ravel_multi_index((1, 1, 1), (3, 3, 3))
        degree = graph[center_idx].sum()
        assert degree == 6

    def test_corner_voxel_neighbors(self):
        """Corner voxel of 3x3x3 has 3 neighbors."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        corner_idx = np.ravel_multi_index((0, 0, 0), (3, 3, 3))
        degree = graph[corner_idx].sum()
        assert degree == 3

    def test_masked_graph(self):
        """Graph from a mask includes only masked voxels."""
        mask = _make_mask(5, 5, 5)
        graph = cgb_make_graph(mask, mask=mask)
        n_masked = int(mask.data.sum())
        assert graph.shape == (n_masked, n_masked)
        assert n_masked == 27  # 3x3x3 interior

    def test_empty_mask(self):
        """Graph from an empty mask is 0x0."""
        space = _make_space_3d(3, 3, 3)
        mask = np.zeros((3, 3, 3), dtype=bool)
        graph = cgb_make_graph(space, mask=mask)
        assert graph.shape == (0, 0)

    def test_from_neurovol(self):
        """Accepts NeuroVol as input."""
        vol = _make_vol(4, 4, 4)
        graph = cgb_make_graph(vol)
        assert graph.shape == (64, 64)

    def test_from_neurospace(self):
        """Accepts NeuroSpace as input."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        assert graph.shape == (27, 27)

    def test_binary_values(self):
        """Adjacency values are 0 or 1."""
        space = _make_space_3d(4, 4, 4)
        graph = cgb_make_graph(space)
        unique_vals = np.unique(graph.data)
        np.testing.assert_array_equal(unique_vals, [1.0])


class TestCgbFilter:
    """Tests for cgb_filter."""

    def test_constant_data_unchanged(self):
        """Filtering constant data returns same values."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        data = np.ones(27)
        result = cgb_filter(data, graph)
        np.testing.assert_allclose(result, 1.0)

    def test_smoothing_effect(self):
        """A single hot voxel value should spread to neighbors."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        data = np.zeros(27)
        center = np.ravel_multi_index((1, 1, 1), (3, 3, 3))
        data[center] = 6.0  # give center a high value
        result = cgb_filter(data, graph)
        # Center value should decrease (spread to neighbors)
        assert result[center] < 6.0

    def test_with_weights(self):
        """Filtering with custom weights."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        data = np.ones(27)
        weights = graph * 2.0
        result = cgb_filter(data, graph, weights=weights)
        np.testing.assert_allclose(result, 1.0)

    def test_data_must_be_1d(self):
        """Raises on non-1D data."""
        space = _make_space_3d(3, 3, 3)
        graph = cgb_make_graph(space)
        with pytest.raises(ValueError, match="1-D"):
            cgb_filter(np.ones((27, 2)), graph)


class TestCgbSmooth:
    """Tests for cgb_smooth."""

    def test_constant_volume_unchanged(self):
        """Smoothing a constant volume returns roughly the same values."""
        vol = _make_vol(5, 5, 5, value=3.0)
        result = cgb_smooth(vol, fwhm=2.0)
        assert isinstance(result, DenseNeuroVol)
        np.testing.assert_allclose(result.data, 3.0, atol=1e-10)

    def test_smoothing_reduces_variance(self):
        """Smoothing should reduce data variance."""
        rng = np.random.RandomState(42)
        space = _make_space_3d(8, 8, 8)
        data = rng.randn(8, 8, 8)
        vol = DenseNeuroVol(data, space)
        result = cgb_smooth(vol, fwhm=3.0)
        assert result.data.var() < vol.data.var()

    def test_output_shape_matches(self):
        """Output has the same shape as input."""
        vol = _make_vol(5, 5, 5)
        result = cgb_smooth(vol, fwhm=2.0)
        assert result.shape == vol.shape

    def test_with_mask(self):
        """Smoothing with mask only modifies masked voxels."""
        vol = _make_vol(5, 5, 5, value=1.0)
        mask = _make_mask(5, 5, 5)
        result = cgb_smooth(vol, fwhm=2.0, mask=mask)
        # Unmasked border voxels should remain 1.0
        assert result.data[0, 0, 0] == 1.0

    def test_invalid_fwhm(self):
        """fwhm must be positive."""
        vol = _make_vol(3, 3, 3)
        with pytest.raises(ValueError, match="positive"):
            cgb_smooth(vol, fwhm=-1.0)

    def test_preserves_space(self):
        """Output space matches input space."""
        vol = _make_vol(5, 5, 5)
        result = cgb_smooth(vol, fwhm=2.0)
        np.testing.assert_array_equal(result.space.dim, vol.space.dim)


class TestCgbSmoothLoro:
    """Tests for cgb_smooth_loro."""

    def test_basic_loro_shape(self):
        """Output shape matches input shape."""
        space = _make_space_4d(5, 5, 5, 10)
        data = np.random.RandomState(42).randn(5, 5, 5, 10)
        vec = DenseNeuroVec(data, space)
        run_labels = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        result = cgb_smooth_loro(vec, fwhm=2.0, run_labels=run_labels)
        assert isinstance(result, DenseNeuroVec)
        assert result.shape == vec.shape

    def test_mismatched_labels_raises(self):
        """Raises when run_labels length != number of volumes."""
        space = _make_space_4d(3, 3, 3, 6)
        data = np.ones((3, 3, 3, 6))
        vec = DenseNeuroVec(data, space)
        with pytest.raises(ValueError, match="run_labels"):
            cgb_smooth_loro(vec, fwhm=2.0, run_labels=[1, 1, 2])

    def test_constant_data(self):
        """Constant data remains constant after LORO smoothing."""
        space = _make_space_4d(4, 4, 4, 6)
        data = np.full((4, 4, 4, 6), 5.0)
        vec = DenseNeuroVec(data, space)
        run_labels = np.array([1, 1, 1, 2, 2, 2])
        result = cgb_smooth_loro(vec, fwhm=2.0, run_labels=run_labels)
        np.testing.assert_allclose(result.data, 5.0, atol=1e-10)

    def test_with_mask(self):
        """LORO smoothing with mask works."""
        space = _make_space_4d(5, 5, 5, 4)
        data = np.random.RandomState(0).randn(5, 5, 5, 4)
        vec = DenseNeuroVec(data, space)
        mask = _make_mask(5, 5, 5)
        run_labels = np.array([1, 1, 2, 2])
        result = cgb_smooth_loro(vec, fwhm=2.0, run_labels=run_labels, mask=mask)
        assert result.shape == vec.shape


class TestCgbNuisance:
    """Tests for cgb_nuisance."""

    def test_perfect_confound_removal(self):
        """When data equals confound, residuals should be near zero."""
        space = _make_space_3d(3, 3, 3)
        confound_vals = np.arange(27, dtype=float)
        data_3d = confound_vals.reshape(3, 3, 3)
        vol = DenseNeuroVol(data_3d, space)
        graph = cgb_make_graph(space)
        confounds = confound_vals[:, np.newaxis]
        result = cgb_nuisance(vol, confounds, graph)
        assert isinstance(result, DenseNeuroVol)
        # Residuals should be close to zero
        np.testing.assert_allclose(result.data, 0.0, atol=1e-8)

    def test_output_shape(self):
        """Output shape matches input."""
        vol = _make_vol(4, 4, 4)
        graph = cgb_make_graph(vol)
        confounds = np.random.RandomState(0).randn(64, 2)
        result = cgb_nuisance(vol, confounds, graph)
        assert result.shape == vol.shape

    def test_confound_shape_mismatch(self):
        """Raises when confound rows != graph nodes."""
        vol = _make_vol(3, 3, 3)
        graph = cgb_make_graph(vol)
        with pytest.raises(ValueError, match="confounds"):
            cgb_nuisance(vol, np.ones((10, 2)), graph)

    def test_with_mask(self):
        """Nuisance regression with mask."""
        vol = _make_vol(5, 5, 5)
        mask = _make_mask(5, 5, 5)
        graph = cgb_make_graph(vol, mask=mask)
        n = graph.shape[0]
        confounds = np.random.RandomState(1).randn(n, 1)
        result = cgb_nuisance(vol, confounds, graph, mask=mask)
        assert result.shape == vol.shape


class TestLaplaceEnhance:
    """Tests for laplace_enhance."""

    def test_constant_unchanged(self):
        """Laplacian of constant field is zero, so output == input."""
        vol = _make_vol(5, 5, 5, value=7.0)
        result = laplace_enhance(vol, alpha=0.5)
        assert isinstance(result, DenseNeuroVol)
        np.testing.assert_allclose(result.data, 7.0, atol=1e-10)

    def test_enhancement_increases_contrast(self):
        """Enhancement should sharpen differences."""
        rng = np.random.RandomState(99)
        space = _make_space_3d(6, 6, 6)
        # Create a blob in the center
        data = np.zeros((6, 6, 6))
        data[2:4, 2:4, 2:4] = 10.0
        vol = DenseNeuroVol(data, space)
        result = laplace_enhance(vol, alpha=0.5)
        # Interior blob values should increase (sharpening)
        assert result.data[3, 3, 3] >= vol.data[3, 3, 3]

    def test_output_shape(self):
        """Output shape matches input."""
        vol = _make_vol(4, 4, 4)
        result = laplace_enhance(vol, alpha=1.0)
        assert result.shape == vol.shape

    def test_alpha_zero_is_identity(self):
        """Alpha=0 gives back the original data."""
        rng = np.random.RandomState(0)
        space = _make_space_3d(4, 4, 4)
        data = rng.randn(4, 4, 4)
        vol = DenseNeuroVol(data, space)
        result = laplace_enhance(vol, alpha=0.0)
        np.testing.assert_allclose(result.data, vol.data, atol=1e-12)

    def test_with_mask(self):
        """Enhancement with mask only modifies masked voxels."""
        vol = _make_vol(5, 5, 5, value=2.0)
        mask = _make_mask(5, 5, 5)
        result = laplace_enhance(vol, alpha=0.5, mask=mask)
        # Border voxels outside mask stay at 2.0
        assert result.data[0, 0, 0] == 2.0

    def test_preserves_space(self):
        """Output space matches input."""
        vol = _make_vol(4, 4, 4)
        result = laplace_enhance(vol, alpha=0.3)
        np.testing.assert_array_equal(result.space.dim, vol.space.dim)
