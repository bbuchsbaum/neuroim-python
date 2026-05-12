"""Tests for searchlight variant functions (resampled and cluster-based)."""

import pytest
import numpy as np
from neuroim import (
    NeuroSpace, LogicalNeuroVol, DenseNeuroVol, DenseNeuroVec,
    ClusteredNeuroVol,
)
from neuroim.searchlight_high_level import (
    resampled_searchlight,
    cluster_searchlight_series,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vec(shape=(8, 8, 8), n_time=20, seed=0):
    """Create a small DenseNeuroVec for testing."""
    rng = np.random.default_rng(seed)
    space = NeuroSpace(dim=[*shape, n_time], spacing=[2, 2, 2, 1])
    data = rng.standard_normal((*shape, n_time))
    return DenseNeuroVec(data, space)


def _make_mask(shape=(8, 8, 8)):
    """Create a cubic mask in the center of the volume."""
    space = NeuroSpace(dim=list(shape), spacing=[2, 2, 2])
    mask_data = np.zeros(shape, dtype=bool)
    s = [slice(s // 4, 3 * s // 4) for s in shape]
    mask_data[tuple(s)] = True
    return LogicalNeuroVol(mask_data, space)


# ---------------------------------------------------------------------------
# resampled_searchlight tests
# ---------------------------------------------------------------------------


class TestResampledSearchlight:

    def test_returns_dense_vol(self):
        vec = _make_vec()
        mask = _make_mask()
        result = resampled_searchlight(
            vec, radius=4, fun=lambda x: np.mean(x), n_resamples=5,
            mask=mask, seed=42,
        )
        assert isinstance(result, DenseNeuroVol)
        assert result.shape == mask.shape

    def test_result_has_values_in_mask(self):
        vec = _make_vec()
        mask = _make_mask()
        result = resampled_searchlight(
            vec, radius=4, fun=lambda x: np.mean(x), n_resamples=5,
            mask=mask, seed=42,
        )
        # Voxels inside the mask should be finite
        mask_coords = np.argwhere(mask.data)
        result_vals = result.data[tuple(mask_coords.T)]
        # At least some should be non-NaN (nonzero mask voxels that are
        # also centres)
        assert np.any(np.isfinite(result_vals))

    def test_seed_reproducibility(self):
        vec = _make_vec()
        mask = _make_mask()
        a = resampled_searchlight(
            vec, radius=4, fun=lambda x: np.mean(x), n_resamples=10,
            mask=mask, seed=99,
        )
        b = resampled_searchlight(
            vec, radius=4, fun=lambda x: np.mean(x), n_resamples=10,
            mask=mask, seed=99,
        )
        np.testing.assert_array_equal(a.data, b.data)

    def test_different_seeds_differ(self):
        vec = _make_vec()
        mask = _make_mask()
        a = resampled_searchlight(
            vec, radius=4, fun=lambda x: np.mean(x), n_resamples=10,
            mask=mask, seed=1,
        )
        b = resampled_searchlight(
            vec, radius=4, fun=lambda x: np.mean(x), n_resamples=10,
            mask=mask, seed=2,
        )
        # Not guaranteed to differ everywhere, but should overall
        assert not np.array_equal(a.data, b.data)

    def test_auto_mask(self):
        """When mask=None, it should derive one from the first volume."""
        vec = _make_vec()
        result = resampled_searchlight(
            vec, radius=6, fun=lambda x: np.std(x), n_resamples=3,
            seed=42,
        )
        assert isinstance(result, DenseNeuroVol)

    def test_custom_function(self):
        vec = _make_vec()
        mask = _make_mask()
        # A function that returns a constant should give that constant
        result = resampled_searchlight(
            vec, radius=4, fun=lambda x: 42.0, n_resamples=5,
            mask=mask, seed=0,
        )
        mask_coords = np.argwhere(mask.data)
        vals = result.data[tuple(mask_coords.T)]
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) > 0:
            np.testing.assert_allclose(finite_vals, 42.0)


# ---------------------------------------------------------------------------
# cluster_searchlight_series tests
# ---------------------------------------------------------------------------


class TestClusterSearchlightSeries:

    def _make_cvol(self, shape=(8, 8, 8), n_clusters=3):
        """Build a simple ClusteredNeuroVol."""
        space = NeuroSpace(dim=list(shape), spacing=[2, 2, 2])
        mask_data = np.ones(shape, dtype=bool)
        mask = LogicalNeuroVol(mask_data, space)

        # Assign cluster labels based on x-coordinate slices
        n_voxels = int(np.sum(mask_data))
        clusters = np.zeros(n_voxels, dtype=int)
        flat_idx = 0
        slice_size = shape[0] // n_clusters
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if mask_data[i, j, k]:
                        clusters[flat_idx] = min(i // max(1, slice_size), n_clusters - 1)
                        flat_idx += 1

        return ClusteredNeuroVol(mask, clusters)

    def test_returns_dict(self):
        cvol = self._make_cvol()
        vec = _make_vec()
        result = cluster_searchlight_series(vec, cvol, fun=lambda x: np.mean(x))
        assert isinstance(result, dict)

    def test_all_clusters_present(self):
        n_clusters = 3
        cvol = self._make_cvol(n_clusters=n_clusters)
        vec = _make_vec()
        result = cluster_searchlight_series(vec, cvol, fun=lambda x: np.mean(x))
        assert len(result) == n_clusters

    def test_function_receives_correct_shape(self):
        """Verify that *fun* receives a 2-D (time x voxels) array."""
        cvol = self._make_cvol(n_clusters=2)
        vec = _make_vec(n_time=15)
        shapes_seen = []

        def capture_shape(x):
            shapes_seen.append(x.shape)
            return 0.0

        cluster_searchlight_series(vec, cvol, fun=capture_shape)

        for s in shapes_seen:
            # First dim should be time
            assert s[0] == 15

    def test_constant_function(self):
        cvol = self._make_cvol(n_clusters=4)
        vec = _make_vec()
        result = cluster_searchlight_series(vec, cvol, fun=lambda x: 7.0)
        for v in result.values():
            assert v == 7.0

    def test_scalar_result(self):
        cvol = self._make_cvol()
        vec = _make_vec()
        result = cluster_searchlight_series(vec, cvol, fun=lambda x: float(np.std(x)))
        for v in result.values():
            assert isinstance(v, float)
            assert np.isfinite(v)
