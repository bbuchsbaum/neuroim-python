"""Tests for ROI shape factory functions."""

import pytest
import numpy as np

from neuroimpy import NeuroSpace, DenseNeuroVol
from neuroimpy.roi import (
    cube_roi, ellipsoid_roi, patch_set,
    spherical_roi, cuboid_roi, spherical_roi_set,
)


@pytest.fixture
def bvol():
    """Create a simple 20x20x20 DenseNeuroVol for testing."""
    space = NeuroSpace([20, 20, 20])
    data = np.ones((20, 20, 20), dtype=np.float64)
    return DenseNeuroVol(data, space)


# ---- cube_roi ----

class TestCubeRoi:

    def test_basic(self, bvol):
        roi = cube_roi(bvol, [10, 10, 10], width=2)
        # A cube with half-width 2 -> side length 5 -> 125 voxels (if not clipped)
        assert len(roi) == 5 ** 3

    def test_fills(self, bvol):
        roi = cube_roi(bvol, [10, 10, 10], width=1, fill=42.0)
        assert np.all(roi.data == 42.0)

    def test_at_edge(self, bvol):
        """Cube at edge should be clipped by volume bounds."""
        roi = cube_roi(bvol, [0, 0, 0], width=3)
        # Only the positive octant is available: 0..3 in each dim = 4 voxels each
        assert len(roi) == 4 ** 3

    def test_nonzero(self):
        """nonzero=True drops zero-valued voxels."""
        space = NeuroSpace([10, 10, 10])
        data = np.zeros((10, 10, 10), dtype=np.float64)
        data[5, 5, 5] = 1.0
        bvol = DenseNeuroVol(data, space)
        roi = cube_roi(bvol, [5, 5, 5], width=2, nonzero=True)
        assert len(roi) == 1
        assert roi.data[0] == 1.0

    def test_type_error(self):
        """Should reject non-NeuroVol input."""
        with pytest.raises(TypeError):
            cube_roi("not a volume", [5, 5, 5], width=2)


# ---- ellipsoid_roi ----

class TestEllipsoidRoi:

    def test_sphere_case(self, bvol):
        """With equal radii, ellipsoid should match spherical_roi."""
        r = 3.0
        e_roi = ellipsoid_roi(bvol, [10, 10, 10], radii=[r, r, r], fill=1.0)
        s_roi = spherical_roi(bvol, [10, 10, 10], radius=r, fill=1.0)
        assert len(e_roi) == len(s_roi)

    def test_axis_asymmetry(self, bvol):
        """Different radii should produce an asymmetric ROI."""
        roi = ellipsoid_roi(bvol, [10, 10, 10], radii=[4, 2, 1], fill=1.0)
        coords = roi.coords
        # Extent along axis 0 should be larger than along axis 2
        extent_i = coords[:, 0].max() - coords[:, 0].min()
        extent_k = coords[:, 2].max() - coords[:, 2].min()
        assert extent_i > extent_k

    def test_invalid_radii_length(self, bvol):
        with pytest.raises(ValueError, match="radii must have length 3"):
            ellipsoid_roi(bvol, [10, 10, 10], radii=[3, 3])

    def test_negative_radii(self, bvol):
        with pytest.raises(ValueError, match="radii must be positive"):
            ellipsoid_roi(bvol, [10, 10, 10], radii=[3, -1, 3])

    def test_fills(self, bvol):
        roi = ellipsoid_roi(bvol, [10, 10, 10], radii=[2, 2, 2], fill=7.0)
        assert np.all(roi.data == 7.0)

    def test_nonzero(self):
        space = NeuroSpace([10, 10, 10])
        data = np.zeros((10, 10, 10), dtype=np.float64)
        data[5, 5, 5] = 1.0
        bvol = DenseNeuroVol(data, space)
        roi = ellipsoid_roi(bvol, [5, 5, 5], radii=[2, 2, 2], nonzero=True)
        assert len(roi) == 1

    def test_type_error(self):
        with pytest.raises(TypeError):
            ellipsoid_roi("not a vol", [5, 5, 5], radii=[2, 2, 2])


# ---- patch_set ----

class TestPatchSet:

    def test_sphere_patches(self, bvol):
        centers = np.array([[5, 5, 5], [10, 10, 10], [15, 15, 15]])
        patches = patch_set(bvol, centers, radius=2.0, shape="sphere", fill=1.0)
        assert len(patches) == 3
        for p in patches:
            assert len(p) > 0

    def test_cube_patches(self, bvol):
        centers = np.array([[5, 5, 5], [10, 10, 10]])
        patches = patch_set(bvol, centers, radius=2, shape="cube", fill=1.0)
        assert len(patches) == 2

    def test_invalid_shape(self, bvol):
        with pytest.raises(ValueError, match="shape must be"):
            patch_set(bvol, np.array([[5, 5, 5]]), radius=2, shape="triangle")

    def test_fill_list(self, bvol):
        centers = np.array([[5, 5, 5], [10, 10, 10]])
        patches = patch_set(bvol, centers, radius=2, shape="sphere",
                            fill=[3.0, 7.0])
        assert np.all(patches[0].data == 3.0)
        assert np.all(patches[1].data == 7.0)

    def test_fill_list_mismatch(self, bvol):
        centers = np.array([[5, 5, 5], [10, 10, 10]])
        with pytest.raises(ValueError, match="fill must be scalar or match"):
            patch_set(bvol, centers, radius=2, fill=[1.0, 2.0, 3.0])

    def test_bad_centroids(self, bvol):
        with pytest.raises(ValueError, match="3 columns"):
            patch_set(bvol, np.array([[5, 5]]), radius=2)


# ---- spherical_roi_set (existing, but test shape-factory integration) ----

class TestSphericalRoiSet:

    def test_basic(self, bvol):
        centers = np.array([[5, 5, 5], [10, 10, 10]])
        rois = spherical_roi_set(bvol, centers, radius=3.0, fill=1.0)
        assert len(rois) == 2
        for r in rois:
            assert len(r) > 0
