"""ROI constructors accept structural volume/mask protocols."""

import numpy as np

from neuroim import NeuroSpace
from neuroim.roi import cuboid_roi, ellipsoid_roi, spherical_roi, square_roi
from neuroim.protocols import MaskLike, NeuroVolLike


class MinimalVolume:
    def __init__(self):
        self.space = NeuroSpace(dim=(5, 5, 5))
        self.data = np.zeros((5, 5, 5), dtype=np.float32)
        self.data[2, 2, 2] = 7.0
        self.data[3, 2, 2] = 3.0

    @property
    def shape(self):
        return self.data.shape


class MinimalMask(MinimalVolume):
    def as_logical(self):
        return self


def test_minimal_volume_satisfies_neurovol_like():
    vol = MinimalVolume()

    assert isinstance(vol, NeuroVolLike)


def test_minimal_mask_satisfies_mask_like():
    mask = MinimalMask()

    assert isinstance(mask, MaskLike)


def test_spherical_roi_accepts_structural_volume():
    vol = MinimalVolume()

    roi = spherical_roi(vol, centroid=(2, 2, 2), radius=1.0, nonzero=True)

    np.testing.assert_array_equal(roi.data, [7.0, 3.0])
    assert roi.space is vol.space


def test_cuboid_and_square_roi_accept_structural_volume():
    vol = MinimalVolume()

    cuboid = cuboid_roi(vol, centroid=(2, 2, 2), surround=1, nonzero=True)
    square = square_roi(vol, centroid=(2, 2, 2), surround=1, fixdim=0, nonzero=True)

    np.testing.assert_array_equal(cuboid.data, [7.0, 3.0])
    np.testing.assert_array_equal(square.data, [7.0])
    assert cuboid.space is vol.space
    assert square.space is vol.space


def test_ellipsoid_roi_accepts_structural_volume():
    vol = MinimalVolume()

    roi = ellipsoid_roi(vol, centroid=(2, 2, 2), radii=(1, 1, 1), nonzero=True)

    np.testing.assert_array_equal(roi.data, [7.0, 3.0])
    assert roi.space is vol.space
