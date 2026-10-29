"""Cross-builder regression tests for ROI geometry and membership contracts."""

from collections.abc import Callable

import numpy as np
import pytest

from neuroim import DenseNeuroVol, NeuroSpace
from neuroim.roi import (
    cuboid_roi,
    ellipsoid_roi,
    patch_set,
    spherical_roi,
    spherical_roi_set,
    square_roi,
)

Builder = Callable[..., object]


def _square(source, centroid, **kwargs):
    return square_roi(source, centroid, surround=1, fixdim=2, **kwargs)


def _cuboid(source, centroid, **kwargs):
    return cuboid_roi(source, centroid, surround=1, **kwargs)


def _sphere(source, centroid, **kwargs):
    return spherical_roi(source, centroid, radius=1.0, **kwargs)


def _ellipsoid(source, centroid, **kwargs):
    return ellipsoid_roi(source, centroid, radii=(1.0, 1.0, 1.0), **kwargs)


BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("square", _square),
    ("cuboid", _cuboid),
    ("sphere", _sphere),
    ("ellipsoid", _ellipsoid),
)


@pytest.fixture
def hostile_volume() -> DenseNeuroVol:
    data = np.zeros((5, 5, 5), dtype=float)
    data[2, 2, 2] = 2.0
    data[3, 2, 2] = -3.0
    data[1, 2, 2] = np.nan
    data[2, 3, 2] = np.inf
    data[2, 1, 2] = -np.inf
    return DenseNeuroVol(data, NeuroSpace(data.shape))


@pytest.mark.parametrize("_name,builder", BUILDERS, ids=[name for name, _ in BUILDERS])
@pytest.mark.parametrize("fill", [None, 9.0], ids=["source-values", "fill"])
def test_nonzero_means_finite_and_nonzero_even_with_fill(
    _name: str, builder: Builder, fill: float | None, hostile_volume: DenseNeuroVol
) -> None:
    roi = builder(hostile_volume, (2, 2, 2), nonzero=True, fill=fill)

    assert {tuple(coord) for coord in roi.coords} == {(2, 2, 2), (3, 2, 2)}
    if fill is None:
        np.testing.assert_array_equal(np.sort(roi.data), [-3.0, 2.0])
    else:
        np.testing.assert_array_equal(roi.data, [9.0, 9.0])


@pytest.mark.parametrize("_name,builder", BUILDERS, ids=[name for name, _ in BUILDERS])
@pytest.mark.parametrize(
    "centroid",
    [
        (np.nan, 2, 2),
        (np.inf, 2, 2),
        (-np.inf, 2, 2),
        (-1, 2, 2),
        (5, 2, 2),
    ],
)
def test_all_builders_reject_nonfinite_and_out_of_bounds_centroids(
    _name: str,
    builder: Builder,
    centroid: tuple[float, int, int],
    hostile_volume: DenseNeuroVol,
) -> None:
    with pytest.raises(ValueError, match="centroid"):
        builder(hostile_volume, centroid)


@pytest.mark.parametrize("radius", [np.nan, np.inf, -np.inf, 0.0, -1.0, [1.0]])
def test_spherical_builders_reject_invalid_radius(
    radius: object, hostile_volume: DenseNeuroVol
) -> None:
    with pytest.raises(ValueError, match="radius"):
        spherical_roi(hostile_volume, (2, 2, 2), radius=radius)
    with pytest.raises(ValueError, match="radius"):
        spherical_roi_set(hostile_volume, [[2, 2, 2]], radius=radius)
    with pytest.raises(ValueError, match="radius"):
        patch_set(hostile_volume, [[2, 2, 2]], radius=radius, shape="sphere")


@pytest.mark.parametrize(
    "radii",
    [
        (np.nan, 1, 1),
        (np.inf, 1, 1),
        (-np.inf, 1, 1),
        (0, 1, 1),
        (-1, 1, 1),
    ],
)
def test_ellipsoid_rejects_nonfinite_or_nonpositive_radii(
    radii: tuple[float, int, int], hostile_volume: DenseNeuroVol
) -> None:
    with pytest.raises(ValueError, match="radii must be positive and finite"):
        ellipsoid_roi(hostile_volume, (2, 2, 2), radii=radii)


@pytest.mark.parametrize(
    "surround",
    [np.nan, np.inf, -np.inf, -1, 1.5, (1, 1), (1, 1, np.nan)],
)
def test_box_builders_reject_invalid_surround(
    surround: object, hostile_volume: DenseNeuroVol
) -> None:
    with pytest.raises(ValueError, match="surround"):
        cuboid_roi(hostile_volume, (2, 2, 2), surround=surround)
    with pytest.raises(ValueError, match="surround"):
        square_roi(hostile_volume, (2, 2, 2), surround=surround)


def test_square_default_fixes_third_axis(hostile_volume: DenseNeuroVol) -> None:
    roi = square_roi(hostile_volume, (2, 2, 2), surround=1)

    assert len(roi) == 9
    np.testing.assert_array_equal(np.unique(roi.coords[:, 2]), [2])


@pytest.mark.parametrize("_name,builder", BUILDERS, ids=[name for name, _ in BUILDERS])
def test_builders_accept_neurospace_as_geometry_source(
    _name: str, builder: Builder
) -> None:
    space = NeuroSpace((5, 5, 5))

    roi = builder(space, (2, 2, 2), nonzero=True)

    assert len(roi) > 0
    assert roi.space is space
    np.testing.assert_array_equal(roi.data, np.ones(len(roi)))
