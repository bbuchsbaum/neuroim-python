"""Conformance test matrix for the implicit VoxelSeriesStore contract.

WP-8 (bd-01KRKEAGDVXH2HS6AD2RETZA25). Prerequisite for WP-9
(bd-01KRKEBYZ5VD29JHGXGMSFH0GZ): the protocol that WP-9 introduces will be the
post-hoc specification of the behaviour proven here.

This file does NOT refactor anything. It documents what every backend does
when run against the same fixture. Divergences found by WP-8 are now part of
the green WP-9 contract.

See `tests/conformance/SERIES_CONTRACT.md` for the prose contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroim.neuro_vol import DenseNeuroVol


# -----------------------------------------------------------------------------
# Shape & dtype
# -----------------------------------------------------------------------------


def test_shape(backend, fixture_array):
    assert backend.shape == fixture_array.shape


def test_dtype_is_real_floating(backend):
    dtype = backend.dtype
    assert np.issubdtype(dtype, np.floating), f"expected floating dtype, got {dtype!r}"


# -----------------------------------------------------------------------------
# series(x, y, z) — single voxel by 3D coordinate -> (T,) 1-D
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("xyz", [(0, 0, 0), (3, 5, 1), (7, 7, 3)])
def test_series_xyz_shape_and_values(backend, fixture_array, xyz):
    x, y, z = xyz
    series = backend.series(x, y, z)
    assert series.ndim == 1
    assert series.shape == (fixture_array.shape[-1],)
    expected = fixture_array[x, y, z, :]
    np.testing.assert_allclose(series, expected, rtol=1e-5, atol=1e-6)


# -----------------------------------------------------------------------------
# series(Nx3 ndarray) — many voxels by coordinate matrix -> (T, N) 2-D
# -----------------------------------------------------------------------------


def test_series_nx3_returns_time_by_voxels(backend, fixture_array):
    coords = np.array([[0, 0, 0], [1, 2, 3], [4, 4, 2]], dtype=int)
    result = backend.series(coords)
    T = fixture_array.shape[-1]
    N = coords.shape[0]
    assert result.shape == (T, N), f"expected (T={T}, N={N}); got {result.shape}"
    expected = fixture_array[coords[:, 0], coords[:, 1], coords[:, 2], :].T
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_series_nx3_out_of_bounds_returns_zero_columns(backend, fixture_array):
    coords = np.array([[0, 0, 0], [99, 99, 99]], dtype=int)
    result = backend.series(coords)
    T = fixture_array.shape[-1]
    assert result.shape == (T, 2)
    np.testing.assert_allclose(result[:, 0], fixture_array[0, 0, 0, :], rtol=1e-5)
    np.testing.assert_array_equal(result[:, 1], np.zeros(T))


# -----------------------------------------------------------------------------
# series(int linear index) and series(1-D ndarray of linear indices)
# -----------------------------------------------------------------------------


def test_series_single_linear_index(backend, fixture_array):
    voxel = (2, 3, 1)
    linear = int(np.ravel_multi_index(voxel, fixture_array.shape[:3], order="F"))
    series = backend.series(linear)
    assert series.ndim == 1
    np.testing.assert_allclose(
        series, fixture_array[voxel[0], voxel[1], voxel[2], :], rtol=1e-5
    )


def test_series_linear_indices_returns_time_by_voxels(backend, fixture_array):
    voxels = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    linear = np.array(
        [np.ravel_multi_index(v, fixture_array.shape[:3], order="F") for v in voxels],
        dtype=int,
    )
    result = backend.series(linear)
    T = fixture_array.shape[-1]
    N = len(voxels)
    assert result.shape == (T, N)
    expected = np.stack([fixture_array[v[0], v[1], v[2], :] for v in voxels], axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


# -----------------------------------------------------------------------------
# __getitem__ — pointwise and volume slicing
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("xyzt", [(0, 0, 0, 0), (3, 5, 1, 7), (7, 7, 3, 9)])
def test_getitem_pointwise(backend, fixture_array, xyzt):
    value = backend[xyzt]
    np.testing.assert_allclose(float(value), fixture_array[xyzt], rtol=1e-5, atol=1e-6)


def test_getitem_volume_at_time(backend, fixture_array):
    """All backends currently return a 3-D ndarray from `[:, :, :, t]`.

    Wrapping in DenseNeuroVol is a documented future improvement (WP-9), not
    part of the current contract. This test asserts only the shape and values.
    """
    vol = backend[:, :, :, 0]
    arr = vol.data if isinstance(vol, DenseNeuroVol) else np.asarray(vol)
    assert arr.shape == fixture_array.shape[:3]
    np.testing.assert_allclose(arr, fixture_array[..., 0], rtol=1e-5, atol=1e-6)


# -----------------------------------------------------------------------------
# series_3d alias
# -----------------------------------------------------------------------------


def test_series_3d_matches_series_xyz(backend, fixture_array):
    np.testing.assert_allclose(
        backend.series_3d(2, 2, 1), backend.series(2, 2, 1), rtol=1e-5
    )
