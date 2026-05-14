"""Tests for typed result objects from ROI/searchlight workflows.

Verifies that:
- Receipts are content-addressable (identical inputs -> identical hashes)
- SearchlightResult.values matches the prior bare-ndarray projection
- SearchlightResult.map_to_volume() roundtrips through to_nibabel preserving affine
- Pre-WP-3 back-compat shim (return_legacy=True) on searchlight() is honored
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroim import (
    DenseNeuroVol,
    LogicalNeuroVol,
    NeuroSpace,
    Receipt,
    SearchlightResult,
    hash_ndarray,
    hash_neurospace,
    searchlight_apply,
)
from neuroim.results import make_receipt


# -----------------------------------------------------------------------------
# Receipt determinism
# -----------------------------------------------------------------------------


def test_hash_ndarray_is_stable():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert hash_ndarray(arr) == hash_ndarray(arr.copy())


def test_hash_ndarray_differs_for_different_content():
    a = np.zeros((2, 2), dtype=np.float32)
    b = a.copy()
    b[0, 0] = 1.0
    assert hash_ndarray(a) != hash_ndarray(b)


def test_hash_ndarray_handles_none():
    assert hash_ndarray(None) == "none"


def test_hash_neurospace_is_stable():
    space_a = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    space_b = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    assert hash_neurospace(space_a) == hash_neurospace(space_b)


def test_hash_neurospace_differs_for_different_spacing():
    space_a = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    space_b = NeuroSpace(dim=(4, 4, 4), spacing=(3.0, 3.0, 3.0))
    assert hash_neurospace(space_a) != hash_neurospace(space_b)


def test_make_receipt_round_trip():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    mask = np.ones((4, 4, 4), dtype=bool)
    a = make_receipt(
        input_space=space,
        mask_data=mask,
        radius=3.0,
        n_voxels=12,
        method_name="mean",
        seed=42,
        source_affine=space.trans,
    )
    b = make_receipt(
        input_space=space,
        mask_data=mask.copy(),
        radius=3.0,
        n_voxels=12,
        method_name="mean",
        seed=42,
        source_affine=space.trans,
    )
    assert a == b


# -----------------------------------------------------------------------------
# Frozen-ness
# -----------------------------------------------------------------------------


def test_receipt_is_frozen():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    rec = make_receipt(input_space=space, mask_data=np.ones((4, 4, 4), dtype=bool))
    with pytest.raises(Exception):
        rec.radius = 99.0  # type: ignore[misc]


def test_searchlight_result_is_frozen():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    rec = make_receipt(input_space=space, mask_data=np.ones((4, 4, 4), dtype=bool))
    res = SearchlightResult(
        values=np.zeros(0),
        centers=np.zeros((0, 3), dtype=int),
        space=space,
        radius=3.0,
        shape="sphere",
        provenance=rec,
    )
    with pytest.raises(Exception):
        res.radius = 99.0  # type: ignore[misc]


# -----------------------------------------------------------------------------
# SearchlightResult.map_to_volume + numeric projection
# -----------------------------------------------------------------------------


def _make_mask_at_center(shape, center, radius=2):
    z, y, x = np.indices(shape)
    cz, cy, cx = center
    return (((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius**2).astype(bool)


def test_searchlight_legacy_vs_result_match_numerically():
    space = NeuroSpace(dim=(6, 6, 6), spacing=(2.0, 2.0, 2.0))
    mask_data = _make_mask_at_center((6, 6, 6), (3, 3, 3), radius=2)
    mask = LogicalNeuroVol(mask_data, space)

    def method(arr):
        return float(np.asarray(arr).sum())

    with pytest.warns(DeprecationWarning, match="return_legacy=True is deprecated"):
        legacy = searchlight_apply(mask, radius=2.0, method=method, return_legacy=True)
    result = searchlight_apply(mask, radius=2.0, method=method, return_legacy=False)

    assert isinstance(legacy, DenseNeuroVol)
    assert isinstance(result, SearchlightResult)

    legacy_data = np.asarray(legacy.data)
    projected = result.map_to_volume()
    np.testing.assert_allclose(
        np.asarray(projected.data), legacy_data, equal_nan=True
    )


def test_searchlight_result_centers_align_with_finite_values():
    space = NeuroSpace(dim=(6, 6, 6), spacing=(2.0, 2.0, 2.0))
    mask_data = _make_mask_at_center((6, 6, 6), (3, 3, 3), radius=2)
    mask = LogicalNeuroVol(mask_data, space)

    result = searchlight_apply(
        mask, radius=2.0, method=lambda a: float(np.asarray(a).mean()), return_legacy=False
    )

    assert result.centers.shape[1] == 3
    assert result.centers.shape[0] == result.values.shape[0]
    assert result.shape == "sphere"
    assert result.provenance.n_voxels == result.values.shape[0]
    assert result.method_name  # not empty


def test_searchlight_result_to_nibabel_preserves_affine():
    pytest.importorskip("nibabel")
    space = NeuroSpace(dim=(5, 5, 5), spacing=(2.0, 2.0, 2.0))
    mask = LogicalNeuroVol(
        _make_mask_at_center((5, 5, 5), (2, 2, 2), radius=1), space
    )
    result = searchlight_apply(
        mask, radius=2.0, method=lambda a: 1.0, return_legacy=False
    )
    img = result.to_nibabel()
    np.testing.assert_allclose(img.affine, space.trans[:4, :4])


def test_searchlight_default_returns_searchlight_result():
    """ME-1: default flipped. The Python-native typed result is the default;
    the legacy ndarray-projection requires an explicit opt-in under
    DeprecationWarning."""
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    mask = LogicalNeuroVol(
        _make_mask_at_center((4, 4, 4), (2, 2, 2), radius=1), space
    )
    out = searchlight_apply(mask, radius=2.0, method=lambda a: 0.0)
    assert isinstance(out, SearchlightResult)


def test_searchlight_explicit_legacy_still_returns_densevol_with_warning():
    """Back-compat is preserved when the caller explicitly opts in; the
    DeprecationWarning is the migration signal."""
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    mask = LogicalNeuroVol(
        _make_mask_at_center((4, 4, 4), (2, 2, 2), radius=1), space
    )
    with pytest.warns(DeprecationWarning, match="return_legacy=True is deprecated"):
        out = searchlight_apply(
            mask, radius=2.0, method=lambda a: 0.0, return_legacy=True
        )
    assert isinstance(out, DenseNeuroVol)


# -----------------------------------------------------------------------------
# ROIExtractionResult — series_roi numeric projection + provenance (WP-4 phase 2)
# -----------------------------------------------------------------------------


def _make_roi_extraction_fixture():
    from neuroim import DenseNeuroVec, spherical_roi

    space_4d = NeuroSpace(dim=(8, 8, 8, 5), spacing=(2.0, 2.0, 2.0, 1.0))
    rng = np.random.default_rng(seed=7)
    data = rng.standard_normal((8, 8, 8, 5)).astype(np.float32)
    vec = DenseNeuroVec(data, space_4d)

    mask_data = np.ones((8, 8, 8), dtype=bool)
    mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=(8, 8, 8), spacing=(2.0, 2.0, 2.0)))
    roi = spherical_roi(mask, centroid=(4, 4, 4), radius=2)
    return vec, roi


def test_series_roi_default_returns_roi_extraction_result():
    """ME-1: typed ROIExtractionResult is the default surface; the legacy
    ndarray projection requires an explicit opt-in."""
    from neuroim.results import ROIExtractionResult

    vec, roi = _make_roi_extraction_fixture()
    out = vec.series_roi(roi)
    assert isinstance(out, ROIExtractionResult)
    assert out.values.shape == (5, len(roi))


def test_series_roi_explicit_legacy_returns_ndarray_with_warning():
    vec, roi = _make_roi_extraction_fixture()
    with pytest.warns(DeprecationWarning, match="return_legacy=True is deprecated"):
        out = vec.series_roi(roi, return_legacy=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, len(roi))


def test_series_roi_typed_result_values_match_legacy():
    from neuroim.results import ROIExtractionResult

    vec, roi = _make_roi_extraction_fixture()
    with pytest.warns(DeprecationWarning):
        legacy = vec.series_roi(roi, return_legacy=True)
    typed = vec.series_roi(roi, return_legacy=False)
    assert isinstance(typed, ROIExtractionResult)
    np.testing.assert_array_equal(typed.values, legacy)


def test_series_roi_result_carries_coords_space_provenance():
    from neuroim.results import Receipt, ROIExtractionResult

    vec, roi = _make_roi_extraction_fixture()
    res = vec.series_roi(roi, return_legacy=False)
    assert isinstance(res, ROIExtractionResult)
    np.testing.assert_array_equal(res.coords, roi.coords)
    assert res.space is not None
    assert isinstance(res.provenance, Receipt)
    assert res.provenance.method_name == "series_roi"
    assert res.provenance.n_voxels == len(roi)
    assert res.mask_hash == res.provenance.mask_hash
    assert res.mask_hash != "none"


def test_series_roi_result_n_properties():
    vec, roi = _make_roi_extraction_fixture()
    res = vec.series_roi(roi, return_legacy=False)
    assert res.n_voxels == len(roi)
    assert res.n_timepoints == 5


def test_roi_extraction_result_is_frozen():
    vec, roi = _make_roi_extraction_fixture()
    res = vec.series_roi(roi, return_legacy=False)
    with pytest.raises(Exception):
        res.values = np.zeros(1)  # type: ignore[misc]
