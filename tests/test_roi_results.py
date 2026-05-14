"""ROIExtractionResult wiring for ROI extraction workflows."""

import numpy as np

from neuroim import (
    DenseNeuroVec,
    DenseNeuroVol,
    NeuroSpace,
    ROIExtractionResult,
    ROICoords,
    ROIVol,
    hash_ndarray,
    hash_neurospace,
)
from neuroim.roi import series_roi, values_roi


def _make_vec():
    space = NeuroSpace(dim=(4, 4, 3, 3))
    data = np.zeros(tuple(space.dim), dtype=np.float32)
    data[1, 1, 1, :] = [1, 2, 3]
    data[2, 1, 1, :] = [4, 5, 6]
    return DenseNeuroVec(data, space)


def _make_roi():
    space = NeuroSpace(dim=(4, 4, 3))
    coords = np.array([[1, 1, 1], [2, 1, 1]], dtype=int)
    return ROICoords(coords, space)


def test_neurovec_series_roi_result_matches_legacy_values():
    vec = _make_vec()
    roi = _make_roi()

    legacy = vec.series_roi(roi)
    result = vec.series_roi(roi, return_legacy=False)

    assert isinstance(result, ROIExtractionResult)
    np.testing.assert_array_equal(result.values, legacy)
    np.testing.assert_array_equal(result.coords, roi.coords)
    assert result.space is roi.space
    assert result.n_voxels == 2
    assert result.n_timepoints == 3


def test_neurovec_series_roi_result_provenance_is_populated():
    vec = _make_vec()
    roi = _make_roi()

    result = vec.series_roi(roi, return_legacy=False)

    assert result.mask_hash == hash_ndarray(roi.coords)
    assert result.provenance.mask_hash == result.mask_hash
    assert result.provenance.input_space_hash == hash_neurospace(vec.space)
    assert result.provenance.source_affine_hash == hash_ndarray(vec.space.trans)
    assert result.provenance.method_name == "series_roi"
    assert result.provenance.n_voxels == result.n_voxels


def test_neurovec_series_roi_accepts_roivol_result_projection():
    vec = _make_vec()
    roi = _make_roi()
    roi_vol = ROIVol(np.array([10.0, 20.0]), roi.space, roi.coords)

    legacy = vec.series_roi(roi_vol)
    result = vec.series_roi(roi_vol, return_legacy=False)

    np.testing.assert_array_equal(result.values, legacy)
    np.testing.assert_array_equal(result.coords, roi.coords)
    assert result.space is roi.space


def test_roi_module_series_roi_dispatches_to_result_object():
    vec = _make_vec()
    roi = _make_roi()

    result = series_roi(vec, roi, return_legacy=False)

    assert isinstance(result, ROIExtractionResult)
    np.testing.assert_array_equal(result.values, vec.series_roi(roi))


def test_values_roi_result_matches_volume_values():
    space = NeuroSpace(dim=(4, 4, 3))
    shape = tuple(space.dim)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    vol = DenseNeuroVol(data, space)
    roi = _make_roi()

    legacy = values_roi(vol, roi)
    result = values_roi(vol, roi, return_legacy=False)

    assert isinstance(result, ROIExtractionResult)
    np.testing.assert_array_equal(result.values, legacy)
    np.testing.assert_array_equal(result.values, data[tuple(roi.coords.T)])
    assert result.n_timepoints is None
    assert result.space is roi.space
    assert result.provenance.method_name == "values_roi"
