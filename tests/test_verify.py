"""ME-2: Composable receipts + verifier.

Tests for ``Receipt.diff``, ``Receipt.merge``, ``result.require_compatible``,
and the top-level ``neuroim.verify`` helpers.

The decisive test (``test_verifier_catches_silent_space_mismatch``)
demonstrates the mission claim: a bare-ndarray pipeline silently produces
wrong numbers when fed mismatched spaces; the typed-result pipeline with
the verifier raises with a Receipt diff.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroim import (
    DenseNeuroVec,
    LogicalNeuroVol,
    NeuroSpace,
    Receipt,
    SearchlightResult,
    spherical_roi,
    searchlight_apply,
    verify,
)
from neuroim.results import ROIExtractionResult, make_receipt


# ----------------------------------------------------------------------
# Receipt.diff
# ----------------------------------------------------------------------


def _receipt(**over):
    base = dict(
        input_space_hash="aaa",
        mask_hash="bbb",
        radius=3.0,
        n_voxels=10,
        method_name="m",
        seed=42,
        neuroim_version="0.1.0",
        source_affine_hash="ccc",
    )
    base.update(over)
    return Receipt(**base)


def test_receipt_diff_empty_when_equal():
    a = _receipt()
    b = _receipt()
    assert a.diff(b) == {}


def test_receipt_diff_reports_only_differing_fields():
    a = _receipt()
    b = _receipt(mask_hash="ddd", seed=7)
    d = a.diff(b)
    assert set(d.keys()) == {"mask_hash", "seed"}
    assert d["mask_hash"] == ("bbb", "ddd")
    assert d["seed"] == (42, 7)


# ----------------------------------------------------------------------
# Receipt.merge
# ----------------------------------------------------------------------


def test_receipt_merge_preserves_space_and_mask_when_agreed():
    a = _receipt(n_voxels=10, method_name="m1")
    b = _receipt(n_voxels=20, method_name="m2")
    merged = a.merge(b)
    assert merged.input_space_hash == a.input_space_hash
    assert merged.mask_hash == a.mask_hash
    assert merged.n_voxels == 20  # max
    assert merged.method_name == "m1+m2"


def test_receipt_merge_is_deterministic():
    a = _receipt(method_name="foo")
    b = _receipt(method_name="bar")
    assert a.merge(b) == a.merge(b)


def test_receipt_merge_raises_on_space_mismatch():
    a = _receipt(input_space_hash="space-A")
    b = _receipt(input_space_hash="space-B")
    with pytest.raises(ValueError, match="input_space_hash"):
        a.merge(b)


def test_receipt_merge_raises_on_mask_mismatch():
    a = _receipt(mask_hash="mask-A")
    b = _receipt(mask_hash="mask-B")
    with pytest.raises(ValueError, match="mask_hash"):
        a.merge(b)


# ----------------------------------------------------------------------
# result.require_compatible
# ----------------------------------------------------------------------


def _make_searchlight_result(space):
    rec = make_receipt(input_space=space, mask_data=np.ones((4, 4, 4), dtype=bool))
    return SearchlightResult(
        values=np.zeros(0),
        centers=np.zeros((0, 3), dtype=int),
        space=space,
        radius=3.0,
        shape="sphere",
        provenance=rec,
    )


def test_require_compatible_accepts_matching_result():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    a = _make_searchlight_result(space)
    b = _make_searchlight_result(space)
    a.require_compatible(b)  # no raise


def test_require_compatible_accepts_matching_neurospace():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    a = _make_searchlight_result(space)
    a.require_compatible(space)  # no raise


def test_require_compatible_raises_on_space_mismatch():
    space_a = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    space_b = NeuroSpace(dim=(4, 4, 4), spacing=(3.0, 3.0, 3.0))
    a = _make_searchlight_result(space_a)
    b = _make_searchlight_result(space_b)
    with pytest.raises(ValueError, match="space"):
        a.require_compatible(b)


def test_require_compatible_on_roi_extraction_result():
    space_a = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    space_b = NeuroSpace(dim=(4, 4, 4), spacing=(3.0, 3.0, 3.0))
    rec_a = make_receipt(input_space=space_a, mask_data=np.ones((4, 4, 4), dtype=bool))
    res_a = ROIExtractionResult(
        values=np.zeros((1, 1)), coords=np.zeros((1, 3), dtype=int),
        space=space_a, mask_hash=rec_a.mask_hash, provenance=rec_a,
    )
    rec_b = make_receipt(input_space=space_b, mask_data=np.ones((4, 4, 4), dtype=bool))
    res_b = ROIExtractionResult(
        values=np.zeros((1, 1)), coords=np.zeros((1, 3), dtype=int),
        space=space_b, mask_hash=rec_b.mask_hash, provenance=rec_b,
    )
    with pytest.raises(ValueError):
        res_a.require_compatible(res_b)


# ----------------------------------------------------------------------
# neuroim.verify helpers
# ----------------------------------------------------------------------


def test_verify_assert_same_space_passes():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    a = _make_searchlight_result(space)
    verify.assert_same_space(a, space)
    verify.assert_same_space(a, a.provenance)


def test_verify_assert_same_space_raises():
    space_a = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    space_b = NeuroSpace(dim=(4, 4, 4), spacing=(3.0, 3.0, 3.0))
    # NeuroSpace-vs-NeuroSpace now routes through compatible_with, which
    # raises with "spatial contract mismatch" rather than the hash-keyed
    # message used for Receipt-only inputs.
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        verify.assert_same_space(space_a, space_b)


def test_verify_assert_same_mask_passes():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    mask = np.ones((4, 4, 4), dtype=bool)
    rec = make_receipt(input_space=space, mask_data=mask)
    verify.assert_same_mask(rec, mask)


def test_verify_assert_same_mask_raises():
    mask_a = np.ones((4, 4, 4), dtype=bool)
    mask_b = mask_a.copy()
    mask_b[0, 0, 0] = False
    with pytest.raises(ValueError, match="mask_hash"):
        verify.assert_same_mask(mask_a, mask_b)


def test_verify_diff_receipts_returns_structured_diff():
    a = _receipt()
    b = _receipt(mask_hash="dddd")
    d = verify.diff_receipts(a, b)
    assert d == {"mask_hash": ("bbb", "dddd")}


def test_verify_diff_receipts_rejects_non_receipt_inputs():
    with pytest.raises(TypeError, match="Receipt"):
        verify.diff_receipts("not a receipt", _receipt())


# ----------------------------------------------------------------------
# Decisive test: the mission claim
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# assert_same_space routes through NeuroSpace.compatible_with for the
# 4-D-BOLD vs 3-D-mask case (per pain-points-from-loop reply
# post-01KRKN0134B0K5FCQS4EHNBG6H).
# ----------------------------------------------------------------------


def test_assert_same_space_accepts_4d_bold_vs_3d_mask():
    """A 4-D BOLD's spatial space is compatible with a 3-D mask in the same
    world frame.  The verifier must accept this — Receipt hashes include
    the time axis but the mask/data contract does not."""
    space_4d = NeuroSpace(dim=(32, 32, 24, 40), spacing=(3.0, 3.0, 3.5, 2.0))
    space_3d = NeuroSpace(dim=(32, 32, 24), spacing=(3.0, 3.0, 3.5))
    # No raise.
    verify.assert_same_space(space_4d, space_3d)


def test_assert_same_space_via_result_dot_space_works_for_4d_vs_3d():
    """The same compatibility holds when one side is a result object that
    exposes ``.space`` (the typical Draw Audit shape)."""
    space_4d = NeuroSpace(dim=(8, 8, 4, 5), spacing=(2.0, 2.0, 2.0, 1.0))
    space_3d = NeuroSpace(dim=(8, 8, 4), spacing=(2.0, 2.0, 2.0))
    rec = make_receipt(input_space=space_3d, mask_data=np.ones((8, 8, 4), dtype=bool))
    res_3d = SearchlightResult(
        values=np.zeros(0), centers=np.zeros((0, 3), dtype=int),
        space=space_3d, radius=2.0, shape="sphere", provenance=rec,
    )
    # Either side can be the 4-D one; both directions must accept.
    verify.assert_same_space(space_4d, res_3d)
    verify.assert_same_space(res_3d, space_4d)


def test_assert_same_space_still_rejects_spatial_dim_mismatch():
    space_a = NeuroSpace(dim=(8, 8, 4), spacing=(2.0, 2.0, 2.0))
    space_b = NeuroSpace(dim=(8, 8, 5), spacing=(2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        verify.assert_same_space(space_a, space_b)


def test_assert_same_space_still_rejects_affine_mismatch():
    space_a = NeuroSpace(dim=(8, 8, 4), spacing=(2.0, 2.0, 2.0))
    space_b = NeuroSpace(dim=(8, 8, 4), spacing=(3.0, 3.0, 3.0))
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        verify.assert_same_space(space_a, space_b)


def test_assert_same_space_receipt_vs_receipt_stays_hash_strict():
    """Two Receipts whose ``input_space_hash`` fields differ must reject.
    Receipt-only inputs lack the structured space to invoke
    ``compatible_with``, so the verifier keeps the strict hash check."""
    rec_a = _receipt(input_space_hash="space-A")
    rec_b = _receipt(input_space_hash="space-B")
    with pytest.raises(ValueError, match="input_space_hash"):
        verify.assert_same_space(rec_a, rec_b)


def test_verifier_catches_silent_space_mismatch():
    """The mission claim: silent space mismatches become visible.

    Two NeuroSpaces differ only in voxel spacing (2.0mm vs 3.0mm).  A
    Before PAIN-5, the explicit legacy projection could still hand a bare
    matrix from one space's data into a downstream op keyed off another
    space's mask.  The series_roi contract now catches that at extraction
    time, and typed results remain verifier-compatible downstream.
    """
    space_native = NeuroSpace(dim=(4, 4, 4, 3), spacing=(2.0, 2.0, 2.0, 1.0))
    space_mni = NeuroSpace(dim=(4, 4, 4, 3), spacing=(3.0, 3.0, 3.0, 1.0))

    rng = np.random.default_rng(seed=0)
    data = rng.standard_normal((4, 4, 4, 3)).astype(np.float32)
    vec_native = DenseNeuroVec(data, space_native)

    mask_3d_native = LogicalNeuroVol(
        np.ones((4, 4, 4), dtype=bool),
        NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0)),
    )
    mask_3d_mni = LogicalNeuroVol(
        np.ones((4, 4, 4), dtype=bool),
        NeuroSpace(dim=(4, 4, 4), spacing=(3.0, 3.0, 3.0)),
    )

    # --- LEGACY PROJECTION PATH (explicit opt-in):
    # return_legacy still enforces the same mask/data space contract.
    roi_mni = spherical_roi(mask_3d_mni, centroid=(2, 2, 2), radius=4.0)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="spatial contract mismatch"):
            vec_native.series_roi(roi_mni, return_legacy=True)

    # --- TYPED PIPELINE: default returns a Receipt-bearing result and the
    # verifier catches the mismatch.
    roi_native = spherical_roi(mask_3d_native, centroid=(2, 2, 2), radius=4.0)
    typed = vec_native.series_roi(roi_native)
    assert isinstance(typed, ROIExtractionResult)

    # The verifier raises when asked to compare against the wrong space.
    with pytest.raises(ValueError):
        verify.assert_same_space(typed, space_mni)
    # And the result object's own require_compatible raises too.
    with pytest.raises(ValueError):
        typed.require_compatible(space_mni)
