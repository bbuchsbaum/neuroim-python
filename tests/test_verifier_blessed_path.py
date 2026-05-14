"""Verifier-blessed-path enumeration test.

Source: bd-01KRKSCS5AGZFQ28EVCJJ76R9K (open-discussion-2026-05-14).

The mission claim "silent space/orientation/mask mismatches are caught at the
contract layer" (MISSION.md rule 4) holds only as long as every public
operation that takes both a data carrier and a mask/ROI invokes
:func:`neuroim.verify.assert_same_space` before doing spatial work.

Before this test, the convention was enforced by author diligence alone --
nothing in CI prevented a refactor from quietly dropping the verifier call.
That was the exact shape of the S02 PAIN-5 bug class.

Two complementary tests below lock the convention:

``test_no_new_unmanaged_mask_carrying_surface``
    Walks the public ``neuroim`` namespace looking for any callable whose
    signature contains both a mask/ROI-shaped parameter and a data-carrier-
    shaped parameter.  Any surface missing from ``VERIFIER_MANIFEST`` fails
    the test with "add to VERIFIER_MANIFEST and decide whether it must
    gate."  Makes new safety-critical surfaces explicit rather than silent.

``test_verifier_invoked_on_blessed_surface``
    Parametrized on every ``VERIFIER_MANIFEST`` entry marked ``verified``.
    Monkey-patches ``neuroim.verify.assert_same_space`` to a recording spy,
    then invokes the surface with a matched fixture.  Asserts the spy
    recorded a call -- i.e., the surface really does route through the
    verifier.  A regression that drops the call fails this test.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import numpy as np
import pytest

import neuroim as ni
from neuroim import verify as _verify
from neuroim.roi import ROICoords


# ----------------------------------------------------------------------
# VERIFIER_MANIFEST
# ----------------------------------------------------------------------
# Each entry classifies one public mask/ROI-bearing surface.
#
#   ``verified``   Surface invokes verify.assert_same_space.  The coverage
#                  test parametrizes over these and confirms with a spy.
#
#   ``gap``        Surface takes data + mask/ROI but currently does not
#                  invoke the verifier.  Documented here so the discovery
#                  test does not fail; a follow-up mote should flip the
#                  entry to ``verified`` and wire the call at the surface.
#                  The ``reason`` records the current behaviour.
#
#   ``exempt``     Surface takes a mask-shaped parameter but the parameter
#                  is not a separate spatial contract (e.g. the mask is a
#                  logical selector applied to a space already shared by
#                  construction, with no foreign-affine vector possible).
#                  A ``reason`` MUST accompany the entry.

VERIFIER_MANIFEST: dict[str, dict[str, Any]] = {
    # Module-level ROI extraction helpers (roi.py).  ``values_roi`` does
    # invoke verify.assert_same_space (roi.py:354).  ``series_roi`` is a
    # passthrough that delegates to NeuroVec.series_roi when the carrier
    # supplies one -- inherits whatever gate that surface enforces.
    "neuroim.values_roi": {"status": "verified"},
    # ``series_roi`` is a passthrough that delegates to NeuroVec.series_roi
    # when the carrier supplies one; inherits that surface's gate.
    "neuroim.series_roi": {"status": "verified"},
    # NeuroVec ROI / temporal surfaces.  Both regressed in commit 173fc26
    # ('Land mission evidence scenarios and provenance contracts') and were
    # subsequently restored; tests/test_series_roi.py::
    # test_series_roi_rejects_roi_space_mismatch and tests/test_verify.py::
    # test_verifier_catches_silent_space_mismatch now pass on main, so the
    # entries are flipped back to ``verified`` to lock the contract.
    "neuroim.DenseNeuroVec.series_roi": {"status": "verified"},
    "neuroim.DenseNeuroVec.series_roi_world": {
        "status": "verified",
        # Delegates to series_roi which gates; the verifier spy catches
        # the inner call.
    },
    "neuroim.DenseNeuroVec.temporal_snr": {"status": "verified"},
    # Sparse-conversion overloads on NeuroVec subclasses now gate
    # LogicalNeuroVol masks through assert_same_space; ndarray masks have
    # no embedded spatial contract to verify.
    "neuroim.DenseNeuroVec.as_sparse": {"status": "verified"},
    "neuroim.SparseNeuroVec.as_sparse": {"status": "verified"},
    "neuroim.BigNeuroVec.as_sparse": {
        "status": "gap",
        "reason": "see neuroim.DenseNeuroVec.as_sparse",
    },
    "neuroim.FileBackedNeuroVec.as_sparse": {
        "status": "gap",
        "reason": "see neuroim.DenseNeuroVec.as_sparse",
    },
    "neuroim.MappedNeuroVec.as_sparse": {
        "status": "gap",
        "reason": "see neuroim.DenseNeuroVec.as_sparse",
    },
    "neuroim.DenseNeuroVol.as_sparse": {
        "status": "gap",
        "reason": "see neuroim.DenseNeuroVec.as_sparse",
    },
    "neuroim.SparseNeuroVol.as_sparse": {
        "status": "gap",
        "reason": "see neuroim.DenseNeuroVec.as_sparse",
    },
    # NeuroHyperVec factory: takes data + optional mask; a foreign-affine
    # mask would propagate into the resulting hypervec without a check.
    "neuroim.NeuroHyperVec": {
        "status": "gap",
        "reason": (
            "Constructor accepts data + optional LogicalNeuroVol mask but "
            "does not invoke verify.assert_same_space.  Same bug class as "
            "PAIN-5; tracked here until a fix lands."
        ),
    },
    # High-level searchlight: mask + data inputs but no verifier call today.
    "neuroim.searchlight": {
        "status": "gap",
        "reason": (
            "Top-level searchlight(mask, data=...) does not invoke "
            "verify.assert_same_space; mismatched spaces produce silently "
            "wrong indexing."
        ),
    },
    "neuroim.searchlight_apply": {
        "status": "gap",
        "reason": "downstream of searchlight; same gap.",
    },
}


# Parameter-name heuristics.
_MASK_PARAM_NAMES = {"mask", "roi", "region"}
_DATA_PARAM_NAMES = {"data", "vec", "vol", "x", "img", "carrier"}
_MASK_TYPE_TOKENS = ("ROICoords", "ROIVol", "LogicalNeuroVol", "MaskLike")
_DATA_TYPE_TOKENS = ("NeuroVec", "NeuroVol", "Volume", "NeuroVecLike", "NeuroVolLike")


def _annotation_repr(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", repr(annotation))


def _is_mask_param(name: str, annotation: Any) -> bool:
    if name in _MASK_PARAM_NAMES:
        return True
    rep = _annotation_repr(annotation)
    return any(tok in rep for tok in _MASK_TYPE_TOKENS)


def _is_data_param(name: str, annotation: Any) -> bool:
    if name in _DATA_PARAM_NAMES:
        return True
    rep = _annotation_repr(annotation)
    return any(tok in rep for tok in _DATA_TYPE_TOKENS)


def _surface_takes_data_and_mask(callable_obj: Callable) -> bool:
    """True iff the signature includes a mask-shaped param and a *distinct*
    data-carrier-shaped param.  A single parameter typed
    ``Union[NeuroVolLike, MaskLike]`` (the ROI factories) or
    ``Union[DenseNeuroVol, ..., LogicalNeuroVol]`` (the writers) matches both
    heuristics on the same name; that is "data OR mask", not "data AND
    mask", and is filtered out by the set difference below.

    Methods on Vec/Vol classes satisfy the data side via ``self``.
    """
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    mask_params = {p.name for p in params if _is_mask_param(p.name, p.annotation)}
    data_params = {p.name for p in params if _is_data_param(p.name, p.annotation)}
    if any(p.name == "self" for p in params):
        data_params.add("self")
    mask_only = mask_params - data_params
    data_only = data_params - mask_params
    return bool(mask_only and data_only)


# Names enumerated from the public surface.  Only methods defined directly on
# the class (not inherited from object) are considered; we walk ``ni.__all__``
# for classes and their declared methods, plus top-level functions.
_DATA_CARRIER_CLASSES = (
    "DenseNeuroVec",
    "DenseNeuroVol",
    "DenseNeuroHyperVec",
    "SparseNeuroVec",
    "SparseNeuroVol",
    "BigNeuroVec",
    "FileBackedNeuroVec",
    "MappedNeuroVec",
    "LogicalNeuroVol",
)


def _enumerate_public_surfaces() -> list[tuple[str, Callable]]:
    surfaces: list[tuple[str, Callable]] = []
    seen: set[str] = set()
    for name in sorted(ni.__all__):
        obj = getattr(ni, name, None)
        if obj is None:
            continue
        if inspect.isclass(obj):
            if name not in _DATA_CARRIER_CLASSES:
                continue
            for meth_name in sorted(vars(obj)):
                if meth_name.startswith("_"):
                    continue
                meth = getattr(obj, meth_name, None)
                if not callable(meth):
                    continue
                qname = f"neuroim.{name}.{meth_name}"
                if qname in seen:
                    continue
                seen.add(qname)
                surfaces.append((qname, meth))
        elif inspect.isfunction(obj):
            qname = f"neuroim.{name}"
            if qname in seen:
                continue
            seen.add(qname)
            surfaces.append((qname, obj))
    return surfaces


# ----------------------------------------------------------------------
# Discovery test
# ----------------------------------------------------------------------


def test_no_new_unmanaged_mask_carrying_surface():
    """Every public (data + mask/ROI) surface must be in VERIFIER_MANIFEST.

    Failure means a new public method/function has been added that accepts
    both a data carrier and a mask-shaped parameter without an explicit
    decision about whether it gates on ``verify.assert_same_space``.

    Resolution: add the qualified name to ``VERIFIER_MANIFEST`` with
    ``status='verified'`` (and wire the verifier call at the surface) or
    ``status='gap' / 'exempt'`` with a ``reason``.
    """
    surfaces = _enumerate_public_surfaces()
    candidates = [
        qname for qname, fn in surfaces if _surface_takes_data_and_mask(fn)
    ]
    unmanaged = sorted(set(candidates) - set(VERIFIER_MANIFEST))
    if unmanaged:
        pytest.fail(
            "New mask/ROI-bearing surface(s) not in VERIFIER_MANIFEST:\n  "
            + "\n  ".join(unmanaged)
            + "\nAdd each to tests/test_verifier_blessed_path.py::VERIFIER_MANIFEST"
            " with status='verified' (and wire verify.assert_same_space at"
            " the surface) or status='gap' / 'exempt' with a documented reason."
        )


# ----------------------------------------------------------------------
# Coverage test
# ----------------------------------------------------------------------


@pytest.fixture
def matched_fixtures():
    """Matched NeuroVec, NeuroVol, LogicalNeuroVol mask, and ROICoords.

    All four share the same spatial NeuroSpace (8, 8, 4).  The vector adds
    a 3-step time axis.  No mismatch; the coverage test only needs to
    confirm the verifier is *called*, not that it rejects.
    """
    spatial = ni.NeuroSpace(dim=[8, 8, 4])
    vec_space = ni.NeuroSpace(dim=[8, 8, 4, 3])
    vec_data = np.zeros((8, 8, 4, 3), dtype=np.float32)
    vec_data[2:6, 2:6, 1:3, :] = 1.0
    vec = ni.DenseNeuroVec(vec_data, vec_space)
    vol_data = np.asarray(vec_data[..., 0])
    vol = ni.DenseNeuroVol(vol_data, spatial)
    mask_data = np.zeros((8, 8, 4), dtype=bool)
    mask_data[2:6, 2:6, 1:3] = True
    mask = ni.LogicalNeuroVol(mask_data, spatial)
    coords = np.argwhere(mask_data)
    roi = ROICoords(np.ascontiguousarray(coords, dtype=int), spatial)
    return vec, vol, mask, roi


def _invocation_for(qualified_name: str, vec, vol, mask, roi) -> Optional[Callable]:
    """Return a zero-arg thunk that calls ``qualified_name`` on the matched
    fixture, or ``None`` if no thunk is defined here yet."""
    if qualified_name == "neuroim.DenseNeuroVec.series_roi":
        return lambda: vec.series_roi(roi)
    if qualified_name == "neuroim.DenseNeuroVec.series_roi_world":
        # World-coord at the center of the mask -> guaranteed in-bounds.
        # Delegates to series_roi internally, so the verifier spy catches
        # the inner call.
        center_voxel = np.array(roi.coords[0], dtype=float)
        world = np.asarray(
            vec.spatial_space.grid_to_world(center_voxel[None, :])[0], dtype=float
        )
        return lambda: vec.series_roi_world(world, radius=0.0)
    if qualified_name == "neuroim.DenseNeuroVec.temporal_snr":
        return lambda: vec.temporal_snr(mask=mask)
    if qualified_name == "neuroim.DenseNeuroVec.as_sparse":
        return lambda: vec.as_sparse(mask)
    if qualified_name == "neuroim.SparseNeuroVec.as_sparse":
        sparse = vec.to_sparse(mask.data)
        return lambda: sparse.as_sparse(mask)
    if qualified_name == "neuroim.values_roi":
        return lambda: ni.values_roi(vol, roi)
    if qualified_name == "neuroim.series_roi":
        return lambda: ni.series_roi(vec, roi)
    return None


_VERIFIED_SURFACES = sorted(
    qname for qname, entry in VERIFIER_MANIFEST.items()
    if entry.get("status") == "verified"
)


@pytest.mark.parametrize("qualified_name", _VERIFIED_SURFACES)
def test_verifier_invoked_on_blessed_surface(qualified_name, matched_fixtures, monkeypatch):
    """Every ``verified`` surface must call ``verify.assert_same_space``.

    Failure mode: a refactor silently drops the verifier call (the exact
    shape of the S02 PAIN-5 bug class).  The matched fixture means the
    verifier would *succeed* if called; the test fails when the verifier
    isn't called at all, regardless of whether the surface produced a
    plausible-looking result.
    """
    vec, vol, mask, roi = matched_fixtures
    invocation = _invocation_for(qualified_name, vec, vol, mask, roi)
    if invocation is None:
        pytest.skip(
            f"No fixture invocation defined for {qualified_name}; "
            "extend _invocation_for() in tests/test_verifier_blessed_path.py"
        )

    calls: list[tuple[Any, Any]] = []
    real_assert = _verify.assert_same_space

    def _spy(a, b):
        calls.append((a, b))
        return real_assert(a, b)

    # Patch the public attribute plus every internal binding that surfaces
    # use via ``from .verify import assert_same_space`` or
    # ``from . import verify as _verify``.
    monkeypatch.setattr(_verify, "assert_same_space", _spy)
    import neuroim.neuro_vec as _nv
    import neuroim.roi as _roi
    for mod in (_nv, _roi):
        if hasattr(mod, "assert_same_space"):
            monkeypatch.setattr(mod, "assert_same_space", _spy, raising=False)

    invocation()
    assert calls, (
        f"Surface {qualified_name} did not invoke verify.assert_same_space "
        "during a normal matched-fixture call.  Either the surface no longer "
        "gates on spatial contract (regression -- restore the verifier "
        "call) or the manifest status is wrong (update VERIFIER_MANIFEST)."
    )
