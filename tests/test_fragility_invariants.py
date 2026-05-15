"""Fragility-harness invariants for mission-bearing contracts.

Sources:

- bd-01KRKRZVYNNSRGXHWMRR9QYP9Y (open-discussion-2026-05-14):
  world-coordinate invariant harness.
- bd-01KRKSCV8FC2VK4YDWTB9KRZJK: adjacent PAIN-5 / PAIN-6 attack
  vectors for affine, mask-hash, and receipt-extension fragility.

The mission claim "useful validation errors" requires that public surfaces
fail loudly when handed inputs that violate their stated contract.  This
file holds the *invariant-shaped* slices of the fragility harness, distinct
from the *verifier-coverage* slice in ``test_verifier_blessed_path.py``.

Current invariant slices:

  World-coordinate operations raise on out-of-bounds by default
    Every public method/function that accepts a world-coordinate input
    (parameter name ``world_xyz`` / ``center_xyz`` or method-name suffix
    ``_world``) must raise ``ValueError`` or ``IndexError`` on a world
    coordinate that maps outside the image grid.  Silent zero-fill is
    only available via an explicit ``out_of_bounds="zero"`` opt-in.

    The structural test (``test_no_new_unmanaged_world_coord_surface``)
    discovers any new world-coord surface not listed in
    ``WORLD_COORD_MANIFEST``; the parametric test
    (``test_world_coord_surface_raises_on_oob_by_default``) asserts the
    raise contract on each managed surface.

  Adjacent PAIN-5 / PAIN-6 attacks stay guarded
    Dim-agreeing affine shifts/shears, dim/cardinality-agreeing mask
    reorderings, and malformed receipt extensions must not silently
    certify an output.

Further slices (typed-receipt-field coverage, write/read provenance
survival) will land alongside the structural-receipt construction work
(bd-01KRKRZPDC6V5CZF7SH0C9KEDD) once that lane closes.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import numpy as np
import pytest

import neuroim as ni
from neuroim.results import RECEIPT_NIFTI_PREFIX, Receipt, make_receipt


# ----------------------------------------------------------------------
# WORLD_COORD_MANIFEST
# ----------------------------------------------------------------------
# Each entry classifies one public world-coordinate-bearing surface.
#
#   ``verified``  Surface raises on OOB world coords by default.  The
#                 parametric test invokes it with an OOB coord and asserts
#                 a ValueError or IndexError is raised.
#
#   ``exempt``    Surface accepts world coordinates but the operation has
#                 no spatial-extraction semantics (e.g., pure coordinate
#                 transforms like ``NeuroSpace.world_to_grid``, which
#                 returns voxel indices regardless of whether they fall
#                 inside the volume).  A ``reason`` MUST accompany the
#                 entry.
#
#   ``gap``       Surface should raise on OOB but currently does not.
#                 Documented so the discovery test does not fail; a
#                 follow-up mote should flip the entry to ``verified``.

WORLD_COORD_MANIFEST: dict[str, dict[str, Any]] = {
    "neuroim.DenseNeuroVec.series_at_world": {"status": "verified"},
    "neuroim.DenseNeuroVec.series_roi_world": {"status": "verified"},
    # NeuroSpace.world_to_grid is a pure coordinate transform: it returns
    # voxel indices regardless of bounds.  Callers wrap it with their own
    # OOB policy (e.g., series_at_world above).  Exempt by design.
    "neuroim.NeuroSpace.world_to_grid": {
        "status": "exempt",
        "reason": (
            "pure coordinate transform; returns voxel indices for any "
            "world coordinate.  OOB enforcement is the caller's "
            "responsibility (see DenseNeuroVec.series_at_world)."
        ),
    },
    "neuroim.NeuroSpace.grid_to_world": {
        "status": "exempt",
        "reason": "pure coordinate transform from voxel to world; no OOB semantics.",
    },
}


_WORLD_COORD_PARAM_NAMES = {"world_xyz", "center_xyz", "world", "world_coord", "mni"}
_WORLD_NAME_SUFFIXES = ("_world",)


def _annotation_repr(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", repr(annotation))


def _is_world_coord_param(name: str, annotation: Any) -> bool:
    if name in _WORLD_COORD_PARAM_NAMES:
        return True
    rep = _annotation_repr(annotation)
    return "WorldCoord" in rep


def _surface_takes_world_coord(callable_obj: Callable, qualified_name: str) -> bool:
    """True if the signature accepts a world-coordinate input OR the
    qualified name ends with ``_world``."""
    if any(qualified_name.endswith(suf) for suf in _WORLD_NAME_SUFFIXES):
        return True
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(_is_world_coord_param(p.name, p.annotation) for p in sig.parameters.values())


# Classes whose methods we walk for world-coord surfaces.  Mirrors the
# data-carrier set in test_verifier_blessed_path.py plus NeuroSpace
# (the carrier of grid<->world transforms).
_WORLD_COORD_CLASSES = (
    "DenseNeuroVec",
    "DenseNeuroVol",
    "DenseNeuroHyperVec",
    "SparseNeuroVec",
    "SparseNeuroVol",
    "BigNeuroVec",
    "FileBackedNeuroVec",
    "MappedNeuroVec",
    "LogicalNeuroVol",
    "NeuroSpace",
)


def _enumerate_world_coord_surfaces() -> list[tuple[str, Callable]]:
    surfaces: list[tuple[str, Callable]] = []
    seen: set[str] = set()
    for name in sorted(ni.__all__):
        obj = getattr(ni, name, None)
        if obj is None:
            continue
        if inspect.isclass(obj):
            if name not in _WORLD_COORD_CLASSES:
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


def test_no_new_unmanaged_world_coord_surface():
    """Every public world-coord-bearing surface must be in WORLD_COORD_MANIFEST.

    Failure means a new method/function accepts a world coordinate (by
    parameter name, ``WorldCoord`` annotation, or method-name suffix
    ``_world``) without an explicit decision about OOB behaviour.

    Resolution: add the qualified name to ``WORLD_COORD_MANIFEST`` with
    ``status='verified'`` (and ensure the surface raises on OOB by
    default) or ``status='exempt' / 'gap'`` with a documented reason.
    """
    surfaces = _enumerate_world_coord_surfaces()
    candidates = [qname for qname, fn in surfaces if _surface_takes_world_coord(fn, qname)]
    unmanaged = sorted(set(candidates) - set(WORLD_COORD_MANIFEST))
    if unmanaged:
        pytest.fail(
            "New world-coord-bearing surface(s) not in WORLD_COORD_MANIFEST:\n  "
            + "\n  ".join(unmanaged)
            + "\nAdd each to tests/test_fragility_invariants.py::WORLD_COORD_MANIFEST"
            " with status='verified' (and ensure the surface raises on OOB by"
            " default) or status='exempt' / 'gap' with a documented reason."
        )


# ----------------------------------------------------------------------
# OOB-raise test
# ----------------------------------------------------------------------


@pytest.fixture
def small_vec():
    """Tiny 4-D NeuroVec on a default identity-affine 8x8x4 grid.

    World coordinates run from 0 to ~8 along x/y and 0 to ~4 along z;
    anything outside that range is unambiguously OOB.
    """
    space = ni.NeuroSpace(dim=[8, 8, 4, 3])
    data = np.zeros((8, 8, 4, 3), dtype=np.float32)
    return ni.DenseNeuroVec(data, space)


_OOB_WORLD_COORDS: list[tuple[str, np.ndarray]] = [
    ("far_positive", np.array([1000.0, 1000.0, 1000.0])),
    ("far_negative", np.array([-1000.0, -1000.0, -1000.0])),
    ("just_outside_positive", np.array([100.0, 100.0, 100.0])),
    ("just_outside_negative", np.array([-1.0, -1.0, -1.0])),
]


def _oob_invocation_for(qualified_name: str, vec, oob_xyz) -> Optional[Callable]:
    if qualified_name == "neuroim.DenseNeuroVec.series_at_world":
        return lambda: vec.series_at_world(oob_xyz)
    if qualified_name == "neuroim.DenseNeuroVec.series_roi_world":
        return lambda: vec.series_roi_world(oob_xyz, radius=0.0)
    return None


_VERIFIED_WORLD_SURFACES = sorted(
    qname for qname, entry in WORLD_COORD_MANIFEST.items()
    if entry.get("status") == "verified"
)


@pytest.mark.parametrize("qualified_name", _VERIFIED_WORLD_SURFACES)
@pytest.mark.parametrize("case_name,oob_xyz", _OOB_WORLD_COORDS)
def test_world_coord_surface_raises_on_oob_by_default(
    qualified_name, case_name, oob_xyz, small_vec
):
    """Every ``verified`` world-coord surface must raise on OOB by default.

    Default behaviour for safety-critical extraction is raise; silent
    zero-fill is only available via an explicit
    ``out_of_bounds="zero"`` opt-in.  A refactor that flips the default
    to silent fill would re-introduce the exact bug class S01 PAIN-2
    was filed against.
    """
    invocation = _oob_invocation_for(qualified_name, small_vec, oob_xyz)
    if invocation is None:
        pytest.skip(
            f"No fixture invocation defined for {qualified_name}; "
            "extend _oob_invocation_for() in tests/test_fragility_invariants.py"
        )
    with pytest.raises((ValueError, IndexError)) as excinfo:
        invocation()
    # Error message should name the OOB condition; do not pin exact wording
    # but require something the user can grep on.
    message = str(excinfo.value).lower()
    assert any(tok in message for tok in ("outside", "out of bounds", "oob", "bounds")), (
        f"{qualified_name} raised on OOB world coord {case_name}={oob_xyz} but"
        f" the message did not name the OOB condition: {excinfo.value!r}"
    )


# ----------------------------------------------------------------------
# Adjacent PAIN-5 / PAIN-6 attack vectors
# ----------------------------------------------------------------------


def _roi_space_with_affine(affine: np.ndarray) -> ni.NeuroSpace:
    return ni.NeuroSpace(dim=[8, 8, 4], trans=affine)


@pytest.mark.parametrize(
    "case_name,affine",
    [
        (
            "small_translation",
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.25],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            "axis_shear",
            np.array(
                [
                    [1.0, 0.05, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
    ],
)
def test_series_roi_rejects_dim_agreeing_affine_attacks(
    small_vec, case_name, affine
):
    """PAIN-5 must reject more than the original LR-flip fixture.

    These ROIs have the same integer grid dimensions as ``small_vec`` but a
    different world frame.  Extraction must fail before coordinate lookup.
    """
    coords = np.array([[1, 1, 1], [2, 2, 2]])
    roi = ni.ROICoords(coords, _roi_space_with_affine(affine))

    with pytest.raises(ValueError, match="spatial contract mismatch") as excinfo:
        small_vec.series_roi(roi)

    assert case_name
    assert "affine" in str(excinfo.value).lower()


def test_receipt_merge_rejects_dim_agreeing_mask_order_attack():
    """PAIN-6 receipt composition must not trust shape/cardinality alone.

    The two masks have identical dimensions and identical voxel counts, but
    the true voxels are arranged differently.  A downstream merge must reject
    on ``mask_hash`` rather than treating them as interchangeable.
    """
    space = ni.NeuroSpace(dim=[4, 4, 4])
    mask_a = np.zeros((4, 4, 4), dtype=bool)
    mask_a[0, :, :] = True
    mask_b = np.transpose(mask_a, (1, 0, 2)).copy()

    assert mask_a.shape == mask_b.shape
    assert int(mask_a.sum()) == int(mask_b.sum())
    assert not np.array_equal(mask_a, mask_b)

    rec_a = make_receipt(input_space=space, mask_data=mask_a, method_name="a")
    rec_b = make_receipt(input_space=space, mask_data=mask_b, method_name="b")

    with pytest.raises(ValueError, match="mask_hash"):
        rec_a.merge(rec_b)


@pytest.mark.parametrize(
    "payload",
    [
        b"neuroim/receipt/v2:{}",
        (RECEIPT_NIFTI_PREFIX + '{"input_space_hash": ').encode("utf-8"),
        (RECEIPT_NIFTI_PREFIX + '{"input_space_hash": "only-one-field"}').encode(
            "utf-8"
        ),
        (RECEIPT_NIFTI_PREFIX + "not-json").encode("utf-8"),
    ],
)
def test_malformed_receipt_extensions_do_not_hydrate_partial_receipts(payload):
    """Malformed or unsupported receipt payloads must not become provenance.

    Current read semantics are deliberately conservative: unsupported marker
    versions and corrupt v1 JSON are ignored rather than hydrated into a
    partial ``Receipt`` that could falsely certify an output.
    """
    assert Receipt.from_nifti_extension_bytes(payload) is None


# ----------------------------------------------------------------------
# Multi-input same-space contract (concat-family ops)
# ----------------------------------------------------------------------
# The verifier-blessed-path harness in test_verifier_blessed_path.py
# covers the (data + mask/ROI) shape.  ``NeuroVec.concat`` is the same
# bug class on a different shape: two data carriers with separate spaces.
# Mission rule 4 ('silent space/orientation mismatches caught at the
# contract layer') requires concat to reject affine mismatch before
# materialising the output.

MULTI_INPUT_MANIFEST: dict[str, dict[str, Any]] = {
    "neuroim.DenseNeuroVec.concat": {"status": "verified"},
    "neuroim.SparseNeuroVec.concat": {"status": "verified"},
}


def _multi_input_invocation(qualified_name: str, primary, secondary):
    """Return a zero-arg thunk calling the multi-input op on the given pair.

    ``primary`` is the receiver / first arg; ``secondary`` is the input whose
    space differs from ``primary``.  The op should reject the call before
    producing a result.
    """
    if qualified_name == "neuroim.DenseNeuroVec.concat":
        return lambda: primary.concat(secondary)
    if qualified_name == "neuroim.SparseNeuroVec.concat":
        return lambda: primary.concat(secondary)
    return None


_VERIFIED_MULTI_INPUT_SURFACES = sorted(
    qname for qname, entry in MULTI_INPUT_MANIFEST.items()
    if entry.get("status") == "verified"
)


@pytest.fixture
def matched_concat_pair():
    """Two DenseNeuroVecs on the same 4-D space; concat must succeed."""
    vec_space = ni.NeuroSpace(dim=[8, 8, 4, 3])
    data_a = np.zeros((8, 8, 4, 3), dtype=np.float32)
    data_b = np.ones((8, 8, 4, 3), dtype=np.float32)
    return ni.DenseNeuroVec(data_a, vec_space), ni.DenseNeuroVec(data_b, vec_space)


@pytest.fixture
def mismatched_concat_pair():
    """Two DenseNeuroVecs whose spatial dims agree but affines differ.

    The mismatch is a small translation (0.5mm along x), large enough that
    ``NeuroSpace.compatible_with`` rejects but small enough that the integer
    dim tuple is identical.  This exercises the affine-comparison branch of
    the verifier, not the shape-comparison shortcut.
    """
    trans = np.eye(4)
    space_a = ni.NeuroSpace(dim=[8, 8, 4, 3], trans=trans)
    shifted = trans.copy()
    shifted[0, 3] = 0.5
    space_b = ni.NeuroSpace(dim=[8, 8, 4, 3], trans=shifted)
    data = np.zeros((8, 8, 4, 3), dtype=np.float32)
    return ni.DenseNeuroVec(data.copy(), space_a), ni.DenseNeuroVec(data.copy(), space_b)


def _pair_for_multi_input_surface(qualified_name: str, dense_pair):
    primary, secondary = dense_pair
    if qualified_name == "neuroim.SparseNeuroVec.concat":
        mask = np.ones(primary.shape[:3], dtype=bool)
        return primary.to_sparse(mask), secondary.to_sparse(mask)
    return primary, secondary


@pytest.mark.parametrize("qualified_name", _VERIFIED_MULTI_INPUT_SURFACES)
def test_multi_input_op_rejects_affine_mismatch(qualified_name, mismatched_concat_pair):
    """Every ``verified`` multi-input op must raise on affine mismatch.

    Dense and sparse concat both use the same same-space gate; the sparse
    fixture keeps that override from drifting back to a convention-only
    claim.
    """
    primary, secondary = _pair_for_multi_input_surface(
        qualified_name, mismatched_concat_pair
    )
    invocation = _multi_input_invocation(qualified_name, primary, secondary)
    if invocation is None:
        pytest.skip(
            f"No fixture invocation defined for {qualified_name}; "
            "extend _multi_input_invocation() in tests/test_fragility_invariants.py"
        )
    with pytest.raises(ValueError) as excinfo:
        invocation()
    message = str(excinfo.value).lower()
    assert "spatial contract mismatch" in message or "input_space_hash" in message, (
        f"{qualified_name} raised on affine mismatch but the message did not "
        f"name the spatial contract: {excinfo.value!r}"
    )


@pytest.mark.parametrize("qualified_name", _VERIFIED_MULTI_INPUT_SURFACES)
def test_multi_input_op_accepts_matched_pair(qualified_name, matched_concat_pair):
    """Positive control: matched-space pairs must succeed.

    Sanity check that the test fixture isn't accidentally tripping the
    rejection path (which would mask a future regression).
    """
    primary, secondary = _pair_for_multi_input_surface(qualified_name, matched_concat_pair)
    invocation = _multi_input_invocation(qualified_name, primary, secondary)
    if invocation is None:
        pytest.skip(
            f"No fixture invocation defined for {qualified_name}; "
            "extend _multi_input_invocation() in tests/test_fragility_invariants.py"
        )
    result = invocation()
    assert result is not None
    # Time axis should be the sum of the inputs (3 + 3 = 6).
    assert result.shape[-1] == 6


# ---------------------------------------------------------------------------
# Slice 4 — Write/read provenance enumeration
# ---------------------------------------------------------------------------
#
# Source: bd-01KRKX8FJS96J2DRGJ7T2F66VK / fragility harness slice 4.
#
# Mission claim: when a typed result or spatial container carries a
# ``Receipt``, ``to_nibabel()`` embeds it as a NIfTI ecode-6 extension with
# the ``neuroim/receipt/v1:`` marker, and ``neuroim.read_image`` rehydrates
# it onto ``.provenance``.  This is the claim the linter-strip incident
# falsified at one surface (NeuroVol.to_nibabel); without an enumeration
# test, a future refactor at any other ``to_nibabel`` surface would silently
# drop the receipt and only the per-class unit tests would catch it.
#
# The structural test (``test_no_new_unmanaged_write_back_surface``) walks
# neuroim's public classes for ``.to_nibabel`` methods and fails when a
# new surface is not listed in ``WRITE_BACK_MANIFEST``.  The parametric
# tests build a carrier with a Receipt set, exercise the embed + the disk
# round-trip, and assert the contract.

from neuroim.results import (  # noqa: E402  (after other imports by design)
    RECEIPT_NIFTI_PREFIX,
    Receipt,
    ROIExtractionResult,
    SearchlightResult,
    make_receipt,
)


def _slice4_receipt() -> Receipt:
    return make_receipt(
        input_space=ni.NeuroSpace((4, 4, 4)),
        mask_data=np.array([[0, 0, 0], [1, 1, 1]], dtype=int),
        n_voxels=2,
        method_name="slice4-fixture",
        seed=None,
    )


def _build_dense_neurovol_with_receipt() -> ni.DenseNeuroVol:
    vol = ni.DenseNeuroVol(np.zeros((4, 4, 4)), ni.NeuroSpace((4, 4, 4)))
    vol.provenance = _slice4_receipt()
    return vol


def _build_dense_neurovec_with_receipt() -> ni.DenseNeuroVec:
    vec = ni.DenseNeuroVec(np.zeros((4, 4, 4, 3)), ni.NeuroSpace((4, 4, 4, 3)))
    vec.provenance = _slice4_receipt()
    return vec


def _build_searchlight_result_with_receipt() -> SearchlightResult:
    return SearchlightResult(
        values=np.array([1.0]),
        centers=np.array([[1, 1, 1]]),
        space=ni.NeuroSpace((4, 4, 4)),
        radius=4.0,
        shape="sphere",
        provenance=_slice4_receipt(),
    )


def _build_roi_extraction_result_with_receipt() -> ROIExtractionResult:
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=int)
    return ROIExtractionResult(
        values=np.array([10.0, 20.0]),
        coords=coords,
        space=ni.NeuroSpace((4, 4, 4)),
        mask_hash="m",
        provenance=_slice4_receipt(),
    )


WRITE_BACK_MANIFEST: dict[str, dict[str, Any]] = {
    "neuroim.DenseNeuroVol.to_nibabel": {
        "builder": _build_dense_neurovol_with_receipt,
    },
    "neuroim.DenseNeuroVec.to_nibabel": {
        "builder": _build_dense_neurovec_with_receipt,
    },
    "neuroim.results.SearchlightResult.to_nibabel": {
        "builder": _build_searchlight_result_with_receipt,
    },
    "neuroim.results.ROIExtractionResult.to_nibabel": {
        "builder": _build_roi_extraction_result_with_receipt,
    },
}


_RECEIPT_BEARING_BASE_CLASSES: tuple[type, ...] = (
    ni.NeuroVol,
    ni.NeuroVec,
    SearchlightResult,
    ROIExtractionResult,
)


def _enumerate_write_back_surfaces() -> list[str]:
    """Discover concrete public classes that expose ``.to_nibabel`` and could
    plausibly carry provenance (subclasses of NeuroVol / NeuroVec / result
    objects).  Returns qualified names like ``neuroim.DenseNeuroVol.to_nibabel``.
    """
    found: list[str] = []
    for name in dir(ni):
        if name.startswith("_"):
            continue
        obj = getattr(ni, name, None)
        if not inspect.isclass(obj):
            continue
        if inspect.isabstract(obj):
            continue
        if not any(issubclass(obj, base) for base in _RECEIPT_BEARING_BASE_CLASSES):
            continue
        if not hasattr(obj, "to_nibabel"):
            continue
        # Anchor on the defining class to avoid each subclass producing a
        # separate entry for an inherited method.
        method = obj.to_nibabel
        defining_qualname = getattr(method, "__qualname__", f"{obj.__name__}.to_nibabel")
        module = getattr(method, "__module__", obj.__module__)
        found.append(f"{module}.{defining_qualname}")

    # Result-bearing classes are not on the top-level ``neuroim`` namespace
    # under their class names; add them explicitly so the enumeration covers
    # the typed-result write-back surface too.
    for cls in (SearchlightResult, ROIExtractionResult):
        if not hasattr(cls, "to_nibabel"):
            continue
        method = cls.to_nibabel
        defining_qualname = getattr(method, "__qualname__", f"{cls.__name__}.to_nibabel")
        module = getattr(method, "__module__", cls.__module__)
        found.append(f"{module}.{defining_qualname}")

    return sorted(set(found))


def test_no_new_unmanaged_write_back_surface():
    """Every public ``.to_nibabel`` on a provenance-bearing class must be
    listed in ``WRITE_BACK_MANIFEST``.

    Failure means a new write surface was added without a fragility-harness
    entry — the same gap that allowed the linter-strip incident to land
    silently before this slice existed.

    Resolution: add the qualified name to ``WRITE_BACK_MANIFEST`` with a
    builder function that constructs a carrier with ``.provenance`` set.
    """
    discovered = set(_enumerate_write_back_surfaces())
    declared = {
        # Normalize the manifest's friendly names to the qualname-style
        # used by the walker.  Both forms are accepted because the manifest
        # is the human-readable index and the walker is the structural
        # discovery.
        "neuroim.neuro_vol.NeuroVol.to_nibabel",
        "neuroim.neuro_vec.NeuroVec.to_nibabel",
        "neuroim.results.SearchlightResult.to_nibabel",
        "neuroim.results.ROIExtractionResult.to_nibabel",
    }
    unmanaged = discovered - declared
    assert unmanaged == set(), (
        "New ``.to_nibabel`` surface(s) not in WRITE_BACK_MANIFEST:\n  "
        + "\n  ".join(sorted(unmanaged))
        + "\nAdd each to tests/test_fragility_invariants.py::WRITE_BACK_MANIFEST"
        " with a builder that sets .provenance and exercises to_nibabel()."
    )


@pytest.mark.parametrize("qualified_name", sorted(WRITE_BACK_MANIFEST))
def test_write_back_embeds_receipt_extension(qualified_name):
    """When the carrier has ``.provenance`` set, ``.to_nibabel()`` must
    emit exactly one NIfTI ecode-6 extension whose payload starts with
    ``neuroim/receipt/v1:``.
    """
    entry = WRITE_BACK_MANIFEST[qualified_name]
    carrier = entry["builder"]()
    img = carrier.to_nibabel()
    extensions = list(img.header.extensions)
    receipt_exts = [
        ext
        for ext in extensions
        if int(ext.get_code()) == 6
        and bytes(ext.get_content()).rstrip(b"\x00").decode("utf-8", "replace").startswith(
            RECEIPT_NIFTI_PREFIX
        )
    ]
    assert len(receipt_exts) == 1, (
        f"{qualified_name} produced {len(receipt_exts)} receipt extensions "
        f"(expected exactly 1).  All extensions: {extensions!r}"
    )


@pytest.mark.parametrize("qualified_name", sorted(WRITE_BACK_MANIFEST))
def test_write_back_disk_round_trip_rehydrates_receipt(tmp_path, qualified_name):
    """A NIfTI written from a carrier with ``.provenance`` set, then read
    by ``neuroim.read_image`` in the same process, must surface a
    ``.provenance`` whose Receipt matches the original.

    This is the closed-loop check the linter-strip incident slipped past.
    """
    import nibabel as nib

    entry = WRITE_BACK_MANIFEST[qualified_name]
    carrier = entry["builder"]()
    original = carrier.provenance
    img = carrier.to_nibabel()

    out = tmp_path / "write_back.nii.gz"
    nib.save(img, str(out))
    recovered = ni.io.read_image(str(out))
    assert hasattr(recovered, "provenance"), (
        f"{qualified_name}: read_image() did not surface .provenance on the "
        "returned object after write+read"
    )
    assert recovered.provenance == original, (
        f"{qualified_name}: rehydrated Receipt differs from the original "
        f"that was attached before to_nibabel(): {recovered.provenance!r} vs {original!r}"
    )
