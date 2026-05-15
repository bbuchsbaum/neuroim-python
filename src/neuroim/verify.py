"""Receipt-based compatibility verification.

Top-level assertion helpers that consume result objects, NeuroSpaces, or
bare Receipts and raise ``ValueError`` with a structured diff when their
provenance disagrees.  Built on :class:`neuroim.results.Receipt` and the
``hash_neurospace`` / ``hash_ndarray`` helpers.

The mission claim is "silent space/orientation/mask mismatches become visible."
Receipts are content-addressable metadata that make those mismatches detectable;
this module is the consumer side that actually catches them.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .results import Receipt, hash_ndarray, hash_neurospace
from .exceptions import SpaceMismatchError, MaskMismatchError

__all__ = [
    "assert_same_space",
    "assert_same_mask",
    "diff_receipts",
    "receipt_of",
]


def receipt_of(obj: Any) -> Optional[Receipt]:
    """Extract a :class:`Receipt` from ``obj`` if possible.

    Returns the Receipt directly when ``obj`` is one; returns
    ``obj.provenance`` when ``obj`` is a typed result; returns ``None`` for
    anything else (e.g., a NeuroSpace or a bare ndarray)."""
    if isinstance(obj, Receipt):
        return obj
    if hasattr(obj, "provenance") and isinstance(obj.provenance, Receipt):
        return obj.provenance
    return None


def _space_of(obj: Any) -> Any:
    """Return a concrete ``NeuroSpace`` if one can be extracted from ``obj``,
    else ``None``.

    Recognises three input shapes:
      - a result/container that exposes ``.space`` (SearchlightResult,
        ROIExtractionResult, NeuroVol, NeuroVec, LogicalNeuroVol).  We unwrap
        the inner ``.space`` first because ``NeuroVol``/``NeuroVec`` proxy
        ``dim``/``spacing``/``trans`` to it but do not themselves implement
        :meth:`NeuroSpace.compatible_with`.
      - a ``NeuroSpace`` itself (duck-typed: implements ``compatible_with``).
      - anything else returns ``None`` (Receipt-only inputs fall back to the
        hash path).
    """
    inner = getattr(obj, "space", None)
    if inner is not None and hasattr(inner, "compatible_with"):
        return inner
    if hasattr(obj, "compatible_with") and hasattr(obj, "trans"):
        return obj
    return None


def _space_hash_of(obj: Any) -> str:
    rec = receipt_of(obj)
    if rec is not None:
        return rec.input_space_hash
    return hash_neurospace(obj)


def _mask_hash_of(obj: Any) -> str:
    rec = receipt_of(obj)
    if rec is not None:
        return rec.mask_hash
    if obj is None:
        return "none"
    arr = np.asarray(obj)
    return hash_ndarray(arr)


def assert_same_space(a: Any, b: Any) -> None:
    """Raise ``ValueError`` if ``a`` and ``b`` do not share the same spatial
    contract.  Each may be a result object, a ``NeuroSpace``, or a ``Receipt``.

    When both sides expose a concrete ``NeuroSpace`` (directly or via ``.space``),
    the check routes through ``NeuroSpace.compatible_with``, which compares only
    the spatial axes and affine.  This is the correct semantics for the
    neuroimaging mask/data contract: a 4-D BOLD's spatial space is compatible
    with a 3-D mask in the same world frame, but Receipt hashes alone include
    the time axis and would reject that case.

    Receipt-only inputs (no structured space available) keep the hash-based
    comparison.  That is intentionally stricter: a bare ``Receipt`` does not
    retain the structured metadata needed to distinguish "only time differs"
    from "spatial shape or affine differs."
    """
    sa = _space_of(a)
    sb = _space_of(b)
    if sa is not None and sb is not None:
        try:
            sa.compatible_with(sb)
        except ValueError as exc:
            raise SpaceMismatchError(
                f"assert_same_space: spatial contract mismatch — {exc}"
            ) from None
        return

    ha = _space_hash_of(a)
    hb = _space_hash_of(b)
    if ha != hb:
        raise SpaceMismatchError(
            "assert_same_space: input_space_hash mismatch\n"
            f"  a.input_space_hash = {ha!r}\n"
            f"  b.input_space_hash = {hb!r}"
        )


def assert_same_mask(a: Any, b: Any) -> None:
    """Raise ``ValueError`` if ``a`` and ``b`` do not share the same mask.

    Each may be a result object, a Receipt, or a bare ndarray-like mask."""
    ha = _mask_hash_of(a)
    hb = _mask_hash_of(b)
    if ha != hb:
        raise MaskMismatchError(
            "assert_same_mask: mask_hash mismatch\n"
            f"  a.mask_hash = {ha!r}\n"
            f"  b.mask_hash = {hb!r}"
        )


def diff_receipts(a: Any, b: Any) -> dict:
    """Return a structured diff of fields that differ between the receipts
    of ``a`` and ``b``.  Returns an empty dict when they agree.  Either side
    may be a result object or a Receipt; passing a non-Receipt raises
    ``TypeError`` (the helpers above accept NeuroSpaces / ndarrays since
    they only care about one hash)."""
    ra = receipt_of(a)
    rb = receipt_of(b)
    if ra is None or rb is None:
        raise TypeError(
            "diff_receipts requires Receipt-bearing inputs; "
            f"got a={type(a).__name__!r}, b={type(b).__name__!r}"
        )
    return ra.diff(rb)
