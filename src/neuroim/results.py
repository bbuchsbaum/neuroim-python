"""Typed result objects for ROI and searchlight workflows.

Per the consensus decision matrix (sticky post-01KRKFEWY2 in the
``neuroim-python-pythonic-value`` mote topic), neuroim returns typed, frozen
result objects from analysis workflows instead of bare ndarrays.  The
result objects carry the values, the coordinates, the spatial frame, and a
``Receipt`` with provenance fields that make silent space/mask mismatches
visible.

Numeric projection (``result.values``, ``result.map_to_volume()``) preserves
the prior R-shaped return values so legacy callers can keep working while
new code reads provenance and metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .neuro_space import NeuroSpace
    from .neuro_vol import NeuroVol


__all__ = [
    "Receipt",
    "ROIExtractionResult",
    "SearchlightResult",
    "hash_ndarray",
    "hash_neurospace",
    "RECEIPT_NIFTI_PREFIX",
]


# Marker prefix used to mark a NIfTI 'comment' extension (ecode 6) as a
# neuroim Receipt payload, so we can recognise it on read without false
# positives on user comments. Format: ``<prefix><utf-8 json>``.
RECEIPT_NIFTI_PREFIX = "neuroim/receipt/v1:"


def hash_ndarray(arr: Optional[np.ndarray]) -> str:
    """Stable SHA-256 hash of an array's bytes and shape/dtype."""
    if arr is None:
        return "none"
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def hash_neurospace(space: Optional["NeuroSpace"]) -> str:
    """Stable SHA-256 hash of a NeuroSpace's identifying fields."""
    if space is None:
        return "none"
    parts = [
        str(tuple(int(d) for d in np.asarray(space.dim))),
        str(tuple(float(s) for s in np.asarray(space.spacing))),
        str(tuple(float(o) for o in np.asarray(space.origin))),
        hash_ndarray(np.asarray(space.trans, dtype=float)),
    ]
    h = hashlib.sha256("|".join(parts).encode("ascii"))
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Receipt:
    """Provenance for ROI / searchlight outputs.

    Fields are designed to catch the silent space/mask mismatches that
    account for most neuroimaging bugs.  Receipts are content-addressable:
    identical inputs produce identical hashes.
    """

    input_space_hash: str
    mask_hash: str
    radius: Optional[float]
    n_voxels: int
    method_name: str
    seed: Optional[int]
    neuroim_version: str
    source_affine_hash: str

    def diff(self, other: "Receipt") -> dict:
        """Return a mapping of field names to (self, other) tuples for fields
        that differ.  Empty dict when receipts agree.

        Used by ``require_compatible`` and ``neuroim.verify.diff_receipts`` to
        produce structured assertion messages.
        """
        out: dict = {}
        for f in (
            "input_space_hash",
            "mask_hash",
            "radius",
            "n_voxels",
            "method_name",
            "seed",
            "neuroim_version",
            "source_affine_hash",
        ):
            a, b = getattr(self, f), getattr(other, f)
            if a != b:
                out[f] = (a, b)
        return out

    def to_json(self) -> str:
        """Serialize this Receipt to a JSON string.

        All Receipt fields are JSON-friendly scalars (strings, ints, floats,
        or None), so this is lossless and stable across Python versions.
        """
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "Receipt":
        """Re-hydrate a Receipt from a :meth:`to_json` payload."""
        data = json.loads(payload)
        return cls(**{field: data[field] for field in cls.__dataclass_fields__})

    def to_nifti_extension_bytes(self) -> bytes:
        """Serialize this Receipt to a NIfTI 'comment' extension payload.

        The payload is the marker prefix :data:`RECEIPT_NIFTI_PREFIX`
        followed by the JSON form, encoded as UTF-8.  ``from_nifti_extension``
        recovers the Receipt; foreign comment extensions are silently
        ignored on read.
        """
        return (RECEIPT_NIFTI_PREFIX + self.to_json()).encode("utf-8")

    @classmethod
    def from_nifti_extension_bytes(cls, payload: bytes) -> Optional["Receipt"]:
        """Recover a Receipt from a NIfTI 'comment' extension payload.

        Returns ``None`` if the payload does not carry the neuroim marker.
        """
        text = payload.rstrip(b"\x00").decode("utf-8", errors="replace")
        if not text.startswith(RECEIPT_NIFTI_PREFIX):
            return None
        try:
            return cls.from_json(text[len(RECEIPT_NIFTI_PREFIX):])
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def merge(self, other: "Receipt", *, method_name: Optional[str] = None) -> "Receipt":
        """Combine two upstream receipts into one downstream receipt.

        Intermediate ops that consume two typed-result inputs should call
        ``merge`` to carry provenance forward.  Raises ``ValueError`` if the
        upstream receipts disagree on ``input_space_hash`` or ``mask_hash`` —
        that's the silent-mismatch case the mission promises to catch.

        Parameters
        ----------
        other
            The other upstream receipt.
        method_name
            Optional explicit method name for the downstream op.  If omitted,
            uses ``f"{self.method_name}+{other.method_name}"`` (or just one
            side if the other is empty).
        """
        space_diff = self.input_space_hash != other.input_space_hash
        # mask_hash "none" is a wildcard: a stage that didn't apply a mask
        # is compatible with a downstream stage that does.  Mismatch only
        # qualifies as a silent-bug case when BOTH sides have real masks.
        both_have_masks = self.mask_hash != "none" and other.mask_hash != "none"
        mask_diff = both_have_masks and self.mask_hash != other.mask_hash
        if space_diff or mask_diff:
            details = []
            if space_diff:
                details.append(
                    f"input_space_hash: {self.input_space_hash!r} vs "
                    f"{other.input_space_hash!r}"
                )
            if mask_diff:
                details.append(
                    f"mask_hash: {self.mask_hash!r} vs {other.mask_hash!r}"
                )
            raise ValueError(
                "Receipt.merge: upstream receipts disagree:\n  " + "\n  ".join(details)
            )

        if method_name is None:
            parts = [p for p in (self.method_name, other.method_name) if p]
            method_name = "+".join(parts) if parts else ""

        # Adopt the more-specific mask: when one side is "none" the other
        # wins; when both are real and equal (the only case left after the
        # check above) we use self's hash.
        if self.mask_hash == "none":
            merged_mask = other.mask_hash
        else:
            merged_mask = self.mask_hash
        return replace(
            self,
            method_name=method_name,
            mask_hash=merged_mask,
            n_voxels=max(self.n_voxels, other.n_voxels),
            seed=self.seed if self.seed == other.seed else None,
            source_affine_hash=(
                self.source_affine_hash
                if self.source_affine_hash == other.source_affine_hash
                else "merged"
            ),
        )


def _current_version() -> str:
    from . import __version__

    return __version__


@dataclass(frozen=True)
class SearchlightResult:
    """Typed searchlight return value.

    Numeric projection: ``.values`` matches the prior bare-ndarray output of
    ``searchlight()``; ``.map_to_volume()`` matches the prior
    ``DenseNeuroVol`` output.  New callers can additionally read ``.centers``,
    ``.space``, and ``.provenance``.
    """

    values: np.ndarray
    centers: np.ndarray
    space: "NeuroSpace"
    radius: float
    shape: str
    provenance: Receipt
    method_name: str = ""

    def map_to_volume(self, *, dtype: Any = np.float64, fill: float = np.nan) -> "NeuroVol":
        """Place each scalar value at its searchlight centre in a NeuroVol.

        The returned :class:`~neuroim.neuro_vol.DenseNeuroVol` carries this
        result's :class:`Receipt` as ``vol.provenance``, so a downstream
        :meth:`~neuroim.neuro_vol.NeuroVol.to_nibabel` call can embed the
        Receipt in the NIfTI header (see PAIN-6 / Scenario 05).
        """
        from .neuro_vol import DenseNeuroVol

        if self.space.ndim != 3:
            raise ValueError(
                f"map_to_volume requires a 3-D space; got {self.space.ndim}-D"
            )
        out = np.full(tuple(int(d) for d in self.space.dim[:3]), fill, dtype=dtype)
        if self.centers.size == 0:
            vol = DenseNeuroVol(out, self.space)
        else:
            ix, iy, iz = self.centers[:, 0], self.centers[:, 1], self.centers[:, 2]
            out[ix, iy, iz] = self.values
            vol = DenseNeuroVol(out, self.space)
        vol.provenance = self.provenance
        return vol

    def to_nibabel(self, *, cls: Any = None) -> Any:
        """Convert this searchlight map to a nibabel image via map_to_volume."""
        return self.map_to_volume().to_nibabel(cls=cls)

    def require_compatible(self, other: Any) -> None:
        """Assert that ``other`` shares this result's space (and mask, when
        applicable).  Raises ``ValueError`` with a structured Receipt diff on
        mismatch.  ``other`` may be a result object, a ``NeuroSpace``, or a
        ``Receipt``."""
        _require_compatible(self.provenance, other, self_label="SearchlightResult")

    def to_dataframe(self):
        """Optional pandas projection.  pandas is imported lazily so the
        package does not require it.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pandas is required for SearchlightResult.to_dataframe()"
            ) from exc
        return pd.DataFrame(
            {
                "x": self.centers[:, 0],
                "y": self.centers[:, 1],
                "z": self.centers[:, 2],
                "value": np.asarray(self.values).ravel(),
            }
        )


@dataclass(frozen=True)
class ROIExtractionResult:
    """Typed return value for ``series_roi`` / values_roi extraction."""

    values: np.ndarray
    coords: np.ndarray
    space: "NeuroSpace"
    mask_hash: str
    provenance: Receipt

    @property
    def n_voxels(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_timepoints(self) -> Optional[int]:
        if self.values.ndim < 2:
            return None
        return int(self.values.shape[0])

    def require_compatible(self, other: Any) -> None:
        """Assert that ``other`` shares this result's space (and mask, when
        applicable).  Raises ``ValueError`` with a structured Receipt diff on
        mismatch.  ``other`` may be a result object, a ``NeuroSpace``, or a
        ``Receipt``."""
        _require_compatible(self.provenance, other, self_label="ROIExtractionResult")


def make_receipt(
    *,
    input_space: Optional["NeuroSpace"] = None,
    mask_data: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    n_voxels: int = 0,
    method_name: str = "",
    seed: Optional[int] = None,
    source_affine: Optional[np.ndarray] = None,
) -> Receipt:
    """Build a content-addressable Receipt for an analysis result."""
    return Receipt(
        input_space_hash=hash_neurospace(input_space),
        mask_hash=hash_ndarray(mask_data),
        radius=radius,
        n_voxels=int(n_voxels),
        method_name=method_name,
        seed=seed,
        neuroim_version=_current_version(),
        source_affine_hash=hash_ndarray(
            None if source_affine is None else np.asarray(source_affine, dtype=float)
        ),
    )


def _require_compatible(
    self_receipt: Receipt, other: Any, *, self_label: str = "result"
) -> None:
    """Shared body for ``Result.require_compatible``.  Accepts another result
    object, a NeuroSpace, or a Receipt as ``other``."""
    if isinstance(other, Receipt):
        other_receipt = other
    elif hasattr(other, "provenance") and isinstance(other.provenance, Receipt):
        other_receipt = other.provenance
    else:
        # Treat as a NeuroSpace-like; check space hash only.
        from .neuro_space import NeuroSpace  # noqa: F401  (type hint)

        space_hash = hash_neurospace(other)
        if self_receipt.input_space_hash != space_hash:
            raise ValueError(
                f"{self_label}: input space differs from supplied space\n"
                f"  expected input_space_hash = {self_receipt.input_space_hash!r}\n"
                f"  got                       = {space_hash!r}"
            )
        return

    diff = self_receipt.diff(other_receipt)
    blocking = {k: v for k, v in diff.items() if k in {"input_space_hash", "mask_hash"}}
    if blocking:
        lines = [
            f"{k}: self={a!r} vs other={b!r}" for k, (a, b) in blocking.items()
        ]
        raise ValueError(
            f"{self_label}: receipts disagree on space/mask:\n  " + "\n  ".join(lines)
        )
