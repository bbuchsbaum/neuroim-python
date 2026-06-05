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
from dataclasses import asdict, dataclass, replace
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
    "make_receipt",
    "receipt_for",
    "chain_receipt",
    "OpParams",
    "RoiOpParams",
    "SearchlightParams",
    "TemporalReductionParams",
    "SpatialFilterParams",
    "ParcelContrastParams",
    "ConcatParams",
    "ResampleParams",
    "TemporalSliceParams",
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
        return cls(**{fname: data[fname] for fname in cls.__dataclass_fields__})

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
        if self.radius == other.radius:
            merged_radius = self.radius
        elif self.radius is None:
            merged_radius = other.radius
        elif other.radius is None:
            merged_radius = self.radius
        else:
            merged_radius = None
        return replace(
            self,
            method_name=method_name,
            mask_hash=merged_mask,
            radius=merged_radius,
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

    def map_to_volume(
        self, *, dtype: Any = np.float64, fill: float = 0.0
    ) -> "NeuroVol":
        """Project this ROI extraction back into a spatially-shaped container.

        - 1-D ``values`` (e.g. from :func:`~neuroim.roi.values_roi`) produces
          a 3-D :class:`~neuroim.neuro_vol.DenseNeuroVol` with one scalar
          per ROI voxel; voxels outside the ROI are ``fill``.
        - 2-D ``values`` shaped ``(n_timepoints, n_voxels)`` (e.g. from
          :func:`~neuroim.NeuroVec.series_roi`) produces a 4-D
          :class:`~neuroim.neuro_vec.DenseNeuroVec` whose time axis matches
          ``values.shape[0]``.

        The returned container carries this result's :class:`Receipt` as
        ``.provenance``, so a downstream
        :meth:`~neuroim.neuro_vol.NeuroVol.to_nibabel` /
        :meth:`~neuroim.neuro_vec.NeuroVec.to_nibabel` call embeds the
        Receipt in the NIfTI header (see PAIN-6 / Scenario 05 spec).
        """
        from .neuro_space import NeuroSpace
        from .neuro_vec import DenseNeuroVec
        from .neuro_vol import DenseNeuroVol

        coords = np.ascontiguousarray(self.coords, dtype=int)
        values = np.asarray(self.values)

        spatial_dim = tuple(int(d) for d in self.space.dim[:3])
        if values.ndim == 1:
            out = np.full(spatial_dim, fill, dtype=dtype)
            if values.size:
                out[coords[:, 0], coords[:, 1], coords[:, 2]] = values
            spatial_space = (
                self.space
                if self.space.ndim == 3
                else NeuroSpace.from_affine(self.space.affine, spatial_dim)
            )
            container = DenseNeuroVol(out, spatial_space)
        elif values.ndim == 2:
            nt = int(values.shape[0])
            out = np.full(spatial_dim + (nt,), fill, dtype=dtype)
            if values.size:
                # values is (nt, n_voxels) — scatter each row into out[..., t]
                out[coords[:, 0], coords[:, 1], coords[:, 2], :] = values.T
            shape4 = spatial_dim + (nt,)
            if self.space.ndim == 4:
                vec_space = NeuroSpace.from_affine(self.space.affine, shape4)
            elif self.space.ndim == 3:
                vec_space = NeuroSpace.from_affine(self.space.affine, shape4)
            else:
                raise ValueError(
                    f"map_to_volume cannot project 2-D values from a "
                    f"{self.space.ndim}-D space"
                )
            container = DenseNeuroVec(out, vec_space)
        else:
            raise ValueError(
                f"ROIExtractionResult.map_to_volume expects values.ndim in (1, 2); "
                f"got {values.ndim}"
            )

        container.provenance = self.provenance
        return container

    def to_nibabel(self, *, cls: Any = None) -> Any:
        """Convert this ROI extraction to a nibabel image via :meth:`map_to_volume`.

        The Receipt rides along as a NIfTI 'comment' header extension; a
        ``neuroim.read_image`` of the resulting file re-hydrates
        ``.provenance``.  See ``docs/spec/receipt-nifti-extension.md``.
        """
        return self.map_to_volume().to_nibabel(cls=cls)

    def require_compatible(self, other: Any) -> None:
        """Assert that ``other`` shares this result's space (and mask, when
        applicable).  Raises ``ValueError`` with a structured Receipt diff on
        mismatch.  ``other`` may be a result object, a ``NeuroSpace``, or a
        ``Receipt``."""
        _require_compatible(self.provenance, other, self_label="ROIExtractionResult")


@dataclass(frozen=True)
class OpParams:
    """Structural per-operation parameters used when constructing a Receipt.

    Subclass per operation family so call sites cannot silently omit fields
    that an op should always record.  Use :func:`receipt_for` rather than
    hand-listing kwargs into :func:`make_receipt` — it is the single
    structural path the structural-provenance contract relies on.
    """

    method_name: str
    radius: Optional[float] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class RoiOpParams(OpParams):
    """Op-params for ROI extraction (``series_roi``, ``values_roi``,
    ``series_roi_world``)."""


@dataclass(frozen=True)
class SearchlightParams(OpParams):
    """Op-params for searchlight analyses; ``radius`` is required."""

    def __post_init__(self) -> None:
        if self.radius is None:
            raise ValueError("SearchlightParams requires a numeric radius")


@dataclass(frozen=True)
class TemporalReductionParams(OpParams):
    """Op-params for temporal reductions (``temporal_snr``, future
    generic reducers)."""


@dataclass(frozen=True)
class SpatialFilterParams(OpParams):
    """Op-params for spatial filters such as ``gaussian_blur``."""


@dataclass(frozen=True)
class ParcelContrastParams(OpParams):
    """Op-params for parcel-level condition contrasts."""

    positive_name: str = "task"
    negative_name: str = "rest"


@dataclass(frozen=True)
class ConnectomeParams(OpParams):
    """Op-params for a parcel-to-parcel connectivity reduction.

    ``metric`` records how the matrix was formed (``"correlation"`` for a
    Pearson connectome, ``"covariance"`` for the unnormalized form) so a
    downstream consumer can tell what the off-diagonal entries mean.
    """

    metric: str = "correlation"


@dataclass(frozen=True)
class ConcatParams(OpParams):
    """Op-params for time-axis concat across inputs."""


@dataclass(frozen=True)
class ResampleParams(OpParams):
    """Op-params for spatial resampling between two NeuroSpaces.

    ``interpolation`` is the numeric order passed to the underlying
    resampler (0=nearest, 1=linear, 3=cubic).  Records the policy on
    the Receipt so a downstream consumer can tell *how* the data
    arrived in the target space.
    """

    interpolation: int = 1


@dataclass(frozen=True)
class TemporalSliceParams(OpParams):
    """Op-params for a pure time-axis slice of a 4-D :class:`NeuroVec`.

    Records the ``slice(start, stop, step)`` the caller applied so a
    downstream consumer can tell which timepoints the derived vec
    actually came from.  The defaults (``None``) match Python's
    ``slice(None)`` shorthand for "whole axis".
    """

    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None


def receipt_for(
    carrier: Any,
    *,
    mask: Any = None,
    n_voxels: int,
    params: OpParams,
    upstream: Any = None,
) -> Receipt:
    """Structural Receipt factory — call this from every result-producing op.

    ``carrier`` is the input data (a :class:`~neuroim.NeuroVec`, :class:`
    ~neuroim.NeuroVol`, or :class:`~neuroim.NeuroSpace`).  Its ``space``
    becomes the ``input_space`` and its ``trans`` becomes the
    ``source_affine``, so call sites stop hand-deriving those two.

    ``params`` is a typed :class:`OpParams` subclass naming the op and its
    parameters.  ``mask`` is the per-call mask payload (ndarray, mask vol,
    or ``None``).  ``n_voxels`` is the count the op produced/consumed.

    Existing :func:`make_receipt` remains as a thin shim for ad-hoc
    receipt construction; new internal call sites should use
    :func:`receipt_for` and an :class:`OpParams` subclass instead.
    """
    space = _carrier_space(carrier)
    source_affine = getattr(space, "trans", None) if space is not None else None
    receipt = make_receipt(
        input_space=space,
        mask_data=mask,
        n_voxels=int(n_voxels),
        method_name=params.method_name,
        radius=params.radius,
        seed=params.seed,
        source_affine=source_affine,
    )

    # Structural chain: when ``upstream`` carries a Receipt, merge it into
    # the new receipt so a multi-op pipeline (e.g. resample -> temporal_snr)
    # records every stage in ``method_name``.  Accepts a Receipt directly or
    # any object exposing ``.provenance``.
    upstream_receipt = None
    if isinstance(upstream, Receipt):
        upstream_receipt = upstream
    elif upstream is not None:
        upstream_receipt = getattr(upstream, "provenance", None)
    if isinstance(upstream_receipt, Receipt):
        receipt = upstream_receipt.merge(
            receipt,
            method_name=f"{upstream_receipt.method_name}+{params.method_name}",
        )
    return receipt


def chain_receipt(
    upstream: Any,
    *,
    params: OpParams,
    n_voxels: Optional[int] = None,
) -> Receipt:
    """Extend an upstream Receipt with a typed downstream operation.

    Use this when a downstream operation is a pure reduction/projection of an
    already-typed result and should preserve the upstream spatial contract
    rather than re-anchor provenance to the reduced container's own shape.
    """
    if isinstance(upstream, Receipt):
        upstream_receipt = upstream
    else:
        upstream_receipt = getattr(upstream, "provenance", None)
    if not isinstance(upstream_receipt, Receipt):
        raise ValueError("chain_receipt requires an upstream Receipt")

    return replace(
        upstream_receipt,
        method_name=f"{upstream_receipt.method_name}+{params.method_name}",
        n_voxels=(
            upstream_receipt.n_voxels
            if n_voxels is None
            else int(n_voxels)
        ),
    )


def _carrier_space(carrier: Any) -> Any:
    """Extract a NeuroSpace from a carrier, or pass through if ``carrier``
    already is one.  Returns ``None`` for non-space-bearing inputs."""
    if carrier is None:
        return None
    inner = getattr(carrier, "space", None)
    if inner is not None:
        return inner
    if hasattr(carrier, "trans") and hasattr(carrier, "dim"):
        return carrier
    return None


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
