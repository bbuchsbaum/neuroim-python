"""Typed atlas/parcellation objects for neuroim workflows.

This module is intentionally a thin in-core atlas layer.  It owns typed
metadata, source confidence, label tables, space checks, and conversion into
neuroim containers.  It does not own broad downloader catalogs, large bundled
atlas payloads, surface visualization, or TemplateFlow integration at import
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .clustered_neuro_vol import ClusteredNeuroVol
from .neuro_vol import DenseNeuroVol, LogicalNeuroVol
from .results import hash_ndarray


__all__ = [
    "AtlasLabel",
    "AtlasProvenance",
    "AtlasSpec",
    "AtlasArtifact",
    "AtlasProvider",
    "VolumetricAtlas",
    "schaefer_200",
]


@dataclass(frozen=True)
class AtlasLabel:
    """One integer label in a parcellation image."""

    id: int
    name: str
    hemi: Optional[str] = None
    network: Optional[str] = None
    rgba: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class AtlasProvenance:
    """Source and confidence metadata for an atlas artifact."""

    family: str
    model: str
    canonical_source: str
    delivery_backend: str
    citation_doi: str
    source_url: str
    source_ref: str
    template_space: str
    resolution: Optional[str]
    label_table_source: str
    image_hash: str
    label_table_hash: str
    confidence: str
    lineage: str
    notes: str = ""


@dataclass(frozen=True)
class AtlasSpec:
    """Dependency-free request object for optional atlas providers."""

    family: str
    representation: str = "volume"
    space: Optional[str] = None
    resolution: Optional[str] = None
    model: Optional[str] = None
    parcels: Optional[int] = None
    networks: Optional[int] = None
    backend: Optional[str] = None
    density: Optional[str] = None


@dataclass(frozen=True)
class AtlasArtifact:
    """Metadata for one source artifact used to build an atlas."""

    role: str
    source_url: str
    source_ref: str
    hash: str
    backend: str
    format: Optional[str] = None
    path: Optional[str] = None
    template_space: Optional[str] = None
    confidence: str = "unknown"
    notes: str = ""


@runtime_checkable
class AtlasProvider(Protocol):
    """Structural interface for optional atlas source providers."""

    def get(self, spec: AtlasSpec) -> "VolumetricAtlas":
        """Return a typed atlas for ``spec``."""
        ...


@dataclass(frozen=True)
class VolumetricAtlas:
    """Integer-labelled atlas image plus label/source metadata.

    Parameters
    ----------
    id, name, version
        Stable atlas identity fields.
    label_image
        A 3-D integer-labelled :class:`~neuroim.neuro_vol.DenseNeuroVol`.
        The image's ``space`` is the atlas spatial contract.
    labels
        Label rows for all non-background integer IDs that should be exposed
        to users and carried through parcel reductions.
    background
        Integer background label.  Defaults to zero.
    provenance
        Source, backend, hash, and confidence metadata.
    """

    id: str
    name: str
    version: str
    label_image: DenseNeuroVol
    labels: Tuple[AtlasLabel, ...]
    background: int
    provenance: AtlasProvenance

    def __post_init__(self) -> None:
        if type(self.label_image) is not DenseNeuroVol:
            raise TypeError("label_image must be a DenseNeuroVol")
        if self.label_image.ndim != 3:
            raise ValueError("VolumetricAtlas requires a 3-D label image")
        ids = [int(label.id) for label in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError("atlas label ids must be unique")
        if int(self.background) in ids:
            raise ValueError("background id must not appear in labels")
        image_ids = {
            int(value)
            for value in np.unique(np.asarray(self.label_image.data, dtype=np.int32))
            if int(value) != int(self.background)
        }
        metadata_ids = set(ids)
        missing = sorted(image_ids - metadata_ids)
        extra = sorted(metadata_ids - image_ids)
        if missing:
            raise ValueError(
                "atlas label image contains ids with missing metadata: "
                f"{missing}"
            )
        if extra:
            raise ValueError(
                "atlas labels contain ids absent from label image: "
                f"{extra}"
            )

    @property
    def space(self):
        """The atlas spatial contract."""
        return self.label_image.space

    @property
    def label_ids(self) -> np.ndarray:
        """Sorted array of label IDs in metadata order."""
        return np.asarray([label.id for label in self.labels], dtype=np.int32)

    @property
    def label_names(self) -> Tuple[str, ...]:
        """Tuple of label names in metadata order."""
        return tuple(label.name for label in self.labels)

    def to_clustered_vol(self) -> ClusteredNeuroVol:
        """Convert this atlas to the clustered volume used by parcel reducers."""
        label_data = np.asarray(self.label_image.data, dtype=np.int32)
        mask = LogicalNeuroVol(label_data != int(self.background), self.space)
        label_map = {label.name: int(label.id) for label in self.labels}
        cvol = ClusteredNeuroVol(mask, label_data, label_map=label_map)
        cvol.atlas_provenance = self.provenance
        return cvol


def _labels_hash(labels: Sequence[AtlasLabel]) -> str:
    payload = "|".join(
        f"{int(label.id)}:{label.name}:{label.hemi or ''}:{label.network or ''}"
        for label in labels
    )
    return hash_ndarray(np.frombuffer(payload.encode("utf-8"), dtype=np.uint8))


def _default_schaefer_labels(label_image: DenseNeuroVol) -> Tuple[AtlasLabel, ...]:
    data = np.asarray(label_image.data, dtype=np.int32)
    ids = sorted(int(v) for v in np.unique(data) if int(v) != 0)
    return tuple(AtlasLabel(id=label_id, name=f"parcel_{label_id}") for label_id in ids)


def _make_schaefer_atlas(
    label_image: DenseNeuroVol,
    *,
    parcels: int,
    networks: int,
    labels: Optional[Sequence[AtlasLabel]] = None,
    resolution: str = "2mm",
    template_space: str = "MNI152NLin6Asym",
    delivery_backend: str = "provided",
    source_url: str = "https://github.com/ThomasYeoLab/CBIG",
    source_ref: Optional[str] = None,
    confidence: str = "high",
) -> VolumetricAtlas:
    """Build a typed Schaefer2018 atlas from an already-loaded label image.

    The canonical upstream is ThomasYeoLab/CBIG Schaefer2018.  TemplateFlow or
    a future optional fetcher may provide ``label_image``; this constructor
    records the source chain without importing those backends.
    """
    if parcels <= 0:
        raise ValueError("parcels must be positive")
    if networks not in {7, 17}:
        raise ValueError("networks must be 7 or 17")

    label_rows = tuple(labels) if labels is not None else _default_schaefer_labels(label_image)
    if source_ref is None:
        source_ref = (
            f"Schaefer2018_{parcels}Parcels_{networks}Networks_order_"
            f"FSLMNI152_{resolution.replace('mm', '')}mm.nii.gz"
        )
    image_data = np.asarray(label_image.data, dtype=np.int32)
    provenance = AtlasProvenance(
        family="schaefer",
        model="Schaefer2018",
        canonical_source="ThomasYeoLab/CBIG",
        delivery_backend=delivery_backend,
        citation_doi="10.1093/cercor/bhx179",
        source_url=source_url,
        source_ref=source_ref,
        template_space=template_space,
        resolution=resolution,
        label_table_source="CBIG freeview_lut",
        image_hash=hash_ndarray(image_data),
        label_table_hash=_labels_hash(label_rows),
        confidence=confidence,
        lineage="CBIG-distributed Schaefer2018 parcellation label image.",
    )
    return VolumetricAtlas(
        id=f"schaefer-{parcels}-{networks}",
        name=f"Schaefer-{parcels}-{networks}networks",
        version="2018",
        label_image=label_image,
        labels=label_rows,
        background=0,
        provenance=provenance,
    )


def schaefer_200(
    label_image: DenseNeuroVol,
    *,
    networks: int = 7,
    labels: Optional[Sequence[AtlasLabel]] = None,
    resolution: str = "2mm",
    template_space: str = "MNI152NLin6Asym",
    delivery_backend: str = "provided",
) -> VolumetricAtlas:
    """Construct a typed Schaefer-200 atlas from a provided label image."""
    return _make_schaefer_atlas(
        label_image,
        parcels=200,
        networks=networks,
        labels=labels,
        resolution=resolution,
        template_space=template_space,
        delivery_backend=delivery_backend,
    )
