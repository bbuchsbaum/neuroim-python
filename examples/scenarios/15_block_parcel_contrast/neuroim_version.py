"""Rewrite: block-design parcel contrast through neuroim containers."""

from __future__ import annotations

import numpy as np

import neuroim as ni


def schaefer_fixture(atlas: ni.DenseNeuroVol) -> ni.atlas.VolumetricAtlas:
    """Wrap the synthetic fixture atlas as a typed Schaefer-like atlas."""
    ids = sorted(int(v) for v in np.unique(atlas.data) if int(v) != 0)
    labels = tuple(ni.atlas.AtlasLabel(i, f"parcel_{i}") for i in ids)
    return ni.atlas.schaefer_200(
        atlas,
        labels=labels,
        delivery_backend="scenario_fixture",
    )


def block_parcel_contrast(
    bold: ni.DenseNeuroVec,
    atlas: ni.DenseNeuroVol | ni.atlas.VolumetricAtlas,
    condition: np.ndarray,
) -> ni.ParcelEffectResult:
    """Compute a task-minus-rest parcel effect map.

    The core analysis is now exactly the desired public shape:
    ``bold.parcel_means(atlas).contrast(condition)``. Parcel label alignment,
    same-space checking, atlas provenance, and map projection are owned by
    neuroim containers rather than scenario-local scatter code.
    """
    return bold.parcel_means(atlas).contrast(
        condition,
        positive_name="task",
        negative_name="rest",
    )
