"""Atlas-based parcel time series -- neuroim rewrite.

Target: build a ``ClusteredNeuroVec`` whose ``ts`` matrix holds the per-parcel
mean BOLD time series, with same-space gating against the BOLD's spatial frame
and provenance recording the atlas payload.
"""

from __future__ import annotations

import numpy as np

from neuroim import DenseNeuroVec, DenseNeuroVol
from neuroim.atlas import AtlasLabel, VolumetricAtlas, _make_schaefer_atlas
from neuroim.clustered_neuro_vec import ClusteredNeuroVec


def typed_schaefer_fixture(atlas: DenseNeuroVol) -> VolumetricAtlas:
    """Wrap the scenario's integer label image as a typed atlas object."""
    ids = sorted(int(v) for v in np.unique(atlas.data) if int(v) != 0)
    labels = tuple(AtlasLabel(label_id, f"parcel_{label_id}") for label_id in ids)
    return _make_schaefer_atlas(
        atlas,
        parcels=len(ids),
        networks=7,
        labels=labels,
        delivery_backend="scenario_fixture",
        source_ref="scenario11_synthetic_schaefer_like_atlas",
    )


def parcel_timeseries(
    bold: DenseNeuroVec, atlas: DenseNeuroVol | VolumetricAtlas
) -> ClusteredNeuroVec:
    """Return a ``ClusteredNeuroVec`` whose columns are per-parcel mean series.

    The first-class API owns the same-space gate and provenance threading, so
    the scenario call site is intentionally one line.
    """
    return bold.parcel_means(atlas)
