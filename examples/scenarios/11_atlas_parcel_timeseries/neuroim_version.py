"""Atlas-based parcel time series -- neuroim rewrite.

Target: build a ``ClusteredNeuroVec`` whose ``ts`` matrix holds the per-parcel
mean BOLD time series, with same-space gating against the BOLD's spatial frame
and provenance recording the atlas payload.
"""

from __future__ import annotations

from neuroim import DenseNeuroVec, DenseNeuroVol
from neuroim.clustered_neuro_vec import ClusteredNeuroVec


def parcel_timeseries(
    bold: DenseNeuroVec, atlas: DenseNeuroVol
) -> ClusteredNeuroVec:
    """Return a ``ClusteredNeuroVec`` whose columns are per-parcel mean series.

    The first-class API owns the same-space gate and provenance threading, so
    the scenario call site is intentionally one line.
    """
    return bold.parcel_means(atlas)
