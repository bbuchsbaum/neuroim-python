"""Rewrite: parcellated connectome through neuroim containers.

``DenseNeuroVec.parcel_means(atlas)`` owns the same-space gate against the
BOLD's spatial frame and records a ``parcel_means`` Receipt; the connectome is
then a single ``np.corrcoef`` over the typed ``(n_time, n_parcels)`` matrix.
The result carries the extraction's provenance so the matrix stays traceable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroim import DenseNeuroVec, DenseNeuroVol
from neuroim.atlas import VolumetricAtlas
from neuroim.clustered_neuro_vec import ClusteredNeuroVec
from neuroim.results import Receipt


@dataclass(frozen=True)
class ConnectomeResult:
    """A parcellated connectome plus the provenance of its extraction."""

    labels: np.ndarray  # sorted parcel ids, shape (N,)
    matrix: np.ndarray  # Pearson correlation matrix, shape (N, N)
    provenance: Receipt


def parcel_timeseries_matrix(parcels: ClusteredNeuroVec) -> np.ndarray:
    """Return the ``(n_time, n_parcels)`` per-parcel mean-series matrix.

    Built from the curated public surface (``cluster_ids`` +
    ``cluster_timeseries``) so the column order is the sorted parcel ids.
    """
    return np.column_stack(
        [parcels.cluster_timeseries(cid) for cid in parcels.cluster_ids]
    )


def parcel_connectome(
    bold: DenseNeuroVec, atlas: DenseNeuroVol | VolumetricAtlas
) -> ConnectomeResult:
    """Build the parcel-to-parcel Pearson connectome from a BOLD + atlas.

    The first-class ``parcel_means`` extraction owns the same-space gate and
    provenance threading, so the connectome itself is one ``np.corrcoef``.
    """
    parcels = bold.parcel_means(atlas)
    ts = parcel_timeseries_matrix(parcels)  # (n_time, n_parcels)
    matrix = np.corrcoef(ts, rowvar=False)
    return ConnectomeResult(
        labels=np.asarray(parcels.cluster_ids, dtype=np.int64),
        matrix=matrix,
        provenance=parcels.provenance,
    )
