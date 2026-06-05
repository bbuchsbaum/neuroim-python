"""Rewrite: parcellated connectome through neuroim containers.

``DenseNeuroVec.parcel_means(atlas)`` owns the same-space gate against the
BOLD's spatial frame and records a ``parcel_means`` Receipt; the connectome is
then a single ``ClusteredNeuroVec.connectome()`` call that returns a typed
``ConnectomeResult`` whose Receipt chains the extraction
(``parcel_means+connectome``), so the matrix stays traceable to the atlas and
input space that produced it.
"""

from __future__ import annotations

from neuroim import ConnectomeResult, DenseNeuroVec, DenseNeuroVol
from neuroim.atlas import VolumetricAtlas


def parcel_connectome(
    bold: DenseNeuroVec, atlas: DenseNeuroVol | VolumetricAtlas
) -> ConnectomeResult:
    """Build the parcel-to-parcel Pearson connectome from a BOLD + atlas.

    Two first-class operations: ``parcel_means`` owns the same-space gate and
    provenance, and ``connectome`` reduces the parcel time-series into a typed
    ``ConnectomeResult`` (``.labels``, ``.matrix``, ``.provenance``).
    """
    return bold.parcel_means(atlas).connectome(metric="correlation")
