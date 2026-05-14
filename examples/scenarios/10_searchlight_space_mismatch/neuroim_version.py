"""Rewrite: local-mean searchlight through neuroim's typed API."""

from __future__ import annotations

import numpy as np

import neuroim as ni


def local_mean_searchlight(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
    *,
    radius: float = 2.1,
) -> ni.SearchlightResult:
    """Compute one scalar per mask voxel via ``searchlight_apply``."""
    return ni.searchlight_apply(
        mask,
        radius=radius,
        method=lambda a: float(np.asarray(a).mean()),
        data=bold,
        cores=0,
    )
