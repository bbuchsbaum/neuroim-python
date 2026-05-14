"""Rewrite: mean BOLD time series in a world-mm sphere using the neuroim API.

Two surfaces are exposed:

- :func:`mean_series_in_sphere_at_mni` — the simplest correct neuroim
  form.  Returns a bare ndarray to match the baseline's signature for
  the parity test.
- :func:`mean_series_in_sphere_at_mni_typed` — the canonical mission
  form.  Returns the full
  :class:`~neuroim.results.ROIExtractionResult` carrying per-voxel
  values ``(T, V)``, the ROI's voxel ``coords``, the ROI's ``space``,
  and the provenance :class:`~neuroim.results.Receipt`.

Both forms name the operation the baseline hand-codes:
``vec.series_roi_world(mni, radius=...)`` replaces affine inversion,
spacing-weighted distance maps, masking, bounds-checking, and raw
array indexing.  The typed form additionally carries provenance
forward.

Pain points surfaced during this scenario are filed in :file:`REPORT.md`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import neuroim as ni
from neuroim.results import ROIExtractionResult


def mean_series_in_sphere_at_mni(
    bold: ni.NeuroVec,
    mni_xyz: Sequence[float],
    radius_mm: float,
) -> np.ndarray:
    """Return the time-mean of voxels inside ``radius_mm`` of ``mni_xyz``."""
    return bold.series_roi_world(mni_xyz, radius=radius_mm).values.mean(axis=1)


def mean_series_in_sphere_at_mni_typed(
    bold: ni.NeuroVec,
    mni_xyz: Sequence[float],
    radius_mm: float,
) -> ROIExtractionResult:
    """Return the full typed ROI extraction for the sphere at ``mni_xyz``.

    The canonical neuroim form: per-voxel ``values`` ``(T, V)``, ROI
    voxel ``coords``, ROI ``space``, and a provenance ``Receipt`` all
    travel with the result.
    """
    return bold.series_roi_world(mni_xyz, radius=radius_mm)
