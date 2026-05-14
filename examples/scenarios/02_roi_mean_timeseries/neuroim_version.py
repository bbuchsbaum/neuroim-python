"""Rewrite: ROI mean time series through the neuroim public API.

Two surfaces are exposed:

- :func:`roi_mean_timeseries` — the simplest correct neuroim form,
  matching the baseline's return type for the parity test.
- :func:`roi_mean_timeseries_typed` — the canonical mission form.
  Returns an :class:`~neuroim.results.ROIExtractionResult` carrying
  per-voxel values, coords, the spatial frame, and a
  :class:`~neuroim.results.Receipt`.

What the rewrite gets *for free* compared to the baseline:

- ``bold.series_roi(roi)`` does the mask-space alignment check inside
  the contract layer; no hand-coded ``shape ==`` / ``np.allclose(affine)``.
- The typed form ships a Receipt so a downstream caller can verify the
  output was produced from the expected BOLD and mask without trusting
  the upstream code.

What it does *not* get from neuroim today (filed in ``REPORT.md``):

- there is no ``LogicalNeuroVol.to_roi_coords()`` helper, so the
  rewrite still has to write ``np.argwhere(np.asarray(mask.data))``;
- there is no ``bold.mean_in_roi(mask)`` one-liner that returns the
  collapsed length-``nt`` ndarray directly.
"""

from __future__ import annotations

import numpy as np

import neuroim as ni
from neuroim.results import ROIExtractionResult


def _roi_from_mask(mask: ni.LogicalNeuroVol) -> ni.ROICoords:
    """Build an ROICoords from the True voxels of a LogicalNeuroVol.

    Filed as PAIN-4 in :file:`REPORT.md`: ``LogicalNeuroVol`` is a
    *mask* and yet there is no ``mask.to_roi_coords()`` convenience.
    The caller has to know that ``mask.coords()`` returns *all* grid
    coords (not just the True ones), and use ``np.argwhere`` instead.
    """
    coords = np.argwhere(np.asarray(mask.data))
    return ni.ROICoords(coords, space=mask.space)


def roi_mean_timeseries(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
) -> np.ndarray:
    """Return the mean BOLD time series across mask voxels.

    Mirrors the baseline's return type (bare ndarray) so the two
    implementations can be checked for numeric parity.

    Parameters
    ----------
    bold
        4-D :class:`~neuroim.NeuroVec`.
    mask
        3-D :class:`~neuroim.LogicalNeuroVol` in the same spatial frame
        as ``bold`` (alignment is contract-checked by ``series_roi``).
    """
    if bold.space.ndim != 4:
        raise ValueError(f"expected 4-D BOLD, got {bold.space.ndim}-D")
    if not np.asarray(mask.data).any():
        raise ValueError("mask is empty")
    roi = _roi_from_mask(mask)
    result = bold.series_roi(roi)
    return np.asarray(result.values).mean(axis=1)


def roi_mean_timeseries_typed(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
) -> ROIExtractionResult:
    """Return the per-voxel time series as a typed result with provenance.

    This is the canonical neuroim form: the spatial frame and a Receipt
    travel with the values, so downstream code (mean, regression, RSA)
    can validate compatibility instead of trusting a bare ndarray.
    Callers compute the mean themselves via ``result.values.mean(axis=1)``.
    """
    if bold.space.ndim != 4:
        raise ValueError(f"expected 4-D BOLD, got {bold.space.ndim}-D")
    if not np.asarray(mask.data).any():
        raise ValueError("mask is empty")
    roi = _roi_from_mask(mask)
    return bold.series_roi(roi)
