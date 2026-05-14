"""Rewrite workflow: ROI-time-series + behavioral correlation through the neuroim API.

The same computation as :mod:`examples.draw_audit.baseline_nibabel`, but
through the typed/validated/provenance-bearing neuroim public surface.

What changes vs the baseline:
- ``ni.read_image`` (or ``NeuroVec.from_nibabel``) takes the place of
  ``nib.load`` + ``.get_fdata().astype(np.float64)``.
- The mask is bound to its own NeuroSpace; ``space.compatible_with`` (or
  ``verify.assert_same_space``) makes the cross-space check explicit
  instead of silently trusting that affines agree.
- ``vec.series_roi`` returns a typed ``ROIExtractionResult`` carrying
  ``values``, ``coords``, ``space``, ``mask_hash``, and ``provenance``
  — no manual mask-index bookkeeping, no F-vs-C ravel trap.
- The output is a ``NeuroVol`` built via ``NeuroVol.from_array`` and
  ``write_vol``/``to_nibabel``; the source affine is carried by the
  spatial contract, not handed off as a side argument.

The acceptance test compares this to the baseline:
  - Happy path: values match (numeric projection of the rewrite equals the
    baseline output).
  - Deliberate space mismatch: baseline silently produces a wrong-but-
    plausible correlation map; rewrite raises ``ValueError`` from the
    verifier with a Receipt diff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np

import neuroim as ni
from neuroim.results import ROIExtractionResult


def correlate_roi_with_regressor(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
    regressor: np.ndarray,
    *,
    out_path: Optional[Path] = None,
) -> Tuple[ni.NeuroVol, ROIExtractionResult]:
    """Compute per-voxel Pearson r against a behavioural regressor, inside a mask.

    Unlike the baseline, the spatial contract is checked up front and the
    extraction returns a Receipt-bearing result.
    """
    # Explicit spatial-contract check via the blessed verifier surface.
    # `verify.assert_same_space` now routes 4-D BOLD vs 3-D mask through
    # `NeuroSpace.compatible_with` (spatial dims + affine, time axis ignored)
    # and falls back to the strict Receipt-hash path for Receipt-only inputs.
    ni.verify.assert_same_space(bold, mask)

    # Build an ROI from the mask (every True voxel) and extract typed series.
    coords = np.argwhere(np.asarray(mask.data, dtype=bool))
    roi = ni.ROICoords(coords, space=mask.space)
    extract = bold.series_roi(roi)  # default return_legacy=False
    assert isinstance(extract, ROIExtractionResult)

    nt = extract.values.shape[0]
    regressor = np.asarray(regressor, dtype=np.float64)
    if regressor.shape != (nt,):
        raise ValueError(f"regressor must have shape ({nt},); got {regressor.shape}")

    # Per-voxel Pearson r.  No manual mask-index bookkeeping; the typed
    # ``coords`` array travels with the values.
    series = extract.values.astype(np.float64)  # (T, V)
    reg_c = regressor - regressor.mean()
    reg_n = np.linalg.norm(reg_c)
    if reg_n == 0:
        raise ValueError("regressor has zero variance")

    series_c = series - series.mean(axis=0, keepdims=True)
    series_n = np.linalg.norm(series_c, axis=0)
    safe = series_n > 0
    corrs = np.full(series.shape[1], np.nan, dtype=np.float64)
    corrs[safe] = (reg_c @ series_c[:, safe]) / (reg_n * series_n[safe])

    # Project back to the volume.  The space comes from ``extract.space`` —
    # the same contract the extraction was validated against.
    corr_3d = np.full(tuple(int(d) for d in extract.space.dim[:3]), np.nan, dtype=np.float64)
    corr_3d[extract.coords[:, 0], extract.coords[:, 1], extract.coords[:, 2]] = corrs
    corr_vol = ni.NeuroVol.from_array(corr_3d, space=extract.space)

    if out_path is not None:
        ni.write_vol(corr_vol, out_path)

    return corr_vol, extract
