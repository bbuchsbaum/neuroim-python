"""Baseline workflow: ROI-time-series + behavioral correlation in raw nibabel + numpy.

This is the form a competent nibabel user writes today.  No neuroim.  The
implementation is faithful but exhibits the friction points the mission
claims to remove: manual shape/affine checks, explicit ravel/reshape with
order assumptions, masked-array bookkeeping by index arithmetic, and no
provenance receipt on the output.

The acceptance test (:mod:`examples.draw_audit.test_audit`) compares this
implementation against :mod:`examples.draw_audit.neuroim_rewrite` for
numeric equivalence on the happy path and for diverging behaviour on the
deliberate space-mismatch case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np


def correlate_roi_with_regressor(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    regressor: np.ndarray,
    *,
    out_path: Optional[Path] = None,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Compute per-voxel Pearson r against a behavioural regressor, inside a mask.

    Parameters
    ----------
    bold_img
        4-D NIfTI image (x, y, z, t).
    mask_img
        3-D boolean NIfTI mask.  Must have the same shape and affine as
        ``bold_img`` on the spatial axes — but this function does NOT
        enforce affine agreement (that's the bug class the rewrite catches).
    regressor
        1-D array of length T.
    out_path
        Optional path to write the resulting 3-D correlation map.

    Returns
    -------
    corr_3d
        ``(nx, ny, nz)`` ndarray of Pearson correlations; voxels outside the
        mask are NaN.
    corr_img
        ``nib.Nifti1Image`` of the correlation map, sharing ``bold_img``'s
        affine.
    """
    bold = bold_img.get_fdata().astype(np.float64)
    mask = mask_img.get_fdata().astype(bool)

    # --- Shape check (user-written; affine is NOT checked here)
    if bold.shape[:3] != mask.shape:
        raise ValueError(
            f"bold spatial shape {bold.shape[:3]} != mask shape {mask.shape}"
        )

    # --- Behavioural regressor shape check
    nt = bold.shape[3]
    regressor = np.asarray(regressor, dtype=np.float64)
    if regressor.shape != (nt,):
        raise ValueError(
            f"regressor must have shape ({nt},); got {regressor.shape}"
        )

    # --- Manual: flatten 4-D into (n_voxels_in_mask, T) using mask indices.
    # NB: must use the same axis order (C by default in numpy) on both sides
    # or the mapping back will scramble.  This is exactly the F-vs-C trap
    # the audit calls out.
    mask_indices = np.argwhere(mask)
    n_vox = mask_indices.shape[0]
    masked_series = np.empty((n_vox, nt), dtype=np.float64)
    for i, (ix, iy, iz) in enumerate(mask_indices):
        masked_series[i, :] = bold[ix, iy, iz, :]

    # --- Per-voxel Pearson correlation
    reg_centered = regressor - regressor.mean()
    reg_norm = np.linalg.norm(reg_centered)
    if reg_norm == 0:
        raise ValueError("regressor has zero variance")

    series_centered = masked_series - masked_series.mean(axis=1, keepdims=True)
    series_norm = np.linalg.norm(series_centered, axis=1)
    safe = series_norm > 0
    corrs = np.full(n_vox, np.nan, dtype=np.float64)
    corrs[safe] = (series_centered[safe] @ reg_centered) / (series_norm[safe] * reg_norm)

    # --- Map back to 3-D space (mask_indices order must match the order
    # we used when filling masked_series; off-by-one or order='F' bugs
    # silently produce a scrambled map here)
    corr_3d = np.full(mask.shape, np.nan, dtype=np.float64)
    for i, (ix, iy, iz) in enumerate(mask_indices):
        corr_3d[ix, iy, iz] = corrs[i]

    # --- Save with the BOLD affine.  The mask's affine is silently ignored;
    # if the mask was in a different space, the output map is in the
    # BOLD space but the values are wrong.
    corr_img = nib.Nifti1Image(corr_3d, bold_img.affine, bold_img.header)
    if out_path is not None:
        nib.save(corr_img, out_path)

    return corr_3d, corr_img
