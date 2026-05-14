"""Baseline: ROI mean time series via raw ``nibabel`` + ``numpy``.

This is the form a competent nibabel user writes today.  The function
shape is fixed by the friction the mission claims to remove:

- the user must materialize the full 4-D volume to apply the mask;
- the user must hand-check that mask shape and affine agree with the
  BOLD's spatial frame, and decide which axis is voxels vs time;
- the result is a bare ndarray with no spatial provenance.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np


def roi_mean_timeseries(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
) -> np.ndarray:
    """Return the mean BOLD time series across voxels where ``mask_img > 0``.

    Parameters
    ----------
    bold_img
        4-D NIfTI BOLD volume, shape ``(nx, ny, nz, nt)``.
    mask_img
        3-D NIfTI mask, shape ``(nx, ny, nz)``.

    Returns
    -------
    np.ndarray
        1-D array of length ``nt``: the mean across mask voxels at each
        timepoint.

    Raises
    ------
    ValueError
        If shapes or affines disagree, or if the mask is empty.
    """
    bold = bold_img.get_fdata()
    mask = mask_img.get_fdata().astype(bool)

    if bold.ndim != 4:
        raise ValueError(f"expected 4-D BOLD, got {bold.ndim}-D")
    if mask.ndim != 3:
        raise ValueError(f"expected 3-D mask, got {mask.ndim}-D")
    if bold.shape[:3] != mask.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match BOLD spatial shape "
            f"{bold.shape[:3]}"
        )
    if not np.allclose(bold_img.affine, mask_img.affine):
        raise ValueError("mask affine does not match BOLD affine")
    if not mask.any():
        raise ValueError("mask is empty")

    return bold[mask].mean(axis=0)
