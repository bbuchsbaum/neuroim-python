"""Baseline: multi-subject concat via raw ``nibabel`` + ``numpy``.

This is the form a competent nibabel user writes today.  The function
shape is fixed by the friction the mission claims to remove:

- the user must hand-check that the two subjects' affines agree
  before concatenation;
- the user must hand-stack on the right axis;
- the result is a bare ndarray with no spatial provenance.
"""

from __future__ import annotations

from typing import Sequence

import nibabel as nib
import numpy as np


def concat_subjects_and_mean(
    subjects: Sequence[nib.Nifti1Image],
    mask_img: nib.Nifti1Image,
) -> np.ndarray:
    """Concatenate per-subject BOLD time series and return the mean across mask voxels.

    Parameters
    ----------
    subjects
        Iterable of 4-D ``Nifti1Image`` BOLD volumes.  Each must agree
        on spatial shape and affine — the function raises otherwise.
    mask_img
        3-D NIfTI mask in the same spatial frame as ``subjects[0]``.

    Returns
    -------
    np.ndarray
        1-D array of length ``sum(s.shape[3] for s in subjects)``.
    """
    if len(subjects) < 2:
        raise ValueError("expected at least two subjects to concat")

    ref = subjects[0]
    for s in subjects[1:]:
        if s.shape[:3] != ref.shape[:3]:
            raise ValueError(
                f"shape mismatch: subject has spatial shape {s.shape[:3]}, "
                f"reference has {ref.shape[:3]}"
            )
        if not np.allclose(s.affine, ref.affine):
            raise ValueError("affine mismatch across subjects — subjects do not share a spatial frame")

    if mask_img.shape != ref.shape[:3]:
        raise ValueError("mask spatial shape does not match subject shape")
    if not np.allclose(mask_img.affine, ref.affine):
        raise ValueError("mask affine does not match subject affine")

    stacked = np.concatenate([s.get_fdata() for s in subjects], axis=3)
    mask = mask_img.get_fdata().astype(bool)
    return stacked[mask].mean(axis=0)
