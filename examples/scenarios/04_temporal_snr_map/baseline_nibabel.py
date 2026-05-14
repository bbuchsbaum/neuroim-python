"""Baseline: temporal SNR map using raw nibabel + numpy.

This is the compact form a nibabel user writes for a standard BOLD QC map:
load dense data, validate a mask, reduce along time, and return a 3-D NIfTI.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np


def temporal_snr_map(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Return a masked 3-D temporal SNR image.

    Raises
    ------
    ValueError
        If the BOLD image is not 4-D, the mask shape does not match the
        BOLD spatial shape, or the mask affine differs from the BOLD affine.
    """
    data = np.asarray(bold_img.get_fdata(dtype=np.float64))
    mask = np.asarray(mask_img.get_fdata(dtype=np.float64)) > 0

    if data.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got {data.ndim}D")
    if mask.shape != data.shape[:3]:
        raise ValueError(
            f"mask shape {mask.shape} does not match BOLD spatial shape {data.shape[:3]}"
        )
    if not np.allclose(mask_img.affine, bold_img.affine):
        raise ValueError("mask affine does not match BOLD affine")

    mean = data.mean(axis=3)
    std = data.std(axis=3)
    tsnr = np.zeros(data.shape[:3], dtype=np.float64)
    valid = mask & (std > 0)
    tsnr[valid] = mean[valid] / std[valid]

    return nib.Nifti1Image(tsnr, bold_img.affine, header=bold_img.header.copy())
