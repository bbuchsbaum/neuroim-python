"""Baseline: masked temporal-SNR then FWHM-mm smoothing in nibabel + scipy."""

from __future__ import annotations

import math
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def smoothed_tsnr_qc(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    fwhm_mm: float,
    smoothing_mask_img: nib.Nifti1Image | None = None,
) -> tuple[nib.Nifti1Image, dict[str, Any]]:
    """Compute masked tSNR, smooth it, and return image plus QC summary.

    This baseline checks shapes but deliberately does not check affines. A
    same-shape mask with a different affine therefore produces a plausible
    but spatially wrong smoothed map.
    """
    if fwhm_mm <= 0:
        raise ValueError("fwhm_mm must be positive")

    bold = np.asarray(bold_img.dataobj, dtype=np.float64)
    mask = np.asarray(mask_img.dataobj, dtype=bool)
    if bold.ndim != 4:
        raise ValueError(f"expected 4-D BOLD, got {bold.ndim}-D")
    if mask.shape != bold.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != BOLD shape {bold.shape[:3]}")

    mean = bold.mean(axis=-1)
    std = bold.std(axis=-1)
    tsnr = np.zeros(mask.shape, dtype=np.float64)
    valid = mask & (std > 0)
    tsnr[valid] = mean[valid] / std[valid]

    voxel_sizes_mm = np.abs(np.diag(np.asarray(bold_img.affine)[:3, :3]))
    sigma_voxels = (fwhm_mm * _FWHM_TO_SIGMA) / voxel_sizes_mm
    smoothed = ndimage.gaussian_filter(tsnr, sigma=sigma_voxels, truncate=4.0)

    smooth_mask = mask
    if smoothing_mask_img is not None:
        smooth_mask = np.asarray(smoothing_mask_img.dataobj, dtype=bool)
        if smooth_mask.shape != tsnr.shape:
            raise ValueError(
                f"smoothing mask shape {smooth_mask.shape} != tSNR shape {tsnr.shape}"
            )
    smoothed = np.where(smooth_mask, smoothed, tsnr)

    img = nib.Nifti1Image(smoothed.astype(np.float32), bold_img.affine)
    summary = {
        "p50": float(np.percentile(smoothed[mask], 50.0)),
        "p95": float(np.percentile(smoothed[mask], 95.0)),
        "n_voxels": int(mask.sum()),
        "fwhm_mm": float(fwhm_mm),
        "method_name": None,
        "provenance_chain": None,
    }
    return img, summary

