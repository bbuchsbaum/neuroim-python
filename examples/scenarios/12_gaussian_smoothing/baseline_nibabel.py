"""Gaussian spatial smoothing at FWHM = 6 mm -- nibabel + scipy baseline.

Standard preprocessing pattern: read a stat-map NIfTI, convert FWHM-in-mm
to per-axis voxel-space sigma using the image's voxel sizes, apply
``scipy.ndimage.gaussian_filter`` with a per-axis sigma sequence, return
the smoothed image.

What this baseline does (and what it deliberately doesn't):

  Per-axis sigma
      Voxel sizes are read from ``diag(affine[:3, :3])`` and the FWHM is
      divided through to give the correct *isotropic-mm-space* sigma per
      axis.  This is the only honest way to smooth at FWHM = X mm.

  Shape check
      The data and (optional) mask arrays must share spatial shape.
      Standard nibabel-user paranoia.

  Affine check
      None.  Two volumes with identical ``(nx, ny, nz)`` and totally
      different affines look interchangeable here.  See
      ``test_baseline_silently_accepts_mismatched_affine_mask`` for the
      proof that the bug class is real on this surface.
"""

from __future__ import annotations

import math

import nibabel as nib
import numpy as np
from scipy import ndimage


_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def smooth_fwhm_mm(
    stat_img: nib.Nifti1Image,
    fwhm_mm: float,
    mask_img: nib.Nifti1Image | None = None,
) -> nib.Nifti1Image:
    """Return ``stat_img`` smoothed at ``fwhm_mm`` mm FWHM."""
    if fwhm_mm <= 0:
        raise ValueError("fwhm_mm must be positive")
    data = np.asarray(stat_img.dataobj, dtype=np.float64)
    voxel_sizes_mm = np.abs(np.diag(np.asarray(stat_img.affine)[:3, :3]))
    sigma_voxels = (fwhm_mm * _FWHM_TO_SIGMA) / voxel_sizes_mm

    smoothed = ndimage.gaussian_filter(data, sigma=sigma_voxels)
    if mask_img is not None:
        mask = np.asarray(mask_img.dataobj).astype(bool)
        if mask.shape != data.shape:
            raise ValueError(
                f"mask shape {mask.shape} != stat-map shape {data.shape}"
            )
        smoothed = np.where(mask, smoothed, 0.0)
    return nib.Nifti1Image(smoothed, stat_img.affine)
