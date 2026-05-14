"""Baseline: resample native-space BOLD to template, then compute tSNR."""

from __future__ import annotations

import hashlib
from typing import Any

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def _space_hash(img: nib.Nifti1Image) -> str:
    return hashlib.sha256(
        np.asarray(img.affine, dtype=np.float64).tobytes()
        + np.asarray(img.shape[:3], dtype=np.int64).tobytes()
    ).hexdigest()


def resample_to_template(
    native_bold: nib.Nifti1Image,
    template_bold: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Resample a native-space 4-D image to the template image grid."""
    return resample_from_to(native_bold, template_bold, order=1)


def temporal_snr_map(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Compute a masked temporal-SNR map in the resampled/template space."""
    data = np.asarray(bold_img.get_fdata(dtype=np.float64))
    mask = np.asarray(mask_img.get_fdata(dtype=np.float64)) > 0
    if data.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got {data.ndim}D")
    if mask.shape != data.shape[:3]:
        raise ValueError("mask shape does not match BOLD spatial shape")
    if not np.allclose(mask_img.affine, bold_img.affine):
        raise ValueError("mask affine does not match BOLD affine")
    mean = data.mean(axis=3)
    std = data.std(axis=3)
    out = np.zeros(data.shape[:3], dtype=np.float64)
    valid = mask & (std > 0)
    out[valid] = mean[valid] / std[valid]
    return nib.Nifti1Image(out, bold_img.affine)


def manifest(
    native_bold: nib.Nifti1Image,
    template_bold: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
) -> dict[str, Any]:
    """Manual manifest a careful baseline user must keep in sync."""
    mask = np.asarray(mask_img.get_fdata()).astype(bool)
    return {
        "method_name": "resample_to_template+temporal_snr",
        "source_space_hash": _space_hash(native_bold),
        "target_space_hash": _space_hash(template_bold),
        "mask_hash": hashlib.sha256(mask.tobytes()).hexdigest(),
        "resample_order": 1,
    }


def native_to_template_tsnr(
    native_bold: nib.Nifti1Image,
    template_bold: nib.Nifti1Image,
    template_mask: nib.Nifti1Image,
) -> tuple[nib.Nifti1Image, dict[str, Any]]:
    """Return template-space tSNR plus a hand-written provenance manifest."""
    resampled = resample_to_template(native_bold, template_bold)
    return temporal_snr_map(resampled, template_mask), manifest(
        native_bold, template_bold, template_mask
    )

