"""Baseline: stack split 3-D NIfTI volumes with explicit affine validation."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_split_run(paths: list[Path]) -> nib.Nifti1Image:
    """Load one 3-D file per time point after checking shape and affine."""
    if not paths:
        raise ValueError("paths cannot be empty")
    imgs = [nib.load(str(path)) for path in paths]
    first = imgs[0]
    volumes = []
    for idx, img in enumerate(imgs):
        if img.shape != first.shape:
            raise ValueError(
                f"volume {idx} shape {img.shape} does not match {first.shape}"
            )
        if not np.allclose(img.affine, first.affine):
            raise ValueError(f"volume {idx} affine does not match volume 0 affine")
        volumes.append(np.asarray(img.get_fdata(dtype=np.float64)))
    data = np.stack(volumes, axis=3)
    return nib.Nifti1Image(data, first.affine)


def temporal_snr_from_split_run(
    paths: list[Path],
    mask_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Compute masked temporal SNR from a validated split-volume run."""
    bold = load_split_run(paths)
    data = np.asarray(bold.get_fdata(dtype=np.float64))
    mask = np.asarray(mask_img.get_fdata(dtype=np.float64)) > 0
    if mask.shape != data.shape[:3]:
        raise ValueError("mask shape does not match BOLD spatial shape")
    if not np.allclose(mask_img.affine, bold.affine):
        raise ValueError("mask affine does not match BOLD affine")
    mean = data.mean(axis=3)
    sd = data.std(axis=3)
    out = np.zeros(data.shape[:3], dtype=np.float64)
    valid = mask & (sd > 0)
    out[valid] = mean[valid] / sd[valid]
    return nib.Nifti1Image(out, bold.affine)
