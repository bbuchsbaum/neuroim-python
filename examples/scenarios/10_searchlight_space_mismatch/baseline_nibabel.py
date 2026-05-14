"""Baseline: local-mean searchlight with explicit space validation."""

from __future__ import annotations

import nibabel as nib
import numpy as np


def _sphere_coords(
    center: np.ndarray,
    shape: tuple[int, int, int],
    spacing: np.ndarray,
    radius: float,
) -> np.ndarray:
    voxel_radius = np.ceil(radius / spacing).astype(int)
    lower = np.maximum(0, np.floor(center - voxel_radius).astype(int))
    upper = np.minimum(shape, np.ceil(center + voxel_radius + 1).astype(int))
    grid = np.meshgrid(
        np.arange(lower[0], upper[0]),
        np.arange(lower[1], upper[1]),
        np.arange(lower[2], upper[2]),
        indexing="ij",
    )
    coords = np.column_stack([axis.ravel() for axis in grid])
    distances = np.sqrt(np.sum(((coords - center) * spacing) ** 2, axis=1))
    return coords[distances <= radius]


def local_mean_searchlight(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    radius: float = 2.1,
) -> nib.Nifti1Image:
    """Compute one scalar per mask voxel after checking shared space."""
    data = np.asarray(bold_img.get_fdata(dtype=np.float64))
    mask = np.asarray(mask_img.get_fdata(dtype=np.float64)) > 0
    if data.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got {data.ndim}D")
    if mask.shape != data.shape[:3]:
        raise ValueError("mask shape does not match BOLD spatial shape")
    if not np.allclose(mask_img.affine, bold_img.affine):
        raise ValueError("mask affine does not match BOLD affine")

    spacing = np.sqrt(np.sum(np.asarray(bold_img.affine[:3, :3]) ** 2, axis=0))
    out = np.full(mask.shape, np.nan, dtype=np.float64)
    for center in np.argwhere(mask):
        coords = _sphere_coords(center.astype(float), mask.shape, spacing, radius)
        roi = data[coords[:, 0], coords[:, 1], coords[:, 2], :]
        out[tuple(center)] = float(roi.mean())
    return nib.Nifti1Image(out, bold_img.affine)
