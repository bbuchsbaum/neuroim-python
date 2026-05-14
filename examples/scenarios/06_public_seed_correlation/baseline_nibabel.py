"""Raw nibabel/numpy baseline for Scenario 06.

The workflow mirrors the public Nilearn seed-to-voxel correlation examples:
choose a seed coordinate, extract its time series from a 4-D BOLD image,
correlate it with every in-mask voxel, and return a 3-D correlation map.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib


def _seed_voxel_from_world(affine: np.ndarray, seed_xyz) -> tuple[int, int, int]:
    ijk = np.linalg.inv(affine) @ np.array([*seed_xyz, 1.0], dtype=float)
    return tuple(int(v) for v in np.round(ijk[:3]))


def _correlate_seed_to_voxels(seed_ts: np.ndarray, voxel_by_time: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed_ts, dtype=np.float64)
    voxels = np.asarray(voxel_by_time, dtype=np.float64)
    seed = seed - seed.mean()
    voxels = voxels - voxels.mean(axis=1, keepdims=True)

    numerator = voxels @ seed
    denominator = np.linalg.norm(voxels, axis=1) * np.linalg.norm(seed)
    corr = np.zeros(voxels.shape[0], dtype=np.float64)
    np.divide(numerator, denominator, out=corr, where=denominator > 0)
    return corr


def seed_correlation_map(
    bold_img: nib.spatialimages.SpatialImage,
    mask_img: nib.spatialimages.SpatialImage,
    *,
    seed_xyz,
) -> nib.Nifti1Image:
    """Compute a seed-to-voxel map with explicit nibabel/numpy guardrails."""
    if bold_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(
            f"mask shape {mask_img.shape[:3]} does not match BOLD shape {bold_img.shape[:3]}"
        )
    if not np.allclose(bold_img.affine, mask_img.affine):
        raise ValueError("mask affine does not match BOLD affine")

    data = np.asanyarray(bold_img.dataobj, dtype=np.float64)
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    seed_ijk = _seed_voxel_from_world(np.asarray(bold_img.affine), seed_xyz)

    spatial_shape = data.shape[:3]
    if not all(0 <= seed_ijk[i] < spatial_shape[i] for i in range(3)):
        raise ValueError(f"seed coordinate maps outside BOLD grid: {seed_ijk}")

    seed_ts = data[seed_ijk[0], seed_ijk[1], seed_ijk[2], :]
    voxel_by_time = data[mask, :]
    corr_values = _correlate_seed_to_voxels(seed_ts, voxel_by_time)

    corr_map = np.zeros(mask.shape, dtype=np.float64)
    corr_map[mask] = corr_values
    return nib.Nifti1Image(corr_map, bold_img.affine)
