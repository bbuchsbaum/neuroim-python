"""nibabel + numpy port of Nilearn's seed-to-voxel correlation tutorial.

Mirrors the structure of
<https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html>
without importing nilearn.  Each public-facing function corresponds to a
named step in the Nilearn tutorial; see ``REPORT.md`` for the line-for-line
mapping table.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np


# ----------------------------------------------------------------------
# NiftiSpheresMasker substitute -- radius-8 mm sphere mean time series
# ----------------------------------------------------------------------


def _sphere_mask(
    shape_3d: tuple[int, int, int],
    affine: np.ndarray,
    seed_xyz: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """Return a boolean mask of voxels within ``radius_mm`` of ``seed_xyz``.

    Mirrors ``NiftiSpheresMasker(seeds=[seed_xyz], radius=radius_mm)``
    voxel selection: world-space Euclidean distance from the seed in mm.
    """
    nx, ny, nz = shape_3d
    ii, jj, kk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    voxel_xyz = np.stack(
        [ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], axis=0
    )
    world = (affine @ voxel_xyz)[:3].T.reshape(nx, ny, nz, 3)
    dists = np.linalg.norm(world - seed_xyz[None, None, None, :], axis=-1)
    return dists <= radius_mm


def extract_seed_time_series(
    bold_img: nib.spatialimages.SpatialImage,
    seed_xyz: np.ndarray,
    radius_mm: float = 8.0,
) -> np.ndarray:
    """Mean BOLD time series over a radius-``radius_mm`` sphere at ``seed_xyz``.

    Nilearn equivalent: ``NiftiSpheresMasker(seeds=[seed_xyz], radius=radius_mm)``.
    Raises ``ValueError`` if the sphere doesn't intersect the BOLD grid.
    """
    data = np.asanyarray(bold_img.dataobj, dtype=np.float64)
    affine = np.asarray(bold_img.affine, dtype=float)
    sphere = _sphere_mask(data.shape[:3], affine, np.asarray(seed_xyz), radius_mm)
    if not sphere.any():
        raise ValueError(
            f"sphere of radius {radius_mm} mm at {seed_xyz!r} does not "
            f"intersect the BOLD grid"
        )
    return data[sphere, :].mean(axis=0)


# ----------------------------------------------------------------------
# NiftiMasker substitute -- in-mask voxel time series
# ----------------------------------------------------------------------


def extract_voxel_time_series(
    bold_img: nib.spatialimages.SpatialImage,
    mask_img: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coords, voxel_by_time)`` for in-mask voxels.

    Nilearn equivalent: ``NiftiMasker(mask_img=mask).fit_transform(bold)``.
    Affine + shape agreement is enforced explicitly (Nilearn's
    NiftiMasker enforces affine internally; nibabel does not).
    """
    if bold_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(
            f"mask shape {mask_img.shape[:3]} != BOLD shape {bold_img.shape[:3]}"
        )
    if not np.allclose(bold_img.affine, mask_img.affine):
        raise ValueError("mask affine does not match BOLD affine")
    data = np.asanyarray(bold_img.dataobj, dtype=np.float64)
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    coords = np.argwhere(mask)
    return coords, data[mask, :]


# ----------------------------------------------------------------------
# Pearson correlation -- the dot-product trick from the Nilearn tutorial
# ----------------------------------------------------------------------


def _pearson_seed_to_voxels(
    seed_ts: np.ndarray, voxel_by_time: np.ndarray
) -> np.ndarray:
    """Z-score, dot, divide by ``n-1`` -- mirrors the Nilearn tutorial body."""
    seed = np.asarray(seed_ts, dtype=np.float64)
    voxels = np.asarray(voxel_by_time, dtype=np.float64)
    seed_z = (seed - seed.mean()) / seed.std(ddof=1)
    voxels_z = (
        voxels - voxels.mean(axis=1, keepdims=True)
    ) / voxels.std(axis=1, keepdims=True, ddof=1)
    n = seed_z.size
    return (voxels_z @ seed_z) / (n - 1)


# ----------------------------------------------------------------------
# End-to-end -- the cited tutorial's main path
# ----------------------------------------------------------------------


def seed_to_voxel_correlation_map(
    bold_img: nib.spatialimages.SpatialImage,
    mask_img: nib.spatialimages.SpatialImage,
    *,
    seed_xyz: np.ndarray,
    radius_mm: float = 8.0,
) -> nib.Nifti1Image:
    """Pearson seed-to-voxel correlation map at ``radius_mm`` mm sphere.

    Returns a 3-D ``Nifti1Image`` whose voxels in the mask contain the
    Pearson correlation between that voxel's BOLD time series and the
    seed-sphere mean.  Voxels outside the mask are zero.
    """
    seed_ts = extract_seed_time_series(bold_img, np.asarray(seed_xyz), radius_mm)
    coords, voxel_by_time = extract_voxel_time_series(bold_img, mask_img)
    corr_values = _pearson_seed_to_voxels(seed_ts, voxel_by_time)
    corr_map = np.zeros(np.asanyarray(mask_img.dataobj).shape, dtype=np.float64)
    corr_map[tuple(coords.T)] = corr_values
    return nib.Nifti1Image(corr_map, bold_img.affine)
