"""Threshold-and-cluster on a stat map -- scipy + numpy baseline.

Canonical post-stats workflow in task fMRI: given a stat map (e.g.,
z-scores or t-values), threshold at |z| > threshold within a brain
mask, label connected components, and report a per-cluster table with
voxel count, peak value, and peak world-mm coordinate.

This baseline mirrors what users write today with raw ``scipy.ndimage``
and bookkeeping: ``np.abs`` for two-tailed thresholding, ``mask``
gating, ``scipy.ndimage.label`` for connected components,
``np.argmax`` over per-cluster voxels for the peak, and
``affine @ [i, j, k, 1]`` for the world-mm conversion.

Output: a pandas DataFrame with columns
``[cluster_id, n_voxels, peak_value, peak_x_mm, peak_y_mm, peak_z_mm]``,
sorted by descending ``n_voxels``.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


def _structure_for_connectivity(connect: str) -> np.ndarray:
    if connect == "6-connect":
        return ndimage.generate_binary_structure(3, 1)
    if connect == "18-connect":
        return ndimage.generate_binary_structure(3, 2)
    if connect == "26-connect":
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError(f"unknown connectivity {connect!r}")


def cluster_table_from_stat_map(
    stat_img: nib.spatialimages.SpatialImage,
    mask_img: nib.spatialimages.SpatialImage,
    *,
    threshold: float = 2.3,
    connect: str = "26-connect",
    min_extent: int = 0,
) -> pd.DataFrame:
    """Return a cluster table for ``|stat| > threshold`` inside ``mask``."""
    if stat_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(
            f"mask shape {mask_img.shape[:3]} != stat shape {stat_img.shape[:3]}"
        )
    if not np.allclose(stat_img.affine, mask_img.affine):
        raise ValueError("mask affine does not match stat-map affine")

    data = np.asanyarray(stat_img.dataobj, dtype=np.float64)
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    affine = np.asarray(stat_img.affine, dtype=float)

    suprathreshold = mask & (np.abs(data) > threshold)
    structure = _structure_for_connectivity(connect)
    labels, n_clusters = ndimage.label(suprathreshold, structure=structure)

    rows = []
    for cid in range(1, n_clusters + 1):
        cluster_mask = labels == cid
        n_voxels = int(cluster_mask.sum())
        if n_voxels < min_extent:
            continue
        cluster_values = data[cluster_mask]
        # Peak by absolute value; report the signed value at the peak.
        peak_local_idx = int(np.argmax(np.abs(cluster_values)))
        peak_value = float(cluster_values[peak_local_idx])
        cluster_coords = np.argwhere(cluster_mask)
        peak_ijk = cluster_coords[peak_local_idx]
        peak_world = affine @ np.array([*peak_ijk, 1.0])
        rows.append(
            {
                "cluster_id": cid,
                "n_voxels": n_voxels,
                "peak_value": peak_value,
                "peak_x_mm": float(peak_world[0]),
                "peak_y_mm": float(peak_world[1]),
                "peak_z_mm": float(peak_world[2]),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("n_voxels", ascending=False, kind="stable").reset_index(
            drop=True
        )
        # Re-number cluster_id by descending size so the largest cluster is #1.
        table["cluster_id"] = np.arange(1, len(table) + 1)
    return table


def synthesize_stat_map(
    shape_3d: tuple[int, int, int] = (16, 16, 12),
    voxel_size_mm: tuple[float, float, float] = (3.0, 3.0, 3.5),
    *,
    seed: int = 0xBEAD,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Build a deterministic stat-map + mask pair with two real clusters.

    One positive cluster (z ~ +3.5) and one negative cluster (z ~ -3.2),
    embedded in noise that stays below |z| = 2.0 elsewhere.  Returns a
    (stat_img, mask_img) pair of nibabel Nifti1Images sharing the same
    affine.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape_3d
    sx, sy, sz = voxel_size_mm
    affine = np.diag([sx, sy, sz, 1.0]).astype(float)

    data = 0.8 * rng.standard_normal((nx, ny, nz))
    # Positive cluster in the upper-front-right quadrant (well inside
    # the ellipsoid mask defined below).
    data[10:13, 10:13, 6:9] += 3.5
    # Negative cluster in the lower-back-left quadrant.  Placed to land
    # *inside* the ellipsoid mask -- the obvious lower-corner placement
    # falls outside on a 16 x 16 x 12 grid.
    data[4:7, 4:7, 3:5] -= 3.2

    mask = np.ones((nx, ny, nz), dtype=bool)
    # Brain-shaped ellipsoid mask: keep voxels inside; both clusters fall
    # inside.
    ii, jj, kk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    cx, cy, cz = (nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2
    rx, ry, rz = nx * 0.45, ny * 0.45, nz * 0.45
    mask = (
        ((ii - cx) / rx) ** 2
        + ((jj - cy) / ry) ** 2
        + ((kk - cz) / rz) ** 2
    ) <= 1.0

    stat_img = nib.Nifti1Image(data, affine)
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    return stat_img, mask_img
