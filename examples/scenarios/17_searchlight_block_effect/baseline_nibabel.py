"""Baseline: local block-effect searchlight in nibabel + numpy."""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np


def _block_effect(ts: np.ndarray, condition: np.ndarray) -> float:
    """Task-minus-rest mean for a time-by-voxel neighborhood matrix."""
    condition = np.asarray(condition, dtype=bool)
    if condition.ndim != 1:
        raise ValueError("condition must be a 1-D boolean vector")
    if not np.any(condition) or np.all(condition):
        raise ValueError("condition must contain both task and rest samples")
    if ts.shape[0] != condition.size:
        raise ValueError(
            f"condition length {condition.size} != time length {ts.shape[0]}"
        )
    return float(ts[condition].mean() - ts[~condition].mean())


def local_block_effect_searchlight(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    condition: np.ndarray,
    *,
    radius_mm: float,
) -> tuple[np.ndarray, nib.Nifti1Image, dict[str, Any]]:
    """Compute a spherical local block-effect map.

    This baseline checks array shapes but deliberately does not check that
    the mask affine matches the BOLD affine. A same-shape foreign mask
    therefore yields a plausible but spatially wrong result.
    """
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")

    bold = np.asarray(bold_img.dataobj, dtype=np.float64)
    mask = np.asarray(mask_img.dataobj, dtype=bool)
    condition = np.asarray(condition, dtype=bool)
    if bold.ndim != 4:
        raise ValueError(f"expected 4-D BOLD, got {bold.ndim}-D")
    if mask.shape != bold.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != BOLD shape {bold.shape[:3]}")
    if condition.size != bold.shape[3]:
        raise ValueError(
            f"condition length {condition.size} != BOLD time length {bold.shape[3]}"
        )

    spacing = np.abs(np.diag(np.asarray(bold_img.affine)[:3, :3]))
    centers = np.argwhere(mask)
    out = np.full(mask.shape, np.nan, dtype=np.float64)
    values = np.empty(centers.shape[0], dtype=np.float64)

    for row, center in enumerate(centers):
        half_width = np.ceil(radius_mm / spacing).astype(int)
        lower = np.maximum(0, center - half_width)
        upper = np.minimum(np.asarray(mask.shape), center + half_width + 1)
        grids = np.meshgrid(
            np.arange(lower[0], upper[0]),
            np.arange(lower[1], upper[1]),
            np.arange(lower[2], upper[2]),
            indexing="ij",
        )
        coords = np.column_stack([g.ravel() for g in grids])
        dist = np.sqrt(np.sum(((coords - center) * spacing) ** 2, axis=1))
        coords = coords[dist <= radius_mm]
        coords = coords[mask[coords[:, 0], coords[:, 1], coords[:, 2]]]
        ts = bold[coords[:, 0], coords[:, 1], coords[:, 2], :].T
        value = _block_effect(ts, condition)
        values[row] = value
        out[tuple(center)] = value

    img = nib.Nifti1Image(out.astype(np.float32), bold_img.affine)
    summary = {
        "n_centers": int(centers.shape[0]),
        "radius_mm": float(radius_mm),
        "max_effect": float(np.nanmax(out)),
        "method_name": None,
        "provenance_chain": None,
    }
    return values, img, summary
