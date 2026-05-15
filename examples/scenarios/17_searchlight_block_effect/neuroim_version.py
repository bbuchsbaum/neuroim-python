"""Rewrite: local block-effect searchlight through neuroim typed results."""

from __future__ import annotations

from typing import Any

import numpy as np

import neuroim as ni


def block_effect(ts: np.ndarray, condition: np.ndarray) -> float:
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
    bold: ni.DenseNeuroVec,
    mask: ni.LogicalNeuroVol,
    condition: np.ndarray,
    *,
    radius_mm: float,
) -> tuple[ni.SearchlightResult, dict[str, Any]]:
    """Compute a typed local block-effect searchlight result."""
    condition = np.asarray(condition, dtype=bool)

    def local_effect(ts: np.ndarray) -> float:
        return block_effect(ts, condition)

    local_effect.__name__ = "local_block_effect"
    result = ni.searchlight_apply(
        mask,
        radius=radius_mm,
        method=local_effect,
        data=bold,
        nonzero=True,
    )
    volume = result.map_to_volume()
    summary = {
        "n_centers": int(result.centers.shape[0]),
        "radius_mm": float(result.radius),
        "max_effect": float(np.nanmax(volume.data)),
        "method_name": result.provenance.method_name,
        "provenance_chain": result.provenance.method_name,
    }
    return result, summary
