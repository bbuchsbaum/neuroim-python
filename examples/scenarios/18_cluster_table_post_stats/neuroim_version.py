"""Threshold-and-cluster on a stat map -- neuroim rewrite.

After S18 PAIN-1..5 closed, the rewrite collapses to one call:

    result = conn_comp(stat_vol, threshold=..., mask=mask, two_tailed=True)

and one column-shape projection to match the baseline's table format.
``conn_comp`` now (a) accepts a ``mask=`` keyword and invokes
``assert_same_space(stat_vol, mask)`` (PAIN-3), (b) accepts
``two_tailed=True`` to threshold ``|x.data| > threshold`` while keeping
signed peak values in the table (PAIN-4), (c) always populates
``peak_x_mm / peak_y_mm / peak_z_mm / peak_value`` columns in
``cluster_table`` (PAIN-5), and (d) populates a ``provenance`` Receipt
on the result (PAIN-2).  ``conn_comp`` is now in ``neuroim.__all__``
(PAIN-1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import neuroim as ni
from neuroim.connected_components import conn_comp


def cluster_table_from_stat_map(
    stat_vol: ni.DenseNeuroVol,
    mask: ni.LogicalNeuroVol,
    *,
    threshold: float = 2.3,
    connect: str = "26-connect",
    min_extent: int = 0,
) -> "_ClusterTableResult":
    """Cluster table with the same column shape the baseline emits."""
    result = conn_comp(
        stat_vol,
        threshold=threshold,
        connect=connect,
        local_maxima=False,
        mask=mask,
        two_tailed=True,
    )
    raw = result.cluster_table
    if raw is None or raw.empty:
        return _ClusterTableResult(
            table=pd.DataFrame(
                columns=[
                    "cluster_id",
                    "n_voxels",
                    "peak_value",
                    "peak_x_mm",
                    "peak_y_mm",
                    "peak_z_mm",
                ]
            ),
            provenance=result.provenance,
        )

    table = raw.rename(columns={"N": "n_voxels"})[
        ["n_voxels", "peak_value", "peak_x_mm", "peak_y_mm", "peak_z_mm"]
    ].copy()
    if min_extent > 0:
        table = table[table["n_voxels"] >= min_extent].reset_index(drop=True)
    table = table.sort_values(
        "n_voxels", ascending=False, kind="stable"
    ).reset_index(drop=True)
    table.insert(0, "cluster_id", np.arange(1, len(table) + 1, dtype=int))
    return _ClusterTableResult(table=table, provenance=result.provenance)


class _ClusterTableResult:
    """Thin wrapper preserving the prior (table + provenance) shape used
    by the scenario test.  The provenance now comes from the underlying
    ``ConnCompResult.provenance`` (PAIN-2), so this wrapper is sugar
    rather than a workaround."""

    __slots__ = ("table", "provenance")

    def __init__(self, table: pd.DataFrame, provenance) -> None:
        self.table = table
        self.provenance = provenance


def typed_from_nibabel(stat_img, mask_img):
    """Adapter so the test can drive both paths from one fixture call."""
    space = ni.NeuroSpace.from_affine(
        np.asarray(stat_img.affine), stat_img.shape[:3]
    )
    stat_vol = ni.DenseNeuroVol(
        np.asarray(stat_img.dataobj, dtype=np.float64), space
    )
    mask_vol = ni.LogicalNeuroVol(
        np.asarray(mask_img.dataobj).astype(bool), space
    )
    return stat_vol, mask_vol
