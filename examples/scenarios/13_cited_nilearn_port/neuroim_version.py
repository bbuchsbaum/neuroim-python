"""neuroim port of the cited Nilearn seed-to-voxel correlation tutorial.

The Nilearn primitives (`NiftiSpheresMasker`, `NiftiMasker`, manual
correlation) collapse to three neuroim calls:

  * ``bold.series_roi_world(seed_xyz, radius=8.0)`` -- spherical seed
    extraction with the same-space contract enforced on the BOLD
    spatial frame.
  * ``bold.series_roi(brain_roi)`` -- mask -> in-mask time series with
    ``assert_same_space`` invoked inside the API.
  * ``NeuroVol.from_array(corr, mask.space, coords=coords)`` -- scatter
    the correlation values back to a spatial frame.

Provenance is threaded onto the output map so a downstream consumer can
inspect "what seed at what radius produced this connectivity map?" after
the NIfTI extension round-trip lands.

See ``REPORT.md`` for the line-for-line nilearn -> neuroim mapping table.
"""

from __future__ import annotations

import numpy as np

import neuroim as ni
from neuroim.results import make_receipt


def _pearson_seed_to_voxels(
    seed_ts: np.ndarray, voxel_by_time: np.ndarray
) -> np.ndarray:
    """Z-score / dot / ``n-1``: identical to the baseline helper.

    Inlined here (rather than imported from ``baseline_nibabel``) because
    the scenario test loads both modules via ``importlib`` outside a
    package context, so the relative import would fail.  Keeping the
    formula textually identical is the parity contract.
    """
    seed = np.asarray(seed_ts, dtype=np.float64)
    voxels = np.asarray(voxel_by_time, dtype=np.float64)
    seed_z = (seed - seed.mean()) / seed.std(ddof=1)
    voxels_z = (
        voxels - voxels.mean(axis=1, keepdims=True)
    ) / voxels.std(axis=1, keepdims=True, ddof=1)
    n = seed_z.size
    return (voxels_z @ seed_z) / (n - 1)


def seed_to_voxel_correlation_map(
    bold: ni.DenseNeuroVec,
    mask: ni.LogicalNeuroVol,
    *,
    seed_xyz,
    radius_mm: float = 8.0,
) -> ni.DenseNeuroVol:
    """Pearson seed-to-voxel correlation map via neuroim spatial contracts."""
    seed_xyz = np.asarray(seed_xyz, dtype=float)
    # NiftiSpheresMasker substitute: returns ROIExtractionResult; .values is
    # (n_time, n_voxels).  Mean across voxels yields the sphere mean.
    seed_extraction = bold.series_roi_world(seed_xyz, radius=radius_mm)
    seed_ts = np.asarray(seed_extraction.values).mean(axis=1)

    # NiftiMasker substitute: same-space contract is enforced inside
    # NeuroVec.series_roi via assert_same_space.
    coords = np.argwhere(np.asarray(mask.data, dtype=bool))
    roi = ni.ROICoords(coords, space=mask.space)
    voxel_by_time = np.asarray(bold.series_roi(roi).values).T

    corr_values = _pearson_seed_to_voxels(seed_ts, voxel_by_time)
    corr_map = ni.NeuroVol.from_array(corr_values, mask.space, coords=coords)
    corr_map.provenance = make_receipt(
        input_space=bold.space,
        mask_data=np.asarray(mask.data, dtype=bool),
        n_voxels=int(coords.shape[0]),
        radius=float(radius_mm),
        method_name="seed_to_voxel_correlation_map",
        source_affine=bold.space.trans,
    )
    return corr_map
