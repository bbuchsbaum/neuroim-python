"""neuroim rewrite for Scenario 06."""

from __future__ import annotations

import numpy as np

import neuroim as ni
from neuroim.results import make_receipt

from .baseline_nibabel import _correlate_seed_to_voxels


def seed_correlation_map(bold_img, mask_img, *, seed_xyz):
    """Compute a seed-to-voxel map using neuroim spatial contracts."""
    bold = ni.NeuroVec.from_nibabel(bold_img)
    mask = ni.NeuroVol.from_nibabel(mask_img)
    bold.space.compatible_with(mask.space)

    seed_ts = bold.series_at_world(np.asarray(seed_xyz, dtype=float))
    coords = np.argwhere(np.asarray(mask.data, dtype=bool))
    roi = ni.ROICoords(coords, space=mask.space)
    voxel_by_time = bold.series_roi(roi).values.T

    corr_values = _correlate_seed_to_voxels(seed_ts, voxel_by_time)
    corr_map = ni.NeuroVol.from_array(corr_values, mask.space, coords=coords)
    corr_map.provenance = make_receipt(
        input_space=bold.space,
        mask_data=np.asarray(mask.data, dtype=bool),
        n_voxels=int(coords.shape[0]),
        method_name="seed_correlation_map",
        seed=None,
        source_affine=bold.space.trans,
    )
    return corr_map
