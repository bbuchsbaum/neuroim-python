"""Baseline: block-design parcel contrast in raw nibabel + numpy.

Given a 4-D BOLD image, a 3-D integer atlas, and a boolean condition vector,
compute the task-minus-rest mean signal in each parcel. The return value is a
tuple of sorted parcel ids, parcel effects, and a 3-D NIfTI where every atlas
voxel receives its parcel's effect.

The baseline intentionally performs only a shape check. A careful nibabel user
can add an affine check by hand, but the point of the scenario is that every
call site has to remember to do so. Scenario 15's mismatch test proves that a
same-shape, different-affine atlas otherwise produces a plausible wrong map.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np


def block_parcel_contrast(
    bold_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    condition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Return ``(labels, effects, effect_map)`` for task-minus-rest parcels."""
    bold = np.asarray(bold_img.dataobj, dtype=np.float64)
    atlas = np.asarray(atlas_img.dataobj, dtype=np.int32)
    condition = np.asarray(condition, dtype=bool)

    if bold.ndim != 4:
        raise ValueError(f"expected 4-D BOLD, got {bold.ndim}-D")
    if atlas.shape != bold.shape[:3]:
        raise ValueError(
            f"atlas shape {atlas.shape} != bold spatial shape {bold.shape[:3]}"
        )
    if condition.shape != (bold.shape[-1],):
        raise ValueError(
            f"condition shape {condition.shape} != time axis {(bold.shape[-1],)}"
        )
    if not condition.any() or condition.all():
        raise ValueError("condition must contain both task and rest samples")

    labels = np.unique(atlas)
    labels = labels[labels != 0]
    effects = np.empty(labels.size, dtype=np.float64)
    effect_map = np.zeros(atlas.shape, dtype=np.float64)

    for col, label in enumerate(labels):
        parcel_ts = bold[atlas == label].mean(axis=0)
        effect = parcel_ts[condition].mean() - parcel_ts[~condition].mean()
        effects[col] = effect
        effect_map[atlas == label] = effect

    return labels, effects, nib.Nifti1Image(effect_map.astype(np.float32), bold_img.affine)

