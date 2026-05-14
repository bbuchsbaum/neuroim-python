"""Baseline: write a derived map + a hand-rolled sidecar JSON.

This is what a careful nibabel + numpy user does to make a derived
volume reproducible: write the .nii.gz, write a parallel `<name>.json`
manifest describing how it was computed, and let the collaborator
read both files together.

The sidecar is hand-curated. The user has to remember every field;
nothing in the pipeline guarantees the sidecar agrees with the .nii.gz
on shape, affine, or input hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np


def write_mean_volume_with_sidecar(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    out_nii: Path,
    out_json: Path,
    *,
    method_name: str = "mean_over_time",
) -> None:
    """Compute a per-voxel temporal mean inside the mask, write nii + sidecar."""
    bold = bold_img.get_fdata()
    mask = mask_img.get_fdata().astype(bool)
    if bold.shape[:3] != mask.shape:
        raise ValueError("mask/bold spatial shape mismatch")
    if not np.allclose(bold_img.affine, mask_img.affine):
        raise ValueError("mask/bold affine mismatch")

    derived = np.zeros(bold.shape[:3], dtype=np.float64)
    derived[mask] = bold[mask].mean(axis=1)

    img = nib.Nifti1Image(derived, bold_img.affine)
    nib.save(img, str(out_nii))

    manifest: Mapping[str, object] = {
        "method_name": method_name,
        "n_voxels": int(mask.sum()),
        "input_space_hash": hashlib.sha256(bold_img.affine.tobytes()).hexdigest(),
        "mask_hash": hashlib.sha256(mask.tobytes()).hexdigest(),
        "input_shape": list(bold.shape),
        "output_shape": list(derived.shape),
        "affine": bold_img.affine.tolist(),
    }
    out_json.write_text(json.dumps(manifest, indent=2))


def read_provenance(nii_path: Path, json_path: Path) -> Mapping[str, object]:
    """Read the sidecar.  No cross-check against the .nii.gz contents."""
    return json.loads(Path(json_path).read_text())
