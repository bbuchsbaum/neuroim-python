"""Baseline: drop pre-steady-state volumes, compute masked temporal SNR.

The careful raw-`nibabel` user explicitly records the truncation in a
sidecar manifest so a downstream collaborator who only has the written
NIfTI can answer "was this map computed on the full series or a
subset?" — the same question Scenario 14 asks of the neuroim path.
"""

from __future__ import annotations

import hashlib
from typing import Any, Tuple

import nibabel as nib
import numpy as np


def _space_hash(img: nib.Nifti1Image) -> str:
    return hashlib.sha256(
        np.asarray(img.affine, dtype=np.float64).tobytes()
        + np.asarray(img.shape[:3], dtype=np.int64).tobytes()
    ).hexdigest()


def _mask_hash(mask_img: nib.Nifti1Image) -> str:
    data = np.asarray(mask_img.get_fdata(), dtype=bool)
    return hashlib.sha256(data.tobytes()).hexdigest()


def temporal_snr_after_slice(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    start: int,
) -> Tuple[nib.Nifti1Image, dict[str, Any]]:
    """Drop the first ``start`` timepoints, then compute masked tSNR.

    Returns the tSNR map plus a manifest recording the truncation. The
    manifest is what a careful user hand-writes today because raw
    nibabel does not carry it forward through ``nib.save``.
    """
    if bold_img.shape[:3] != mask_img.shape:
        raise ValueError("bold and mask spatial shape mismatch")
    if not np.allclose(bold_img.affine, mask_img.affine):
        raise ValueError("bold and mask spatial affine mismatch")

    data = np.asarray(bold_img.get_fdata())
    sliced = data[..., start:]
    mask = np.asarray(mask_img.get_fdata(), dtype=bool)

    mean = sliced.mean(axis=-1)
    std = sliced.std(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(std > 0, mean / std, 0.0)
    snr = snr * mask

    out = nib.Nifti1Image(snr.astype(np.float32), bold_img.affine)
    manifest = {
        "method_name": "temporal_slice+temporal_snr",
        "temporal_slice_start": int(start),
        "temporal_slice_stop": None,
        "temporal_slice_step": None,
        "n_timepoints_in": int(data.shape[-1]),
        "n_timepoints_used": int(sliced.shape[-1]),
        "input_space_hash": _space_hash(bold_img),
        "mask_hash": _mask_hash(mask_img),
    }
    return out, manifest
