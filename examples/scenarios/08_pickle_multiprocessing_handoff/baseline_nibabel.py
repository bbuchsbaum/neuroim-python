"""Baseline: pickle a derived map through a process boundary with nibabel.

Raw nibabel can pickle an image's data and affine.  It does not know what
operation produced the image, what mask was used, or which input space was
checked.  A careful user has to create and keep a sidecar manifest in sync.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import subprocess
import sys
from typing import Any

import nibabel as nib
import numpy as np


def temporal_snr_image(bold_img: nib.Nifti1Image, mask_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Compute a masked temporal-SNR map as a bare nibabel image."""
    if bold_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError("bold and mask spatial shapes differ")
    if not np.allclose(bold_img.affine, mask_img.affine):
        raise ValueError("bold and mask affines differ")

    data = np.asarray(bold_img.get_fdata(dtype=np.float64))
    mask = np.asarray(mask_img.get_fdata()).astype(bool)
    mean = data.mean(axis=3)
    sd = data.std(axis=3)
    out = np.zeros(data.shape[:3], dtype=np.float64)
    valid = mask & (sd > 0)
    out[valid] = mean[valid] / sd[valid]
    return nib.Nifti1Image(out, bold_img.affine)


def manual_manifest(bold_img: nib.Nifti1Image, mask_img: nib.Nifti1Image) -> dict[str, Any]:
    """Hand-written provenance a careful nibabel user must keep in sync."""
    mask = np.asarray(mask_img.get_fdata()).astype(bool)
    return {
        "method_name": "temporal_snr",
        "n_voxels": int(mask.sum()),
        "input_space_hash": hashlib.sha256(
            np.asarray(bold_img.affine, dtype=np.float64).tobytes()
            + np.asarray(bold_img.shape[:3], dtype=np.int64).tobytes()
        ).hexdigest(),
        "mask_hash": hashlib.sha256(mask.tobytes()).hexdigest(),
    }


def pickle_payload(obj: Any) -> bytes:
    """Serialize an object as a joblib/multiprocessing-style pickle payload."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def inspect_payload_in_fresh_process(payload: bytes) -> dict[str, Any]:
    """Unpickle a payload in a clean Python process and summarize it as JSON."""
    code = r"""
import base64, json, pickle, sys
obj = pickle.loads(base64.b64decode(sys.stdin.read().encode("ascii")))
if isinstance(obj, dict):
    image = obj.get("image")
    manifest = obj.get("manifest")
else:
    image = obj
    manifest = None
summary = {
    "payload_type": type(obj).__name__,
    "image_type": type(image).__name__,
    "shape": tuple(image.shape),
    "has_manifest": manifest is not None,
    "method_name": None if manifest is None else manifest.get("method_name"),
    "input_space_hash": None if manifest is None else manifest.get("input_space_hash"),
    "mask_hash": None if manifest is None else manifest.get("mask_hash"),
}
print(json.dumps(summary, sort_keys=True))
"""
    encoded = base64.b64encode(payload).decode("ascii")
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=encoded,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)

