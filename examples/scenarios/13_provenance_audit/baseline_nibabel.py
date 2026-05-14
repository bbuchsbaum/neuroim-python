"""Baseline: chained derived-map pipeline in raw nibabel + numpy.

Runs the *same numeric pipeline* as :mod:`neuroim_version`:

  1. resample a native 4-D BOLD into a template space using
     ``nibabel.processing.resample_from_to`` (the canonical raw-nibabel
     resampler — what neuroim itself delegates to internally);
  2. compute a masked 3-D temporal SNR map on the resampled BOLD;
  3. save the result to disk as a 3-D NIfTI.

Then at audit time, opens the file produced and tries to recover:

  - the method that produced it,
  - the input spatial frame's identity,
  - the mask used,
  - any pipeline parameters,
  - the producing library/version.

In raw nibabel, *none of those answers ride in the file*.  The producer
would have had to write a sidecar JSON; this baseline does not.  The
audit returns an empty manifest, which is the failure mode the
mission claim is staked against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def write_tsnr_template(
    native_bold_img: nib.Nifti1Image,
    template_mask_img: nib.Nifti1Image,
    template_4d_img: nib.Nifti1Image,
    out_path: Path,
    *,
    interpolation: int = 1,
) -> nib.Nifti1Image:
    """Resample BOLD to template space, compute tSNR, write to ``out_path``.

    Returns the in-memory result as a :class:`nibabel.Nifti1Image` so the
    parity test can compare numeric arrays against the neuroim rewrite.
    """
    native_data = native_bold_img.get_fdata()
    if native_data.ndim != 4:
        raise ValueError(f"expected 4-D native BOLD, got {native_data.ndim}-D")

    template_shape = tuple(int(s) for s in template_mask_img.shape[:3])
    n_time = native_data.shape[3]

    # Resample each volume from native space to template space via the
    # canonical nibabel resampler.  This matches what neuroim's internal
    # resample delegates to, so numeric parity is exact.
    target_3d_template = nib.Nifti1Image(
        np.zeros(template_shape, dtype=np.float32),
        template_4d_img.affine[:4, :4],
    )
    resampled = np.empty(template_shape + (n_time,), dtype=np.float64)
    for t in range(n_time):
        vol_img = nib.Nifti1Image(native_data[..., t], native_bold_img.affine)
        resampled_vol = resample_from_to(vol_img, target_3d_template, order=interpolation)
        resampled[..., t] = resampled_vol.get_fdata()

    mask_data = np.asarray(template_mask_img.get_fdata(), dtype=bool)
    if mask_data.shape != template_shape:
        raise ValueError(
            f"template mask shape {mask_data.shape} != template_shape {template_shape}"
        )

    mean = resampled.mean(axis=3)
    std = resampled.std(axis=3)
    tsnr = np.zeros(template_shape, dtype=np.float64)
    valid = mask_data & (std > 0)
    tsnr[valid] = mean[valid] / std[valid]

    img = nib.Nifti1Image(tsnr.astype(np.float32), target_3d_template.affine)
    nib.save(img, str(out_path))
    return img


def audit(file_path: Path) -> Dict[str, Any]:
    """Open the file and recover what produced it.

    Raw nibabel cannot answer any of the audit questions from the file
    alone.  ``img.header.extensions`` is empty; the affine and shape say
    what spatial frame the map lives in, but not what *input* frame it
    came from, what mask was used, or what method produced it.

    This function returns the best-effort manifest a careful nibabel
    user could assemble.  The empty fields are the falsifiable part:
    every empty value is a question a neuroim user can answer from the
    same on-disk bytes.
    """
    img = nib.load(str(file_path))
    extensions = list(img.header.extensions)
    return {
        "format": "nifti1",
        "n_header_extensions": len(extensions),
        "output_shape": tuple(int(s) for s in img.shape),
        "output_affine_present": True,
        # Everything below is unknown from the file alone.
        "method_name": None,
        "input_space_hash": None,
        "mask_hash": None,
        "pipeline_parameters": None,
        "producing_library": None,
        "library_version": None,
    }
