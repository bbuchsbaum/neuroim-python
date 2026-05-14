"""Rewrite: chained derived-map pipeline through the neuroim public API.

Same numeric pipeline as :mod:`baseline_nibabel`:

  1. ``ni.resample_vec(native_bold, template, interpolation=1)`` — output
     carries a :class:`~neuroim.results.Receipt` whose ``method_name`` is
     ``"resample_vec"`` and whose ``ResampleParams`` records the source/
     target space hashes and interpolation order.
  2. ``resampled.temporal_snr(mask=template_mask)`` — the temporal
     reduction *merges* the upstream Receipt, producing a new Receipt
     whose ``method_name`` is ``"resample_vec+temporal_snr"`` (chained)
     and whose ``input_space_hash`` and ``mask_hash`` record the
     terminal frame and the mask used.
  3. ``tsnr.to_nibabel()`` embeds the chained Receipt into a NIfTI
     comment extension (ecode 6, prefix ``neuroim/receipt/v1:``) on
     write.

At audit time, ``ni.read_image(path).provenance`` re-hydrates the same
Receipt and the audit succeeds for the file alone — no producer, no
sidecar, no Slack thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import numpy as np

import neuroim as ni


def write_tsnr_template(
    native_bold: ni.NeuroVec,
    template_mask: ni.LogicalNeuroVol,
    out_path: Path,
    *,
    interpolation: int = 1,
) -> ni.DenseNeuroVol:
    """Resample BOLD to template, compute tSNR, write with chained Receipt.

    The output 3-D :class:`~neuroim.DenseNeuroVol` carries a chained
    ``Receipt`` in ``.provenance`` covering both pipeline stages.
    ``nib.save(tsnr.to_nibabel(), out_path)`` embeds it on disk.
    """
    template_space = template_mask.space.add_dim(size=native_bold.shape[-1])
    resampled = ni.resample_vec(
        native_bold,
        template_space,
        interpolation=interpolation,
    )
    tsnr = resampled.temporal_snr(mask=template_mask)
    nib.save(tsnr.to_nibabel(), str(out_path))
    return tsnr


def audit(file_path: Path) -> Dict[str, Any]:
    """Open the file and recover what produced it from on-disk bytes alone.

    The full audit answer rides in the NIfTI comment extension; this
    function reads it back through :func:`neuroim.read_image` and
    surfaces the five questions S13 stakes the mission claim on.
    """
    vol = ni.read_image(str(file_path))
    prov = vol.provenance
    if prov is None:
        return {
            "format": "nifti1",
            "method_name": None,
            "input_space_hash": None,
            "mask_hash": None,
            "pipeline_parameters": None,
            "producing_library": None,
            "library_version": None,
        }

    # Also count the NIfTI header extensions on the raw nibabel side so
    # the audit can confirm "the Receipt rode as a NIfTI extension, not
    # as a magic neuroim-side cache."
    raw_img = nib.load(str(file_path))
    n_extensions = len(list(raw_img.header.extensions))

    return {
        "format": "nifti1",
        "n_header_extensions": n_extensions,
        "method_name": prov.method_name,
        "input_space_hash": prov.input_space_hash,
        "mask_hash": prov.mask_hash,
        "pipeline_parameters": {
            "radius": prov.radius,
            "seed": prov.seed,
            "source_affine_hash": prov.source_affine_hash,
        },
        "producing_library": "neuroim",
        "library_version": prov.neuroim_version,
        "output_shape": tuple(int(s) for s in vol.shape),
    }
