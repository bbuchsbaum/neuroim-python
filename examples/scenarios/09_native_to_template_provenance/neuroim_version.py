"""Rewrite: native-to-template resample followed by typed temporal SNR."""

from __future__ import annotations

import neuroim as ni
from neuroim.resample import resample_vec


def native_to_template_tsnr(
    native_bold: ni.NeuroVec,
    template_bold: ni.NeuroVec,
    template_mask: ni.LogicalNeuroVol,
) -> ni.DenseNeuroVol:
    """Resample native-space BOLD to template, then compute template-space tSNR."""
    template_bold_resampled = resample_vec(native_bold, template_bold.space, interpolation=1)
    return template_bold_resampled.temporal_snr(mask=template_mask)

