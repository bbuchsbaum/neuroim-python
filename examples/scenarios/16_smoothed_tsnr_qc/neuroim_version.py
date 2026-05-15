"""Rewrite: smoothed temporal-SNR QC map through neuroim containers."""

from __future__ import annotations

from typing import Any

import numpy as np

import neuroim as ni


def smoothed_tsnr_qc(
    bold: ni.DenseNeuroVec,
    mask: ni.LogicalNeuroVol,
    *,
    fwhm_mm: float,
    smoothing_mask: ni.LogicalNeuroVol | None = None,
) -> tuple[ni.DenseNeuroVol, dict[str, Any]]:
    """Compute masked tSNR, smooth it in millimetres, and summarize it."""
    tsnr = bold.temporal_snr(mask=mask)
    smoothed = ni.gaussian_blur(
        tsnr,
        mask=mask if smoothing_mask is None else smoothing_mask,
        fwhm_mm=fwhm_mm,
    )
    values = np.asarray(smoothed.data)[np.asarray(mask.data, dtype=bool)]
    summary = {
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "n_voxels": int(np.count_nonzero(mask.data)),
        "fwhm_mm": float(fwhm_mm),
        "method_name": smoothed.provenance.method_name,
        "provenance_chain": smoothed.provenance.method_name,
    }
    return smoothed, summary

