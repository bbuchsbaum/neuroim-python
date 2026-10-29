"""Gaussian spatial smoothing at FWHM = 6 mm -- neuroim rewrite.

The first-class API now owns FWHM-in-mm, the mask same-space gate, and
provenance threading.  See ``examples/scenarios/12_gaussian_smoothing/REPORT.md``
for what the prior hand-rolled body looked like and which PAINs the
single-line form closed.
"""

from __future__ import annotations

from neuroim import DenseNeuroVol, LogicalNeuroVol, gaussian_blur


def smooth_fwhm_mm(
    vol: DenseNeuroVol,
    fwhm_mm: float,
    mask: LogicalNeuroVol | None = None,
) -> DenseNeuroVol:
    """Isotropic-mm-space FWHM smoothing through the curated public API."""
    return gaussian_blur(vol, mask=mask, fwhm_mm=fwhm_mm)


def smooth_via_sigma_mm(vol: DenseNeuroVol, sigma_mm: float) -> DenseNeuroVol:
    """Smooth with a scalar sigma, which (like neuroim2) is in millimetres.

    Retained so the anisotropy test can show, on a 3 x 3 x 3.5 mm fixture,
    that ``sigma`` is converted per axis with the voxel spacing. Releases
    before 0.3.0 read ``sigma`` in voxels and smoothed anisotropically in mm.
    """
    return gaussian_blur(vol, sigma=sigma_mm)
