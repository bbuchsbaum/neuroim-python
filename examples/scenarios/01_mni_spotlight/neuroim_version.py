"""Rewrite: extract a BOLD time series at a world-mm coordinate using the neuroim API.

Two surfaces are exposed:

- :func:`series_at_mni` — the simplest correct neuroim form.  Returns a
  bare ndarray to match the baseline's signature for the parity test.
- :func:`series_at_mni_typed` — the canonical mission form.  Returns an
  :class:`~neuroim.results.ROIExtractionResult` carrying values, coords,
  space, and a :class:`~neuroim.results.Receipt`.

Both forms name the operation the baseline hand-codes:
``vec.series_at_world(mni)`` replaces affine inversion, rounding,
bounds-checking, and raw array indexing.  The typed form additionally
carries provenance forward.
"""

from __future__ import annotations

from typing import Sequence

import neuroim as ni
from neuroim.results import ROIExtractionResult


def series_at_mni(bold: ni.NeuroVec, mni_xyz: Sequence[float]) -> np.ndarray:
    """Return the BOLD time series at the voxel nearest ``mni_xyz`` (mm).

    Mirrors the baseline's return type (bare ndarray) so the two
    implementations can be checked for numeric parity.

    Parameters
    ----------
    bold
        4-D :class:`~neuroim.NeuroVec`.
    mni_xyz
        World coordinate in mm.
    """
    if bold.space.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got {bold.space.ndim}D")
    return bold.series_at_world(mni_xyz)


def series_at_mni_typed(
    bold: ni.NeuroVec, mni_xyz: Sequence[float]
) -> ROIExtractionResult:
    """Return the time series at ``mni_xyz`` as a typed, provenance-bearing result.

    This is the canonical neuroim form: the spatial frame and a Receipt
    travel with the values, so downstream code can validate space
    compatibility instead of trusting a bare ndarray.
    """
    if bold.space.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got {bold.space.ndim}D")
    return bold.series_roi_world(mni_xyz)
