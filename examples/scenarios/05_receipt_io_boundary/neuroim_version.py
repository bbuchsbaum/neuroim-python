"""Rewrite: write a derived map + read back with provenance via neuroim.

The aspiration is that the typed result's Receipt travels with the
NIfTI file — embedded in a header extension or a paired sidecar — so a
collaborator who only has the .nii.gz path can re-hydrate
``Receipt(method_name, n_voxels, input_space_hash, mask_hash, ...)``
without trusting the upstream Python session.

Today this aspiration is **not met** — see PAIN-6 in :file:`REPORT.md`
and tracker bd-01KRKR7SX4GKW1QZ9KF6G73ZWR.

The functions below implement the expected interface anyway so the
acceptance test can either:

- pass once neuroim's typed-write path embeds and re-hydrates the
  Receipt; or
- explicitly mark itself ``xfail(strict=True)`` until that lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import neuroim as ni
from neuroim.results import Receipt, SearchlightResult


def write_searchlight_mean(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
    out_path: Path,
    *,
    radius: float = 4.5,
) -> SearchlightResult:
    """Run a mean-searchlight and write the resulting derived map.

    Returns the in-memory :class:`SearchlightResult` so the test can
    compare its Receipt against what survives the round-trip.
    """
    sl = ni.searchlight_apply(
        mask,
        radius=radius,
        method=lambda a: float(np.asarray(a).mean()),
        data=bold,
        cores=0,
    )
    img = sl.to_nibabel()
    import nibabel as nib
    nib.save(img, str(out_path))
    return sl


def read_provenance_from_file(out_path: Path) -> Optional[Receipt]:
    """Re-hydrate the Receipt from a written derived map, if possible.

    Returns ``None`` when no provenance is recoverable — which is the
    current state on main (PAIN-6).  The acceptance test treats
    ``None`` as the falsifying observation.
    """
    img = ni.read_image(str(out_path))
    return getattr(img, "provenance", None)
