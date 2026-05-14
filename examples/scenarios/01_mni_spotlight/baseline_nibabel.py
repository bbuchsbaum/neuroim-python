"""Baseline: extract a BOLD time series at a world-mm coordinate using raw nibabel + numpy.

This is the form a competent nibabel user writes today.  The function
shape is fixed by the friction the mission claims to remove:

- the user must invert the 4x4 affine themselves;
- the user must construct a homogeneous coordinate ``[x, y, z, 1]``;
- the user must round to int and bounds-check by hand;
- the result is a bare ndarray with no spatial provenance.

The parity test in :mod:`test_scenario` checks that this function
returns the same numbers as :mod:`neuroim_version` on the happy path,
and surfaces the bug-class divergence on the out-of-bounds path.
"""

from __future__ import annotations

from typing import Sequence

import nibabel as nib
import numpy as np


def series_at_mni(bold_img: nib.Nifti1Image, mni_xyz: Sequence[float]) -> np.ndarray:
    """Return the BOLD time series at the voxel nearest the world-mm coord.

    Parameters
    ----------
    bold_img
        4-D NIfTI image ``(nx, ny, nz, nt)``.
    mni_xyz
        World coordinate in mm, as a length-3 sequence.

    Returns
    -------
    np.ndarray
        1-D array of length ``nt`` — the time series at the nearest voxel.

    Raises
    ------
    ValueError
        If ``bold_img`` is not 4-D, or if ``mni_xyz`` maps to a voxel
        outside the image grid.
    """
    data = bold_img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got {data.ndim}D")

    mni_xyz = np.asarray(mni_xyz, dtype=float)
    if mni_xyz.shape != (3,):
        raise ValueError(f"mni_xyz must have shape (3,); got {mni_xyz.shape}")

    inv = np.linalg.inv(bold_img.affine)
    vox_homog = inv @ np.array([mni_xyz[0], mni_xyz[1], mni_xyz[2], 1.0])
    i, j, k = np.round(vox_homog[:3]).astype(int)

    nx, ny, nz = data.shape[:3]
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        raise ValueError(
            f"world coord {tuple(float(x) for x in mni_xyz)} mm maps to voxel "
            f"({int(i)}, {int(j)}, {int(k)}) which is outside the image grid "
            f"of shape ({nx}, {ny}, {nz}). "
            f"NB: numpy's negative-index wrap would silently return wrong "
            f"data here — this check exists to prevent that."
        )

    return data[i, j, k, :]
