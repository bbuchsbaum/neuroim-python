"""Baseline: mean BOLD time series in a world-mm sphere using raw nibabel + numpy.

The form a competent nibabel user writes today.  No neuroim.  The
function shape is fixed by the friction the mission claims to remove:

- the user must invert the 4x4 affine themselves to convert the MNI
  centre into voxel space;
- the user must remember that "8 mm" is a *physical* radius and so
  must weight voxel-grid distances by per-axis spacing (this is the
  subtle bug a lot of hand-rolled sphere code gets wrong);
- the user must bounds-check the centre by hand; an OOB centre wraps
  silently via numpy without it.

The parity test in :mod:`test_s03_seed_sphere_mean` asserts numeric
equivalence with :mod:`neuroim_version` on the happy path and matching
``ValueError`` behaviour on the OOB path.
"""

from __future__ import annotations

from typing import Sequence

import nibabel as nib
import numpy as np


def mean_series_in_sphere_at_mni(
    bold_img: nib.Nifti1Image,
    mni_xyz: Sequence[float],
    radius_mm: float,
) -> np.ndarray:
    """Return the time-mean of voxels inside an ``radius_mm`` sphere at ``mni_xyz``.

    Parameters
    ----------
    bold_img
        4-D NIfTI image ``(nx, ny, nz, nt)``.
    mni_xyz
        World coordinate in mm, as a length-3 sequence.
    radius_mm
        Sphere radius in mm.

    Returns
    -------
    np.ndarray
        1-D array of length ``nt`` — the mean across in-sphere voxels
        for each time point.

    Raises
    ------
    ValueError
        If ``bold_img`` is not 4-D, ``radius_mm`` is negative, or
        ``mni_xyz`` maps to a voxel outside the image grid.
    """
    data = bold_img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got {data.ndim}D")

    mni_xyz = np.asarray(mni_xyz, dtype=float)
    if mni_xyz.shape != (3,):
        raise ValueError(f"mni_xyz must have shape (3,); got {mni_xyz.shape}")
    if radius_mm < 0:
        raise ValueError(f"radius_mm must be non-negative; got {radius_mm}")

    inv = np.linalg.inv(bold_img.affine)
    vox_homog = inv @ np.array([mni_xyz[0], mni_xyz[1], mni_xyz[2], 1.0])
    cx, cy, cz = vox_homog[:3]

    nx, ny, nz = data.shape[:3]
    ci, cj, ck = int(round(cx)), int(round(cy)), int(round(cz))
    if not (0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz):
        raise ValueError(
            f"world coord {tuple(float(x) for x in mni_xyz)} mm maps to voxel "
            f"({ci}, {cj}, {ck}) which is outside the image grid "
            f"of shape ({nx}, {ny}, {nz})."
        )

    spacing = np.linalg.norm(bold_img.affine[:3, :3], axis=0)

    grid = np.mgrid[:nx, :ny, :nz].astype(float)
    dist = np.sqrt(
        ((grid[0] - cx) * spacing[0]) ** 2
        + ((grid[1] - cy) * spacing[1]) ** 2
        + ((grid[2] - cz) * spacing[2]) ** 2
    )
    mask = dist <= radius_mm
    if not mask.any():
        mask[ci, cj, ck] = True

    return data[mask].mean(axis=0)
