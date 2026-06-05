"""Parcellated connectome -- nibabel + numpy baseline.

Given a 4-D BOLD image and an integer-labelled 3-D atlas (0 = background,
1..N = parcels), return the sorted parcel ids and the ``(N, N)`` Pearson
correlation matrix of the per-parcel mean time series.

What this baseline does (and what it deliberately doesn't):

  Shape check
      The data arrays must have matching spatial shape, or we raise.  This
      is the standard nibabel-user level of paranoia.

  Affine check
      None.  Two volumes with identical ``(nx, ny, nz)`` shape and totally
      different affines look interchangeable here -- the same failure mode
      S02/S11 demonstrated for masks and atlases, surfacing again at the
      connectome boundary.  ``test_baseline_silently_accepts_mismatched_affine``
      in the scenario test file proves the silent miscorrelation: an
      LR-flipped atlas yields a same-shape, plausible-but-wrong matrix.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np


def parcel_timeseries(
    bold_img: nib.Nifti1Image, atlas_img: nib.Nifti1Image
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, ts)`` for ``ts`` of shape ``(n_labels, n_time)``."""
    bold = np.asarray(bold_img.dataobj, dtype=np.float64)
    atlas = np.asarray(atlas_img.dataobj, dtype=np.int32)
    if atlas.shape != bold.shape[:3]:
        raise ValueError(
            f"atlas shape {atlas.shape} != bold spatial shape {bold.shape[:3]}"
        )
    labels = np.unique(atlas)
    labels = labels[labels != 0]
    ts = np.empty((labels.size, bold.shape[-1]), dtype=np.float64)
    for row, label in enumerate(labels):
        ts[row] = bold[atlas == label].mean(axis=0)
    return labels, ts


def parcel_connectome(
    bold_img: nib.Nifti1Image, atlas_img: nib.Nifti1Image
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, connectome)`` with ``connectome`` of shape ``(N, N)``.

    The connectome is the Pearson correlation matrix of the per-parcel mean
    time series.  Only spatial shape is checked; a wrong-affine atlas is
    accepted silently.
    """
    labels, ts = parcel_timeseries(bold_img, atlas_img)
    # np.corrcoef with rowvar=True treats each row (parcel) as a variable.
    connectome = np.corrcoef(ts)
    return labels, connectome
