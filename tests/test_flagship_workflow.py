"""Flagship workflow contract for the Python-native neuroim surface.

This is the public proof path for the first Python-native adapter slice.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

import neuroim as ni


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "golden_tests" / "fixtures"
BOLD_PATH = FIXTURE_DIR / "tiny_bold.nii.gz"
MASK_PATH = FIXTURE_DIR / "tiny_mask.nii.gz"


def test_flagship_workflow_uses_neuroim_as_public_surface():
    """A user enters through neuroim, gets typed spatial analysis, and exits to NIfTI."""
    bold = ni.read_image(BOLD_PATH)
    mask = ni.read_image(MASK_PATH)

    source_img = nib.load(BOLD_PATH)
    assert np.allclose(bold.space.affine, source_img.affine)
    assert bold.shape == source_img.shape
    assert bold.space.compatible_with(mask.space)

    roi = ni.spherical_roi(mask, centroid=(4, 4, 2), radius=2)
    time_by_voxel = bold.series_roi(roi)
    assert time_by_voxel.shape == (source_img.shape[-1], len(roi.coords))

    mean_data = np.zeros(mask.shape, dtype=np.float32)
    mean_data[tuple(roi.coords.T)] = time_by_voxel.mean(axis=0)
    mean_map = ni.NeuroVol.from_array(mean_data, space=mask.space)

    out = mean_map.to_nibabel()
    assert isinstance(out, nib.Nifti1Image)
    assert out.shape == mask.shape
    assert np.allclose(out.affine, source_img.affine)


def test_flagship_workflow_keeps_explicit_nibabel_interop():
    """Existing nibabel users can wrap and unwrap SpatialImage objects explicitly."""
    source_img = nib.load(BOLD_PATH)

    bold = ni.NeuroVec.from_nibabel(source_img)
    assert np.allclose(bold.space.affine, source_img.affine)
    assert bold.shape == source_img.shape

    round_tripped = bold.to_nibabel()
    assert isinstance(round_tripped, nib.Nifti1Image)
    assert round_tripped.shape == source_img.shape
    assert np.allclose(round_tripped.affine, source_img.affine)
