"""Acceptance tests for Scenario 06: public seed-to-voxel correlation."""

from __future__ import annotations

import importlib

import numpy as np
import nibabel as nib
import pytest

baseline_nibabel = importlib.import_module(
    "examples.scenarios.06_public_seed_correlation.baseline_nibabel"
)
neuroim_version = importlib.import_module(
    "examples.scenarios.06_public_seed_correlation.neuroim_version"
)


def _fixture_images():
    rng = np.random.default_rng(606)
    shape = (5, 5, 4, 30)
    affine = np.eye(4)
    seed_ijk = (2, 2, 1)
    seed_xyz = np.array(seed_ijk, dtype=float)

    seed_signal = np.sin(np.linspace(0, 4 * np.pi, shape[3]))
    data = rng.normal(scale=0.05, size=shape)
    data[seed_ijk] = seed_signal
    data[1, 1, 1, :] = seed_signal * 0.75 + rng.normal(scale=0.01, size=shape[3])
    data[3, 3, 2, :] = -seed_signal * 0.5 + rng.normal(scale=0.01, size=shape[3])

    mask = np.zeros(shape[:3], dtype=np.uint8)
    mask[1:4, 1:4, 1:3] = 1
    mask[seed_ijk] = 1

    bold_img = nib.Nifti1Image(data.astype(np.float32), affine)
    mask_img = nib.Nifti1Image(mask, affine)
    return bold_img, mask_img, seed_xyz


def test_public_seed_correlation_numeric_parity():
    bold_img, mask_img, seed_xyz = _fixture_images()

    baseline = baseline_nibabel.seed_correlation_map(
        bold_img, mask_img, seed_xyz=seed_xyz
    )
    rewrite = neuroim_version.seed_correlation_map(
        bold_img, mask_img, seed_xyz=seed_xyz
    )

    np.testing.assert_allclose(rewrite.data, baseline.get_fdata(), atol=1e-10)
    np.testing.assert_allclose(rewrite.space.affine, baseline.affine)
    assert rewrite.provenance.method_name == "seed_correlation_map"
    assert rewrite.provenance.mask_hash != "none"


def test_public_seed_correlation_rejects_mismatched_mask_affine():
    bold_img, mask_img, seed_xyz = _fixture_images()
    shifted_affine = mask_img.affine.copy()
    shifted_affine[0, 3] = 10.0
    shifted_mask = nib.Nifti1Image(np.asanyarray(mask_img.dataobj), shifted_affine)

    with pytest.raises(ValueError, match="affine"):
        baseline_nibabel.seed_correlation_map(
            bold_img, shifted_mask, seed_xyz=seed_xyz
        )
    with pytest.raises(ValueError, match="affine"):
        neuroim_version.seed_correlation_map(
            bold_img, shifted_mask, seed_xyz=seed_xyz
        )


def test_public_seed_correlation_rejects_oob_seed():
    bold_img, mask_img, _ = _fixture_images()

    with pytest.raises(ValueError, match="outside|out of bounds"):
        neuroim_version.seed_correlation_map(
            bold_img, mask_img, seed_xyz=np.array([999.0, 999.0, 999.0])
        )
