"""Acceptance tests for Scenario 10 — searchlight space mismatch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = Path(__file__).resolve().parent / "10_searchlight_space_mismatch"
baseline_nibabel = _load_module(
    "scenario10_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario10_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture
def nib_images():
    rng = np.random.default_rng(1010)
    shape = (5, 5, 4, 12)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    data = rng.normal(size=shape).astype(np.float32)
    data[2, 2, 1, :] += np.linspace(0.0, 1.0, shape[3], dtype=np.float32)

    mask = np.zeros(shape[:3], dtype=np.uint8)
    mask[1:4, 1:4, 1:3] = 1

    bold_img = nib.Nifti1Image(data, affine)
    mask_img = nib.Nifti1Image(mask, affine)
    shifted_affine = affine.copy()
    shifted_affine[:3, 3] = [8.0, 0.0, 0.0]
    shifted_mask_img = nib.Nifti1Image(mask, shifted_affine)
    return bold_img, mask_img, shifted_mask_img


def _neuroim_pair(bold_img, mask_img):
    bold = ni.from_nibabel(bold_img)
    mask = ni.from_nibabel(mask_img).as_logical()
    return bold, mask


def test_aligned_searchlight_numeric_parity(nib_images):
    bold_img, mask_img, _ = nib_images
    baseline = baseline_nibabel.local_mean_searchlight(bold_img, mask_img)
    bold, mask = _neuroim_pair(bold_img, mask_img)
    rewrite = neuroim_version.local_mean_searchlight(bold, mask)
    rewrite_vol = rewrite.map_to_volume()

    np.testing.assert_allclose(
        rewrite_vol.data,
        baseline.get_fdata(),
        equal_nan=True,
        atol=1e-10,
    )
    assert isinstance(rewrite, ni.SearchlightResult)
    assert rewrite.provenance.radius == 2.1
    assert rewrite.provenance.mask_hash != "none"


def test_baseline_rejects_shifted_mask_affine(nib_images):
    bold_img, _, shifted_mask_img = nib_images

    with pytest.raises(ValueError, match="affine"):
        baseline_nibabel.local_mean_searchlight(bold_img, shifted_mask_img)


def test_neuroim_rejects_shifted_mask_affine(nib_images):
    """PAIN-10 closed: searchlight_apply now calls assert_same_space on
    the (data, mask) pair before iterating neighborhoods, so a mask in a
    different spatial frame is refused at the contract layer instead of
    silently scattering data bytes.
    """
    bold_img, _, shifted_mask_img = nib_images
    bold, shifted_mask = _neuroim_pair(bold_img, shifted_mask_img)

    with pytest.raises(ValueError, match="affine|space"):
        neuroim_version.local_mean_searchlight(bold, shifted_mask)
