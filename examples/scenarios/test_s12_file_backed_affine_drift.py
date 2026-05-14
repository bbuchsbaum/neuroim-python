"""Acceptance tests for Scenario 12 — file-backed affine drift."""

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


_SCENARIO_DIR = Path(__file__).resolve().parent / "12_file_backed_affine_drift"
baseline_nibabel = _load_module(
    "scenario12_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario12_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture
def split_run(tmp_path):
    rng = np.random.default_rng(1212)
    shape = (5, 4, 3, 8)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    data = rng.normal(loc=10.0, scale=0.5, size=shape).astype(np.float32)
    data[2, 2, 1, :] += np.linspace(0.0, 2.0, shape[3], dtype=np.float32)

    paths = []
    drift_paths = []
    for t in range(shape[3]):
        path = tmp_path / f"vol_{t:03d}.nii.gz"
        nib.save(nib.Nifti1Image(data[..., t], affine), str(path))
        paths.append(path)

        drift_affine = affine.copy()
        if t == 4:
            drift_affine[:3, 3] = [6.0, 0.0, 0.0]
        drift_path = tmp_path / f"drift_vol_{t:03d}.nii.gz"
        nib.save(nib.Nifti1Image(data[..., t], drift_affine), str(drift_path))
        drift_paths.append(drift_path)

    mask = np.zeros(shape[:3], dtype=np.uint8)
    mask[1:4, 1:4, 1:3] = 1
    mask_img = nib.Nifti1Image(mask, affine)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(mask_img, str(mask_path))
    mask_vol = ni.read_image(str(mask_path)).as_logical()
    return paths, drift_paths, mask_img, mask_vol


def test_aligned_file_backed_tsnr_matches_validated_nibabel(split_run):
    paths, _, mask_img, mask_vol = split_run
    baseline = baseline_nibabel.temporal_snr_from_split_run(paths, mask_img)
    rewrite = neuroim_version.temporal_snr_from_split_run(paths, mask_vol)

    assert isinstance(rewrite, ni.DenseNeuroVol)
    np.testing.assert_allclose(rewrite.data, baseline.get_fdata(), atol=1e-6)
    assert rewrite.space.compatible_with(mask_vol.space)
    assert rewrite.provenance.method_name == "temporal_snr"
    assert rewrite.provenance.mask_hash != "none"


def test_baseline_rejects_split_run_affine_drift(split_run):
    _, drift_paths, mask_img, _ = split_run

    with pytest.raises(ValueError, match="affine"):
        baseline_nibabel.temporal_snr_from_split_run(drift_paths, mask_img)


def test_neuroim_rejects_split_run_affine_drift(split_run):
    _, drift_paths, _, mask_vol = split_run

    with pytest.raises(ValueError, match="affine|space"):
        neuroim_version.temporal_snr_from_split_run(drift_paths, mask_vol)
