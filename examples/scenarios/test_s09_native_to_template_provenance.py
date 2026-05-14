"""Acceptance tests for Scenario 09 — native-to-template provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni

from fixtures.realistic_bold import make_realistic_bold, to_nibabel


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = Path(__file__).resolve().parent / "09_native_to_template_provenance"
baseline_nibabel = _load_module(
    "scenario09_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario09_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


def _spatial_affine(space) -> np.ndarray:
    trans = np.asarray(space.trans, dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = trans[:3, :3]
    affine[:3, 3] = trans[:3, -1]
    return affine


@pytest.fixture(scope="module")
def template_fixture():
    return make_realistic_bold(seed=11)


@pytest.fixture(scope="module")
def native_bold(template_fixture):
    """Same data shape as template, but shifted native-space affine."""
    data = np.asarray(template_fixture.bold.data).copy()
    native_affine = _spatial_affine(template_fixture.bold.space)
    native_affine[:3, 3] += [6.0, -3.0, 0.0]
    native_space = ni.NeuroSpace.from_affine(native_affine, template_fixture.bold.shape)
    return ni.DenseNeuroVec(data, native_space)


def test_baseline_manifest_records_source_and_target_spaces(template_fixture, native_bold):
    template_bold_img, template_mask_img = to_nibabel(template_fixture)
    native_img = nib.Nifti1Image(np.asarray(native_bold.data), _spatial_affine(native_bold.space))
    out, manifest = baseline_nibabel.native_to_template_tsnr(
        native_img, template_bold_img, template_mask_img
    )
    assert out.shape == template_fixture.mask.shape
    assert manifest["method_name"] == "resample_to_template+temporal_snr"
    assert manifest["source_space_hash"] != manifest["target_space_hash"]
    assert manifest["mask_hash"]
    assert manifest["resample_order"] == 1


def test_neuroim_resampled_map_is_typed_and_template_compatible(template_fixture, native_bold):
    out = neuroim_version.native_to_template_tsnr(
        native_bold, template_fixture.bold, template_fixture.mask
    )
    assert isinstance(out, ni.DenseNeuroVol)
    assert out.space.compatible_with(template_fixture.mask.space)
    assert "temporal_snr" in out.provenance.method_name
    assert out.provenance.mask_hash


def test_neuroim_receipt_records_resample_then_temporal_snr(template_fixture, native_bold):
    out = neuroim_version.native_to_template_tsnr(
        native_bold, template_fixture.bold, template_fixture.mask
    )
    assert "resample" in out.provenance.method_name
    assert "order=1" in out.provenance.method_name
    assert "source=" in out.provenance.method_name
    assert "target=" in out.provenance.method_name
    assert "temporal_snr" in out.provenance.method_name
