"""Acceptance tests for Scenario 13 — pipeline provenance audit."""

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


_SCENARIO_DIR = Path(__file__).resolve().parent / "13_provenance_audit"
baseline_nibabel = _load_module(
    "scenario13_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario13_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


def _spatial_affine(space) -> np.ndarray:
    return np.asarray(space.affine, dtype=float)


@pytest.fixture(scope="module")
def template_fixture():
    return make_realistic_bold(seed=13)


@pytest.fixture(scope="module")
def native_bold(template_fixture):
    data = np.asarray(template_fixture.bold.data).copy()
    native_affine = _spatial_affine(template_fixture.bold.space)
    native_space = ni.NeuroSpace.from_affine(native_affine, template_fixture.bold.shape)
    return ni.DenseNeuroVec(data, native_space)


def test_pipeline_outputs_match_numeric_map(tmp_path, template_fixture, native_bold):
    template_bold_img, template_mask_img = to_nibabel(template_fixture)
    native_img = nib.Nifti1Image(np.asarray(native_bold.data), _spatial_affine(native_bold.space))

    baseline_path = tmp_path / "baseline_tsnr.nii.gz"
    neuroim_path = tmp_path / "neuroim_tsnr.nii.gz"
    baseline = baseline_nibabel.write_tsnr_template(
        native_img,
        template_mask_img,
        template_bold_img,
        baseline_path,
        interpolation=1,
    )
    rewrite = neuroim_version.write_tsnr_template(
        native_bold,
        template_fixture.mask,
        neuroim_path,
        interpolation=1,
    )

    assert isinstance(rewrite, ni.DenseNeuroVol)
    assert rewrite.space.compatible_with(template_fixture.mask.space)
    np.testing.assert_allclose(rewrite.data, baseline.get_fdata(), rtol=1e-5, atol=1e-5)


def test_raw_nibabel_file_cannot_answer_audit_questions(tmp_path, template_fixture, native_bold):
    template_bold_img, template_mask_img = to_nibabel(template_fixture)
    native_img = nib.Nifti1Image(np.asarray(native_bold.data), _spatial_affine(native_bold.space))
    out = tmp_path / "baseline_tsnr.nii.gz"
    baseline_nibabel.write_tsnr_template(
        native_img,
        template_mask_img,
        template_bold_img,
        out,
        interpolation=1,
    )

    audit = baseline_nibabel.audit(out)
    assert audit["n_header_extensions"] == 0
    assert audit["method_name"] is None
    assert audit["input_space_hash"] is None
    assert audit["mask_hash"] is None
    assert audit["pipeline_parameters"] is None
    assert audit["library_version"] is None


def test_neuroim_file_recovers_chained_provenance_from_disk(
    tmp_path, template_fixture, native_bold
):
    out = tmp_path / "neuroim_tsnr.nii.gz"
    produced = neuroim_version.write_tsnr_template(
        native_bold,
        template_fixture.mask,
        out,
        interpolation=1,
    )

    audit = neuroim_version.audit(out)
    assert audit["n_header_extensions"] >= 1
    assert "resample_vec" in audit["method_name"]
    assert "order=1" in audit["method_name"]
    assert "source=" in audit["method_name"]
    assert "target=" in audit["method_name"]
    assert "temporal_snr" in audit["method_name"]
    assert audit["input_space_hash"] == produced.provenance.input_space_hash
    assert audit["mask_hash"] == produced.provenance.mask_hash
    assert audit["library_version"] == produced.provenance.neuroim_version
    assert audit["producing_library"] == "neuroim"
