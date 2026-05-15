"""Acceptance tests for Scenario 16 - smoothed temporal-SNR QC map."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import neuroim as ni

from neuroim.results import Receipt

from fixtures.realistic_bold import make_realistic_bold, to_nibabel


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "scenarios"
    / "16_smoothed_tsnr_qc"
)
baseline = _load_module("scenario16_baseline", _SCENARIO_DIR / "baseline_nibabel.py")
rewrite = _load_module("scenario16_rewrite", _SCENARIO_DIR / "neuroim_version.py")


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold(seed=16)


@pytest.fixture(scope="module")
def nib_bold_and_mask(fixture):
    return to_nibabel(fixture)


def _mask_to_nifti(mask_vol) -> nib.Nifti1Image:
    affine = np.asarray(mask_vol.space.trans, dtype=float)[:4, :4]
    return nib.Nifti1Image(np.asarray(mask_vol.data, dtype=np.uint8), affine)


def test_baseline_and_neuroim_agree_on_smoothed_tsnr_qc(fixture, nib_bold_and_mask):
    bold_img, mask_img = nib_bold_and_mask
    base_img, base_summary = baseline.smoothed_tsnr_qc(
        bold_img,
        mask_img,
        fwhm_mm=6.0,
    )
    rew_vol, rew_summary = rewrite.smoothed_tsnr_qc(
        fixture.bold,
        fixture.mask,
        fwhm_mm=6.0,
    )

    np.testing.assert_allclose(rew_vol.data, base_img.get_fdata(), rtol=1e-6, atol=1e-6)
    assert rew_summary["n_voxels"] == base_summary["n_voxels"]
    assert rew_summary["fwhm_mm"] == pytest.approx(base_summary["fwhm_mm"])
    assert rew_summary["p50"] == pytest.approx(base_summary["p50"], rel=1e-6)
    assert rew_summary["p95"] == pytest.approx(base_summary["p95"], rel=1e-6)


def test_neuroim_rejects_mismatched_affine_smoothing_mask(fixture):
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        rewrite.smoothed_tsnr_qc(
            fixture.bold,
            fixture.mask,
            fwhm_mm=6.0,
            smoothing_mask=fixture.rotated_mask,
        )


def test_baseline_accepts_same_shape_mismatched_affine_smoothing_mask(
    fixture, nib_bold_and_mask
):
    bold_img, mask_img = nib_bold_and_mask
    rotated_mask_img = _mask_to_nifti(fixture.rotated_mask)
    out, summary = baseline.smoothed_tsnr_qc(
        bold_img,
        mask_img,
        fwhm_mm=6.0,
        smoothing_mask_img=rotated_mask_img,
    )
    assert out.shape == mask_img.shape
    assert summary["n_voxels"] == int(np.asarray(mask_img.dataobj, dtype=bool).sum())


def test_neuroim_terminal_map_chains_provenance_through_disk(tmp_path, fixture):
    smoothed, summary = rewrite.smoothed_tsnr_qc(
        fixture.bold,
        fixture.mask,
        fwhm_mm=6.0,
    )

    assert isinstance(smoothed.provenance, Receipt)
    assert smoothed.provenance.method_name == "temporal_snr+gaussian_blur"
    assert smoothed.provenance.radius == pytest.approx(6.0)
    assert summary["method_name"] == "temporal_snr+gaussian_blur"
    assert summary["provenance_chain"] == "temporal_snr+gaussian_blur"

    out = tmp_path / "smoothed_tsnr.nii.gz"
    nib.save(smoothed.to_nibabel(), str(out))
    reread = ni.io.read_image(str(out))
    assert reread.provenance.method_name == "temporal_snr+gaussian_blur"
    assert reread.provenance.radius == pytest.approx(6.0)
