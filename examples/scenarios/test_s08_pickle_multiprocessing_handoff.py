"""Acceptance tests for Scenario 08 — pickle / multiprocessing handoff.

The scenario models a joblib/multiprocessing-style boundary by pickling a
derived temporal-SNR map, unpickling it in a fresh Python subprocess, and
asking what the receiver can inspect.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fixtures.realistic_bold import make_realistic_bold, to_nibabel


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = Path(__file__).resolve().parent / "08_pickle_multiprocessing_handoff"
baseline_nibabel = _load_module(
    "scenario08_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario08_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


def test_bare_nibabel_pickle_preserves_image_but_no_provenance(fixture):
    bold_img, mask_img = to_nibabel(fixture)
    tsnr = baseline_nibabel.temporal_snr_image(bold_img, mask_img)
    summary = baseline_nibabel.inspect_payload_in_fresh_process(
        baseline_nibabel.pickle_payload(tsnr)
    )
    assert summary["image_type"] == "Nifti1Image"
    assert tuple(summary["shape"]) == fixture.mask.shape
    assert summary["has_manifest"] is False
    assert summary["method_name"] is None
    assert summary["input_space_hash"] is None
    assert summary["mask_hash"] is None


def test_careful_nibabel_user_can_hand_bundle_manifest(fixture):
    bold_img, mask_img = to_nibabel(fixture)
    tsnr = baseline_nibabel.temporal_snr_image(bold_img, mask_img)
    manifest = baseline_nibabel.manual_manifest(bold_img, mask_img)
    summary = baseline_nibabel.inspect_payload_in_fresh_process(
        baseline_nibabel.pickle_payload({"image": tsnr, "manifest": manifest})
    )
    assert summary["image_type"] == "Nifti1Image"
    assert summary["has_manifest"] is True
    assert summary["method_name"] == "temporal_snr"
    assert summary["input_space_hash"]
    assert summary["mask_hash"]


def test_neuroim_pickle_preserves_typed_space_and_receipt(fixture):
    tsnr = neuroim_version.temporal_snr_volume(fixture.bold, fixture.mask)
    summary = neuroim_version.inspect_payload_in_fresh_process(
        neuroim_version.pickle_payload(tsnr)
    )
    assert summary["payload_type"] == "DenseNeuroVol"
    assert tuple(summary["shape"]) == fixture.mask.shape
    assert summary["has_space"] is True
    assert summary["has_provenance"] is True
    assert summary["method_name"] == "temporal_snr"
    assert summary["input_space_hash"] == tsnr.provenance.input_space_hash
    assert summary["mask_hash"] == tsnr.provenance.mask_hash

