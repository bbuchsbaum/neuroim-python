"""Acceptance tests for Scenario 14 — time-axis slicing as silent provenance loss."""

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


_SCENARIO_DIR = Path(__file__).resolve().parent / "14_temporal_slice_provenance"
baseline_nibabel = _load_module(
    "scenario14_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario14_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


_DROP_FIRST = 4


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold(seed=14)


def test_baseline_manifest_records_temporal_slice(fixture):
    bold_img, mask_img = to_nibabel(fixture)
    out, manifest = baseline_nibabel.temporal_snr_after_slice(
        bold_img, mask_img, start=_DROP_FIRST
    )
    assert out.shape == fixture.mask.shape
    assert manifest["method_name"] == "temporal_slice+temporal_snr"
    assert manifest["temporal_slice_start"] == _DROP_FIRST
    assert manifest["n_timepoints_in"] == fixture.bold.shape[-1]
    assert manifest["n_timepoints_used"] == fixture.bold.shape[-1] - _DROP_FIRST
    assert manifest["mask_hash"]


def test_neuroim_tsnr_after_slice_is_typed_and_compatible(fixture):
    out = neuroim_version.temporal_snr_after_slice(
        fixture.bold, fixture.mask, start=_DROP_FIRST
    )
    assert isinstance(out, ni.DenseNeuroVol)
    assert out.space.compatible_with(fixture.mask.space)
    assert "temporal_snr" in out.provenance.method_name
    assert out.provenance.mask_hash


@pytest.mark.xfail(strict=True, reason="PAIN-13: NeuroVec.__getitem__ time-slice drops type")
def test_pain_13_time_axis_slice_preserves_neurovec_type(fixture):
    sliced = fixture.bold[..., _DROP_FIRST:]
    assert isinstance(sliced, ni.NeuroVec), (
        f"bold[..., {_DROP_FIRST}:] returned {type(sliced).__name__}, "
        "expected NeuroVec (PAIN-13)"
    )
    assert sliced.shape[-1] == fixture.bold.shape[-1] - _DROP_FIRST


@pytest.mark.xfail(strict=True, reason="PAIN-14: TemporalSliceParams missing, no chain through slice -> temporal_snr")
def test_pain_14_receipt_records_temporal_slice_then_temporal_snr(fixture):
    out = neuroim_version.temporal_snr_after_slice(
        fixture.bold, fixture.mask, start=_DROP_FIRST
    )
    method = out.provenance.method_name
    assert "temporal_slice" in method, (
        f"method_name={method!r} does not record the upstream temporal slice (PAIN-14)"
    )
    assert "temporal_snr" in method
    assert f"start={_DROP_FIRST}" in method
