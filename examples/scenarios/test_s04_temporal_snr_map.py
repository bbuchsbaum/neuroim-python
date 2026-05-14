"""Acceptance test for Scenario 04 -- Temporal SNR Map.

Six green assertions pin the landed behavior:

1. numeric parity against the nibabel baseline,
2. typed 3-D output with the fixture's spatial frame,
3. mask-affine mismatch raises through neuroim's verifier,
4. zero-variance voxels are zero-filled,
5. ``NeuroVec.temporal_snr(mask=...)`` is a first-class method,
6. the produced map carries a populated Receipt (method name + mask hash).

The last two assertions are the contracts that opened as ``xfail(strict=True)``
when the scenario was first filed; both now land as plain assertions.
"""

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


_SCENARIO_DIR = Path(__file__).resolve().parent / "04_temporal_snr_map"
baseline_nibabel = _load_module(
    "scenario04_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario04_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def nib_pair(fixture):
    return to_nibabel(fixture)


def test_baseline_and_neuroim_tsnr_maps_agree(fixture, nib_pair):
    bold_img, mask_img = nib_pair
    baseline = baseline_nibabel.temporal_snr_map(bold_img, mask_img)
    rewrite = neuroim_version.temporal_snr_map(fixture.bold, fixture.mask)

    assert rewrite.shape == fixture.mask.shape
    np.testing.assert_allclose(
        np.asarray(rewrite.data),
        np.asarray(baseline.get_fdata()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_neuroim_output_is_spatial_3d_volume(fixture):
    result = neuroim_version.temporal_snr_map(fixture.bold, fixture.mask)

    assert isinstance(result, ni.DenseNeuroVol)
    assert result.space.compatible_with(fixture.mask.space)
    assert result.space.compatible_with(fixture.bold.spatial_space)


def test_mask_affine_mismatch_raises_through_neuroim(fixture):
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        neuroim_version.temporal_snr_map(fixture.bold, fixture.rotated_mask)


def test_zero_variance_voxels_are_zero_filled(fixture):
    data = np.asarray(fixture.bold.data).copy()
    data[0, 0, 0, :] = 7.0
    mask_data = np.zeros(tuple(fixture.bold.shape[:3]), dtype=bool)
    mask_data[0, 0, 0] = True

    bold = ni.DenseNeuroVec(data, fixture.bold.space)
    mask = ni.LogicalNeuroVol(mask_data, fixture.bold.spatial_space)
    result = neuroim_version.temporal_snr_map(bold, mask)

    assert result.data[0, 0, 0] == 0.0


def test_neurovec_exposes_first_class_temporal_snr(fixture):
    result = fixture.bold.temporal_snr(mask=fixture.mask)
    assert isinstance(result, ni.DenseNeuroVol)


def test_temporal_snr_map_carries_provenance(fixture):
    result = neuroim_version.temporal_snr_map(fixture.bold, fixture.mask)
    assert result.provenance.method_name == "temporal_snr"
    assert result.provenance.mask_hash != "none"
