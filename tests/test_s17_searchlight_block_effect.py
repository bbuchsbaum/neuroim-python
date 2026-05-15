"""Acceptance tests for Scenario 17 - local block-effect searchlight."""

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
    / "17_searchlight_block_effect"
)
baseline = _load_module("scenario17_baseline", _SCENARIO_DIR / "baseline_nibabel.py")
rewrite = _load_module("scenario17_rewrite", _SCENARIO_DIR / "neuroim_version.py")


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold(shape=(12, 12, 8, 24), seed=17)


@pytest.fixture(scope="module")
def condition(fixture):
    return np.asarray(fixture.regressor > 0.0, dtype=bool)


@pytest.fixture(scope="module")
def nib_bold_and_mask(fixture):
    return to_nibabel(fixture)


def _mask_to_nifti(mask_vol) -> nib.Nifti1Image:
    affine = np.asarray(mask_vol.space.trans, dtype=float)[:4, :4]
    return nib.Nifti1Image(np.asarray(mask_vol.data, dtype=np.uint8), affine)


def test_baseline_and_neuroim_agree_on_local_block_effect_map(
    fixture, condition, nib_bold_and_mask
):
    bold_img, mask_img = nib_bold_and_mask
    base_values, base_img, base_summary = baseline.local_block_effect_searchlight(
        bold_img,
        mask_img,
        condition,
        radius_mm=6.0,
    )
    result, summary = rewrite.local_block_effect_searchlight(
        fixture.bold,
        fixture.mask,
        condition,
        radius_mm=6.0,
    )

    np.testing.assert_allclose(result.values, base_values, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        result.map_to_volume().data,
        base_img.get_fdata(),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    assert summary["n_centers"] == base_summary["n_centers"]
    assert summary["radius_mm"] == pytest.approx(base_summary["radius_mm"])
    assert summary["max_effect"] == pytest.approx(base_summary["max_effect"])


def test_neuroim_rejects_mismatched_affine_searchlight_mask(fixture, condition):
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        rewrite.local_block_effect_searchlight(
            fixture.bold,
            fixture.rotated_mask,
            condition,
            radius_mm=6.0,
        )


def test_baseline_accepts_same_shape_mismatched_affine_searchlight_mask(
    fixture, condition, nib_bold_and_mask
):
    bold_img, _ = nib_bold_and_mask
    rotated_mask_img = _mask_to_nifti(fixture.rotated_mask)
    values, effect_map, summary = baseline.local_block_effect_searchlight(
        bold_img,
        rotated_mask_img,
        condition,
        radius_mm=6.0,
    )
    assert values.size == summary["n_centers"]
    assert effect_map.shape == bold_img.shape[:3]


def test_neuroim_result_is_typed_and_projects_with_receipt(fixture, condition):
    result, summary = rewrite.local_block_effect_searchlight(
        fixture.bold,
        fixture.mask,
        condition,
        radius_mm=6.0,
    )
    effect_map = result.map_to_volume()

    assert isinstance(result, ni.SearchlightResult)
    assert isinstance(result.provenance, Receipt)
    assert result.method_name == "local_block_effect"
    assert result.provenance.method_name == "local_block_effect"
    assert result.provenance.radius == pytest.approx(6.0)
    assert effect_map.provenance == result.provenance
    assert summary["method_name"] == "local_block_effect"


def test_target_region_lands_in_high_effect_tail(fixture, condition):
    result, _ = rewrite.local_block_effect_searchlight(
        fixture.bold,
        fixture.mask,
        condition,
        radius_mm=6.0,
    )
    effect_map = result.map_to_volume()
    target_values = np.asarray(
        [
            effect_map.data[int(i), int(j), int(k)]
            for i, j, k in fixture.target_roi_centers
        ],
        dtype=float,
    )
    target_values = target_values[np.isfinite(target_values)]

    assert target_values.size > 0
    assert float(np.nanmax(target_values)) > float(np.nanquantile(result.values, 0.90))
