"""Acceptance tests for Scenario 15 - block-design parcel contrast."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import Receipt

from fixtures.realistic_bold import (
    make_atlas,
    make_realistic_bold,
    make_rotated_atlas,
    to_nibabel,
)


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
    / "15_block_parcel_contrast"
)
baseline_nibabel = _load_module(
    "scenario15_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario15_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold(seed=15)


@pytest.fixture(scope="module")
def atlas(fixture):
    return make_atlas(fixture, bins=(8, 8, 6))


@pytest.fixture(scope="module")
def condition(fixture):
    return np.asarray(fixture.regressor > 0.0, dtype=bool)


@pytest.fixture(scope="module")
def nib_bold(fixture):
    bold_img, _ = to_nibabel(fixture)
    return bold_img


def _atlas_to_nifti(atlas_vol) -> nib.Nifti1Image:
    affine = np.asarray(atlas_vol.space.trans, dtype=float)[:4, :4]
    return nib.Nifti1Image(np.asarray(atlas_vol.data, dtype=np.int32), affine)


def test_baseline_and_neuroim_agree_on_block_parcel_effects(
    fixture, atlas, condition, nib_bold
):
    base_labels, base_effects, base_map = baseline_nibabel.block_parcel_contrast(
        nib_bold,
        _atlas_to_nifti(atlas),
        condition,
    )
    typed_atlas = neuroim_version.schaefer_fixture(atlas)
    rewrite = neuroim_version.block_parcel_contrast(
        fixture.bold,
        typed_atlas,
        condition,
    )

    np.testing.assert_array_equal(rewrite.labels, base_labels)
    np.testing.assert_allclose(rewrite.effects, base_effects, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        rewrite.map_to_volume().data,
        base_map.get_fdata(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_neuroim_rejects_mismatched_affine_atlas(fixture, atlas, condition):
    rotated = make_rotated_atlas(atlas)
    typed_rotated = neuroim_version.schaefer_fixture(rotated)
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        neuroim_version.block_parcel_contrast(fixture.bold, typed_rotated, condition)


def test_baseline_accepts_same_shape_mismatched_affine_atlas(atlas, condition, nib_bold):
    rotated = make_rotated_atlas(atlas)
    labels, effects, effect_map = baseline_nibabel.block_parcel_contrast(
        nib_bold,
        _atlas_to_nifti(rotated),
        condition,
    )
    assert labels.size == effects.size
    assert effect_map.shape == nib_bold.shape[:3]


def test_neuroim_result_preserves_typed_atlas_and_receipt(fixture, atlas, condition):
    typed_atlas = neuroim_version.schaefer_fixture(atlas)
    result = neuroim_version.block_parcel_contrast(fixture.bold, typed_atlas, condition)

    assert isinstance(result, ni.ParcelEffectResult)
    assert isinstance(result.provenance, Receipt)
    assert result.provenance.method_name == "parcel_means+contrast[task-rest]"
    assert result.map_to_volume().provenance == result.provenance
    assert result.atlas_provenance.family == "schaefer"
    assert result.atlas_provenance.delivery_backend == "scenario_fixture"


def test_target_roi_parcel_lands_in_high_effect_tail(fixture, atlas, condition):
    typed_atlas = neuroim_version.schaefer_fixture(atlas)
    result = neuroim_version.block_parcel_contrast(fixture.bold, typed_atlas, condition)

    target_labels = np.asarray(
        [
            atlas.data[int(i), int(j), int(k)]
            for i, j, k in fixture.target_roi_centers
        ],
        dtype=np.int32,
    )
    target_labels = target_labels[target_labels != 0]
    target_effects = [
        result.effects[list(result.labels).index(label_id)]
        for label_id in set(int(label) for label in target_labels)
    ]
    assert target_labels.size > 0
    assert max(target_effects) > np.quantile(result.effects, 0.90)
