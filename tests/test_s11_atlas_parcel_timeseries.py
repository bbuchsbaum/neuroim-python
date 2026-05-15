"""Acceptance test for Scenario 11 -- atlas-based parcel time series.

Lives in tests/ (not examples/scenarios/) for the cleanest test-collection
path; the scenarios/ tree's conftest.py excludes digit-prefixed folders
from collection, so the runnable test for each scenario lives one level
up.  See examples/scenarios/test_s02_roi_mean_timeseries.py for the
pattern.

Six assertions (all passing now that PAIN-1 / PAIN-3 / PAIN-4 have landed):

1. **Numeric parity (happy path)** -- baseline and neuroim produce the
   same ``(n_clusters, n_time)`` matrix on a matched-space atlas + BOLD.
2. **neuroim catches mismatched-affine atlas** -- the LR-flipped atlas
   raises ``ValueError`` through ``assert_same_space`` before any
   averaging.  The mission claim under test.  PAIN-2 (the same-space
   gate) lives inside ``parcel_means`` and surfaces through the
   first-class API.
3. **baseline silently accepts mismatched-affine atlas** -- the same
   nibabel pipeline returns a wrong-but-plausible matrix on the
   LR-flipped atlas.  This is *not* a baseline indictment per se; it is
   evidence that the bug class is real and exists without neuroim's
   contract layer.
4. **PAIN-1 closed** -- ``DenseNeuroVec.parcel_means(cvol)`` and
   ``ClusteredNeuroVec.from_neurovec(vec, cvol)`` ship as first-class
   extraction APIs.
5. **PAIN-3 closed** -- ``ClusteredNeuroVec`` carries a populated
   ``.provenance`` Receipt with ``method_name == "parcel_means"`` and
   ``n_voxels == n_clusters``.
6. **PAIN-4 closed** -- ``ClusteredNeuroVec`` / ``ClusteredNeuroVol``
   are in ``ni.__all__``; users following the curated public API can
   discover them.
"""

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
    / "11_atlas_parcel_timeseries"
)
baseline_nibabel = _load_module(
    "scenario11_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario11_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def atlas(fixture):
    return make_atlas(fixture)


@pytest.fixture(scope="module")
def nib_bold(fixture):
    bold_img, _ = to_nibabel(fixture)
    return bold_img


def _atlas_to_nifti(atlas_vol) -> nib.Nifti1Image:
    """Project an integer-labelled DenseNeuroVol atlas to nibabel."""
    affine = np.asarray(atlas_vol.space.trans, dtype=float)[:4, :4]
    return nib.Nifti1Image(np.asarray(atlas_vol.data, dtype=np.int32), affine)


# ----------------------------------------------------------------------
# Happy path: numeric parity
# ----------------------------------------------------------------------


def test_baseline_and_neuroim_agree_on_matched_atlas(fixture, atlas, nib_bold):
    """Same input atlas + BOLD -> same (N, T) matrix from both paths."""
    base_labels, base_ts = baseline_nibabel.parcel_timeseries(
        nib_bold, _atlas_to_nifti(atlas)
    )
    rewrite = neuroim_version.parcel_timeseries(fixture.bold, atlas)

    # Baseline returns (N, T); rewrite stores (T, N).  Compare on aligned shape.
    assert base_ts.shape == (base_labels.size, fixture.bold.shape[-1])
    assert rewrite.ts.shape == (fixture.bold.shape[-1], base_labels.size)
    np.testing.assert_allclose(base_ts.T, rewrite.ts, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------
# Mission-bearing PAIN-2 gate (manual call site today)
# ----------------------------------------------------------------------


def test_neuroim_rejects_mismatched_affine_atlas(fixture, atlas):
    """An LR-flipped atlas must raise through ``assert_same_space``.

    Today the gate is in the rewrite's manual call site; when an extraction
    API lands (PAIN-1), the gate moves into it and this test continues to
    pass without modification.
    """
    rotated = make_rotated_atlas(atlas)
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        neuroim_version.parcel_timeseries(fixture.bold, rotated)


def test_baseline_silently_accepts_mismatched_affine_atlas(fixture, atlas, nib_bold):
    """The bug class is real: nibabel + numpy returns a plausible matrix.

    The baseline only compares spatial shape, so an LR-flipped atlas (same
    dims, different affine) sails through.  This is not an indictment of
    nibabel -- it is evidence that ``neuroim``'s same-space contract gate
    catches a class of silent misuse that bare-array pipelines do not.
    """
    rotated = make_rotated_atlas(atlas)
    rotated_img = _atlas_to_nifti(rotated)
    labels, ts = baseline_nibabel.parcel_timeseries(nib_bold, rotated_img)
    # Same n_clusters and time axis as the matched case -- shape alone reveals
    # nothing wrong.  The numeric content differs because the label-to-voxel
    # assignment is now wrong, but the API surface is silent about it.
    assert ts.shape == (labels.size, fixture.bold.shape[-1])


# ----------------------------------------------------------------------
# PAIN-1 / PAIN-3 / PAIN-4 gates
# ----------------------------------------------------------------------


def test_neurovec_exposes_first_class_parcel_means(fixture, atlas):
    assert hasattr(ni.DenseNeuroVec, "parcel_means")
    assert hasattr(ni.ClusteredNeuroVec, "from_neurovec")

    direct = fixture.bold.parcel_means(atlas)
    factory = ni.ClusteredNeuroVec.from_neurovec(fixture.bold, direct.cvol)
    np.testing.assert_allclose(direct.ts, factory.ts, rtol=1e-10, atol=1e-10)


def test_scenario_rewrite_accepts_typed_atlas_with_source_provenance(
    fixture, atlas, nib_bold
):
    typed_atlas = neuroim_version.typed_schaefer_fixture(atlas)

    base_labels, base_ts = baseline_nibabel.parcel_timeseries(
        nib_bold, _atlas_to_nifti(atlas)
    )
    rewrite = neuroim_version.parcel_timeseries(fixture.bold, typed_atlas)

    np.testing.assert_allclose(base_ts.T, rewrite.ts, rtol=1e-10, atol=1e-10)
    assert rewrite.atlas_provenance.family == "schaefer"
    assert rewrite.atlas_provenance.canonical_source == "ThomasYeoLab/CBIG"
    assert rewrite.atlas_provenance.delivery_backend == "scenario_fixture"
    assert rewrite.atlas_provenance.label_table_hash != "none"
    assert rewrite.atlas_provenance.image_hash != "none"


def test_clustered_neuro_vec_carries_provenance(fixture, atlas):
    cv = fixture.bold.parcel_means(atlas)
    assert isinstance(cv.provenance, Receipt)
    assert cv.provenance.method_name == "parcel_means"
    assert cv.provenance.n_voxels == cv.n_clusters


def test_clustered_classes_are_publicly_exported():
    """ClusteredNeuroVec and ClusteredNeuroVol are part of the curated public API.

    Closes the PAIN-4 gate from Scenario 11: users following ``ni.__all__``
    (the documented public surface) can now discover the atlas-workflow
    classes without reaching into ``neuroim.clustered_neuro_vec`` /
    ``neuroim.clustered_neuro_vol``.
    """
    assert "ClusteredNeuroVec" in ni.__all__
    assert "ClusteredNeuroVol" in ni.__all__
