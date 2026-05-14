"""Acceptance test for Scenario 02 — ROI mean time series.

Lives here (one level up from ``02_roi_mean_timeseries/``) for the same
reason as scenario 01: the leading digit in the folder name makes it
an illegal Python module under pytest's default import mode.  Both
implementations are loaded by path via :mod:`importlib`.

Four assertions:

1. **Numeric parity (happy path)** — the simple neuroim form returns
   the same length-``nt`` ndarray as the baseline.
2. **Typed parity** — the canonical typed form's
   ``.values.mean(axis=1)`` matches the baseline numerically, and the
   result carries a populated Receipt.
3. **Mask-space mismatch is caught by neuroim** — a deliberately
   LR-flipped mask raises through the API rather than scattering
   wrong-but-finite numbers into the result (this is *the* mission
   claim under test).
4. **Empty-mask parity** — both implementations reject an empty mask.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import ROIExtractionResult, Receipt

from fixtures.realistic_bold import make_realistic_bold, to_nibabel


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = Path(__file__).resolve().parent / "02_roi_mean_timeseries"
baseline_nibabel = _load_module(
    "scenario02_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario02_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def nib_pair(fixture):
    return to_nibabel(fixture)


def test_numeric_parity_simple_form(fixture, nib_pair):
    """Baseline and the simple neuroim form return the same time series."""
    bold_img, mask_img = nib_pair
    baseline = baseline_nibabel.roi_mean_timeseries(bold_img, mask_img)
    rewrite = neuroim_version.roi_mean_timeseries(fixture.bold, fixture.mask)
    assert baseline.shape == rewrite.shape == (40,)
    np.testing.assert_allclose(baseline, rewrite, rtol=1e-10, atol=1e-10)


def test_typed_form_matches_and_ships_receipt(fixture, nib_pair):
    """Canonical typed form matches baseline numerically and carries a Receipt."""
    bold_img, mask_img = nib_pair
    baseline = baseline_nibabel.roi_mean_timeseries(bold_img, mask_img)

    result = neuroim_version.roi_mean_timeseries_typed(fixture.bold, fixture.mask)
    assert isinstance(result, ROIExtractionResult)
    assert isinstance(result.provenance, Receipt)
    assert result.provenance.method_name == "series_roi"
    assert result.provenance.n_voxels == int(np.asarray(fixture.mask.data).sum())

    mean_ts = np.asarray(result.values).mean(axis=1)
    np.testing.assert_allclose(baseline, mean_ts, rtol=1e-10, atol=1e-10)


def test_mask_space_mismatch_caught_by_baseline_affine_check(fixture, nib_pair):
    """The baseline catches a flipped-affine mask via explicit affine compare."""
    bold_img, _ = nib_pair
    rotated_mask_img = nib.Nifti1Image(
        np.asarray(fixture.rotated_mask.data, dtype=np.uint8),
        np.asarray(fixture.rotated_mask.space.trans)[:4, :4],
    )
    with pytest.raises(ValueError, match="affine"):
        baseline_nibabel.roi_mean_timeseries(bold_img, rotated_mask_img)


def test_mask_space_mismatch_raises_through_neuroim(fixture):
    """When the bug is fixed, neuroim must surface the mismatch as a typed error."""
    with pytest.raises((ValueError, TypeError)):
        neuroim_version.roi_mean_timeseries(fixture.bold, fixture.rotated_mask)


def test_empty_mask_is_rejected_by_both(fixture, nib_pair):
    bold_img, _ = nib_pair
    empty = np.zeros_like(np.asarray(fixture.mask.data), dtype=np.uint8)
    empty_img = nib.Nifti1Image(empty, np.asarray(fixture.mask.space.trans)[:4, :4])
    empty_mask = ni.LogicalNeuroVol(empty.astype(bool), fixture.mask.space)

    with pytest.raises(ValueError, match="empty"):
        baseline_nibabel.roi_mean_timeseries(bold_img, empty_img)
    with pytest.raises(ValueError, match="empty"):
        neuroim_version.roi_mean_timeseries(fixture.bold, empty_mask)
