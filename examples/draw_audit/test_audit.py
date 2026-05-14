"""Acceptance test for the ME-3 Draw Audit.

Two assertions:
  1. **Numeric parity** — both implementations produce the same correlation
     map on the happy-path fixture.
  2. **Bug-class divergence** — when fed the LR-rotated mask (same data,
     different affine), the baseline silently produces a plausible-but-
     wrong map; the rewrite raises ``ValueError`` via the verifier.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from fixtures.realistic_bold import make_realistic_bold, to_nibabel
from examples.draw_audit import baseline_nibabel, neuroim_rewrite


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def _nib_pair(fixture):
    return to_nibabel(fixture)


@pytest.fixture(scope="module")
def nib_bold(_nib_pair):
    return _nib_pair[0]


@pytest.fixture(scope="module")
def nib_mask(_nib_pair):
    return _nib_pair[1]


@pytest.fixture(scope="module")
def nib_rotated_mask(fixture):
    rot_affine = np.asarray(fixture.rotated_mask.space.trans)[:4, :4]
    return nib.Nifti1Image(
        np.asarray(fixture.rotated_mask.data, dtype=np.uint8), rot_affine
    )


# ----------------------------------------------------------------------
# Happy path: same answer
# ----------------------------------------------------------------------


def test_happy_path_numeric_parity(fixture, nib_bold, nib_mask):
    """Baseline and rewrite produce the same correlation map on a valid
    space-matched fixture."""
    base_arr, _ = baseline_nibabel.correlate_roi_with_regressor(
        nib_bold, nib_mask, fixture.regressor
    )
    rewrite_vol, _ = neuroim_rewrite.correlate_roi_with_regressor(
        fixture.bold, fixture.mask, fixture.regressor
    )
    np.testing.assert_allclose(
        rewrite_vol.data, base_arr, equal_nan=True, rtol=1e-9, atol=1e-12,
    )


def test_rewrite_returns_typed_result_with_provenance(
    fixture, nib_bold, nib_mask
):
    """The rewrite ships an ROIExtractionResult with a populated Receipt."""
    rewrite_vol, extract = neuroim_rewrite.correlate_roi_with_regressor(
        fixture.bold, fixture.mask, fixture.regressor
    )
    # The output volume is a NeuroVol (typed), not a bare ndarray.
    import neuroim as ni
    assert isinstance(rewrite_vol, ni.NeuroVol)
    # The intermediate extraction carries provenance.
    assert extract.provenance.method_name == "series_roi"
    assert extract.provenance.input_space_hash != "none"
    assert extract.provenance.mask_hash != "none"
    # Output volume's space matches the extraction's space.
    from neuroim.results import hash_neurospace
    assert (
        hash_neurospace(rewrite_vol.space)
        == hash_neurospace(extract.space)
    )


# ----------------------------------------------------------------------
# Bug-class divergence: silent-wrong vs raised
# ----------------------------------------------------------------------


def test_baseline_silently_accepts_wrong_space_mask(
    fixture, nib_bold, nib_rotated_mask
):
    """The baseline runs to completion on an LR-flipped mask, silently
    producing a wrong-but-plausible correlation map.  This is the
    failure mode the mission claims to remove."""
    arr, _ = baseline_nibabel.correlate_roi_with_regressor(
        nib_bold, nib_rotated_mask, fixture.regressor
    )
    assert arr.shape == fixture.bold.shape[:3]
    # A correlation map was produced — no error was raised, even though
    # the mask's affine flips LR relative to the bold's affine.
    assert np.isfinite(arr[np.isfinite(arr)]).any()


def test_rewrite_raises_on_wrong_space_mask(fixture):
    """The rewrite's verifier catches the LR-flipped mask and raises a
    spatial-contract ValueError before any numbers are produced."""
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        neuroim_rewrite.correlate_roi_with_regressor(
            fixture.bold, fixture.rotated_mask, fixture.regressor
        )
