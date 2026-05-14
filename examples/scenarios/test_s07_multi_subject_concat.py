"""Acceptance test for Scenario 07 — multi-subject concat.

Four assertions:

1. **Matched-affine concat works** — two subjects with identical
   spatial frames produce a coherent merged time series whose masked
   mean matches the baseline.
2. **Baseline catches mismatched affine** — the hand-coded
   ``np.allclose(s.affine, ref.affine)`` check fires.
3. **(xfail, strict)** Neuroim catches mismatched affine — a
   :class:`NeuroVec.concat` of subjects whose spatial sub-affines
   disagree raises through the contract layer.  Today this fails;
   this is the falsifying gate for PAIN-8 (bd-01KRKTA660BNCJS20BB9F99VSK).
4. **Mismatched-shape concat is rejected** — both surfaces agree the
   shapes must match (the un-controversial half of contract).

Note: a "receipt doesn't silently inherit one subject's hash" corollary
was considered and dropped — the merged result's NeuroSpace has a
different ``dim[3]`` than either solo input's, so the input_space_hash
already differs today for the wrong reason (time-dim mismatch, not
affine-heterogeneity sentinel). A spatial-only sub-hash comparison
would be the right shape but requires API not yet present; defer until
the structural-provenance builder (bd-01KRKRZPDC6V5CZF7SH0C9KEDD) ships.
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


_SCENARIO_DIR = Path(__file__).resolve().parent / "07_multi_subject_concat"
baseline_nibabel = _load_module(
    "scenario07_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario07_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _shifted_subject(reference, *, seed: int) -> ni.DenseNeuroVec:
    """Build a 'subject B' with the same shape as reference but a shifted
    + LR-flipped affine.  Same noise structure for clean comparison."""
    other = make_realistic_bold(seed=seed)
    shape = np.asarray(other.bold.data).shape
    shifted = np.diag([3.0, 3.0, 3.5, 1.0]).astype(float)
    shifted[:3, 3] = [100.0, -50.0, 25.0]
    shifted[0, 0] = -3.0  # LR flip
    space = ni.NeuroSpace.from_affine(shifted, shape)
    return ni.DenseNeuroVec(np.asarray(other.bold.data).copy(), space)


@pytest.fixture(scope="module")
def subject_a():
    return make_realistic_bold(seed=1)


@pytest.fixture(scope="module")
def subject_b_matched(subject_a):
    """Subject B with the *same* affine as A (independent noise via different seed)."""
    other = make_realistic_bold(seed=2)
    # Force identical space (matched affine) by reusing A's space.
    return ni.DenseNeuroVec(np.asarray(other.bold.data).copy(), subject_a.bold.space)


@pytest.fixture(scope="module")
def subject_b_shifted(subject_a):
    return _shifted_subject(subject_a, seed=2)


# ---------------------------------------------------------------------------
# 1) Matched-affine concat — happy path
# ---------------------------------------------------------------------------


def test_matched_subjects_concat_numeric_parity(subject_a, subject_b_matched, tmp_path):
    bold_a_nib, mask_nib = to_nibabel(subject_a)
    bold_b_nib = nib.Nifti1Image(
        np.asarray(subject_b_matched.data, dtype=np.float64),
        np.asarray(subject_b_matched.space.trans)[:4, :4],
    )
    baseline = baseline_nibabel.concat_subjects_and_mean(
        [bold_a_nib, bold_b_nib], mask_nib
    )
    rewrite = neuroim_version.concat_subjects_and_mean(
        [subject_a.bold, subject_b_matched], subject_a.mask
    )
    assert baseline.shape == rewrite.shape == (80,)
    np.testing.assert_allclose(baseline, rewrite, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 2) Baseline catches mismatched affine
# ---------------------------------------------------------------------------


def test_baseline_catches_mismatched_affine(subject_a, subject_b_shifted):
    bold_a_nib, mask_nib = to_nibabel(subject_a)
    bold_b_nib = nib.Nifti1Image(
        np.asarray(subject_b_shifted.data, dtype=np.float64),
        np.asarray(subject_b_shifted.space.trans)[:4, :4],
    )
    with pytest.raises(ValueError, match="affine"):
        baseline_nibabel.concat_subjects_and_mean(
            [bold_a_nib, bold_b_nib], mask_nib
        )


# ---------------------------------------------------------------------------
# 3) PAIN-8: neuroim concat must refuse mismatched affine
# ---------------------------------------------------------------------------


def test_neuroim_concat_refuses_mismatched_affine_subjects(
    subject_a, subject_b_shifted
):
    """concat catches the affine mismatch via the contract layer.

    Originally filed PAIN-8 (P0, mission-bearing) on the suspicion this
    case scattered silently. Re-probing showed concat already calls
    ``verify.assert_same_space(self.space, other.space)``, which routes
    through ``NeuroSpace.compatible_with`` and raises on the LR-flipped
    + origin-shifted affine. PAIN-8 was retracted; the test now passes
    as the contract-layer-catches case it always should have been.
    """
    with pytest.raises((ValueError, TypeError)):
        neuroim_version.concat_subjects_and_mean(
            [subject_a.bold, subject_b_shifted], subject_a.mask
        )


# ---------------------------------------------------------------------------
# 4) Mismatched-shape concat is rejected (un-controversial)
# ---------------------------------------------------------------------------


def test_neuroim_concat_rejects_mismatched_spatial_shape(subject_a):
    # Build a subject with a different spatial shape — same template-style
    # affine but (16, 16, 12, 40) instead of (32, 32, 24, 40).
    smaller = make_realistic_bold(shape=(16, 16, 12, 40), seed=3)
    with pytest.raises((ValueError, TypeError)):
        subject_a.bold.concat(smaller.bold)
