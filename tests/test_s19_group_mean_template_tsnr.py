"""Acceptance test for Scenario 19 — group-mean tSNR across subjects.

The scenario asks five questions of the API:

1. **Numeric parity (happy path)** — the raw-nibabel group mean tSNR map
   matches the neuroim group mean tSNR map elementwise.

2. **Same-space gate at the group reduce** — a same-shape subject map
   with a foreign affine raises ``ValueError`` through the cross-subject
   contract.

3. **Per-subject chained provenance** — each subject's map already
   carries a ``resample_vec(...)+temporal_snr`` Receipt (S09 covered
   this; we re-assert here so the regression is local).

4. **Multi-input provenance through the group reduce** — the terminal
   group map's ``method_name`` records both upstream chains.

5. **NIfTI round-trip survives** — ``write_vol`` + ``read_image``
   recovers the terminal Receipt, so an offline collaborator inspecting
   only the saved ``.nii.gz`` can recover the second-level lineage.

The PAIN gates are three strict xfails — one per missing API today:

  * PAIN-1 — ``ni.group_mean`` (or ``ni.mean_volumes``) is not exposed.
  * PAIN-2 — ``ni.concat`` rejects 3-D ``NeuroVol`` inputs, so the
    "stack subject maps and reduce along the new axis" path is closed.
  * PAIN-3 — there is no public helper for multi-input Receipt merge
    that a custom reducer can call (the rewrite walks
    ``Receipt.merge`` by hand).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import neuroim as ni
from neuroim.results import Receipt


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
    / "19_group_mean_template_tsnr"
)
baseline = _load_module("scenario19_baseline", _SCENARIO_DIR / "baseline_nibabel.py")
rewrite = _load_module("scenario19_rewrite", _SCENARIO_DIR / "neuroim_version.py")


@pytest.fixture(scope="module")
def fixture():
    subjects, template = baseline.synthesize_subjects()
    typed_subjects, typed_template = rewrite.typed_from_nibabel(subjects, template)
    return subjects, template, typed_subjects, typed_template


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_numeric_parity_with_raw_nibabel(fixture):
    """Baseline and neuroim end-to-end produce the same group tSNR map."""
    subjects, template, typed_subjects, typed_template = fixture
    base_map = baseline.baseline_group_tsnr(subjects, template)
    rew_map = rewrite.neuroim_group_tsnr(typed_subjects, typed_template)

    base_arr = np.asarray(base_map.dataobj, dtype=np.float64)
    rew_arr = np.asarray(rew_map.data, dtype=np.float64)
    # Cross-engine resample tolerance: ``nibabel.processing.resample_from_to``
    # and neuroim's ``resample`` use slightly different interpolation kernels
    # at order=1 boundaries, producing per-voxel diffs of a few thousandths.
    # The relative diff stays under 1e-5 — far below the noise floor in any
    # downstream stat map.
    np.testing.assert_allclose(base_arr, rew_arr, rtol=1e-4, atol=2e-3)


def test_group_reduce_rejects_foreign_affine_subject_map(fixture):
    """A same-shape subject map on an LR-flipped affine must trip the gate."""
    _, _, typed_subjects, typed_template = fixture
    good_a = rewrite.per_subject_tsnr(typed_subjects[0], typed_template)
    good_b = rewrite.per_subject_tsnr(typed_subjects[1], typed_template)

    flipped_affine = np.asarray(good_b.space.trans, dtype=float).copy()
    flipped_affine[:, 0] = -flipped_affine[:, 0]
    flipped_space = ni.NeuroSpace.from_affine(flipped_affine, good_b.space.dim[:3])
    bad_b = ni.DenseNeuroVol(np.asarray(good_b.data).copy(), flipped_space)

    with pytest.raises(ValueError, match="spatial contract mismatch"):
        rewrite.group_mean_volumes([good_a, bad_b])


def test_per_subject_receipt_chains_resample_and_tsnr(fixture):
    """Per-subject map's Receipt already names both upstream stages."""
    _, _, typed_subjects, typed_template = fixture
    sub_map = rewrite.per_subject_tsnr(typed_subjects[0], typed_template)
    assert isinstance(sub_map.provenance, Receipt)
    method = sub_map.provenance.method_name
    # The exact substring "resample_vec" appears (S09 fix), and the
    # downstream "+temporal_snr" suffix appears via receipt_for(upstream=).
    assert "resample_vec" in method, method
    assert method.endswith("+temporal_snr") or "temporal_snr" in method, method


def test_group_map_records_multi_input_chain(fixture):
    """Terminal group map records both upstream chains and the group op."""
    _, _, typed_subjects, typed_template = fixture
    group_map = rewrite.neuroim_group_tsnr(typed_subjects, typed_template)
    assert isinstance(group_map.provenance, Receipt)
    method = group_map.provenance.method_name
    assert "resample_vec" in method, method
    assert "temporal_snr" in method, method
    assert "group_mean" in method, method
    assert method.endswith(")"), method  # records n_inputs as ``group_mean(n=2)``


def test_group_map_receipt_survives_nifti_round_trip(fixture, tmp_path_factory):
    """Write the group map via ``to_nibabel`` + ``nib.save`` and read it back.

    The receipt rehydrates onto ``reloaded.provenance``.  The careful
    user has to remember to write through ``to_nibabel`` rather than
    ``write_vol`` — see PAIN-4 below.
    """
    import nibabel as nib

    _, _, typed_subjects, typed_template = fixture
    group_map = rewrite.neuroim_group_tsnr(typed_subjects, typed_template)

    out_path = tmp_path_factory.mktemp("s19") / "group_tsnr.nii.gz"
    nib.save(group_map.to_nibabel(), str(out_path))

    reloaded = ni.io.read_image(str(out_path))
    rehydrated = getattr(reloaded, "provenance", None)
    assert isinstance(rehydrated, Receipt), (
        "group map Receipt did not survive to_nibabel + read_image"
    )
    assert rehydrated.method_name == group_map.provenance.method_name


# ---------------------------------------------------------------------------
# PAIN gates — strict xfails that flip to XPASS the moment each lands
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="PAIN-1: no first-class group_mean / mean_volumes")
def test_group_mean_is_in_public_api():
    """PAIN-1 closes when ``ni.group_mean`` (or ``ni.mean_volumes``) ships."""
    assert "group_mean" in ni.__all__ or "mean_volumes" in ni.__all__


@pytest.mark.xfail(
    strict=True,
    reason="PAIN-2: ni.concat does not accept 3-D NeuroVol inputs to stack into a NeuroVec",
)
def test_concat_can_stack_3d_volumes_into_vector(fixture):
    """PAIN-2 closes when ``ni.concat`` (or a sibling ``stack_volumes``) can
    take a sequence of 3-D ``DenseNeuroVol`` inputs and emit a 4-D
    ``DenseNeuroVec`` that the existing ``+`` chain can mean over the
    stacked axis.
    """
    _, _, typed_subjects, typed_template = fixture
    a = rewrite.per_subject_tsnr(typed_subjects[0], typed_template)
    b = rewrite.per_subject_tsnr(typed_subjects[1], typed_template)
    stacked = ni.concat(a, b)  # raises today (NeuroVec-only)
    assert stacked.shape[3] == 2


@pytest.mark.xfail(
    strict=True,
    reason="PAIN-4: write_vol does not embed the receipt; only to_nibabel does",
)
def test_write_vol_embeds_receipt_into_nifti(fixture, tmp_path_factory):
    """PAIN-4 closes when ``ni.write_vol(vol, path)`` writes the same NIfTI
    a ``nib.save(vol.to_nibabel(), path)`` would — i.e., the on-disk file
    carries the receipt comment extension and ``ni.read_image`` rehydrates
    ``.provenance``.  Today the user must remember the ``to_nibabel`` dance
    on every write or silently lose the chain at the IO boundary.
    """
    _, _, typed_subjects, typed_template = fixture
    group_map = rewrite.neuroim_group_tsnr(typed_subjects, typed_template)

    out_path = tmp_path_factory.mktemp("s19_pain4") / "group_tsnr.nii.gz"
    ni.write_vol(group_map, str(out_path))
    reloaded = ni.io.read_image(str(out_path))
    rehydrated = getattr(reloaded, "provenance", None)
    assert isinstance(rehydrated, Receipt)


@pytest.mark.xfail(
    strict=True,
    reason="PAIN-3: no public multi-input Receipt helper for custom reducers",
)
def test_public_multi_input_receipt_helper_exists():
    """PAIN-3 closes when ``neuroim.results`` ships a public helper named
    something like ``merge_receipts(*upstreams, params=...)`` that custom
    reducers can call instead of walking ``Receipt.merge`` by hand.
    """
    from neuroim import results

    candidates = ("merge_receipts", "group_receipt", "reduce_receipts")
    public = set(getattr(results, "__all__", ()))
    assert public & set(candidates), (
        "expected one of "
        f"{candidates} in neuroim.results.__all__; got {sorted(public)}"
    )
