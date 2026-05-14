"""Acceptance test for Scenario 05 — receipts across the IO boundary.

Three assertions:

1. **Baseline sidecar round-trip works** — the hand-rolled JSON
   manifest from the baseline lane can be read back from disk and
   carries every provenance field.
2. **In-memory neuroim Receipt is fully populated** — the
   :class:`~neuroim.results.SearchlightResult` produced before write
   carries method_name, n_voxels, radius, input_space_hash, mask_hash.
3. **(xfail, strict)** Neuroim's IO round-trip preserves the Receipt —
   ``read_provenance_from_file(path)`` returns a Receipt whose fields
   match the in-memory Receipt's.

Assertion 3 is the falsifying test for PAIN-6 (mission-bearing).  It
is currently expected to fail.  When the fix lands, the strict-xfail
will flip the test to passing automatically.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import neuroim as ni
from neuroim.results import Receipt, SearchlightResult

from fixtures.realistic_bold import make_realistic_bold, to_nibabel


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = Path(__file__).resolve().parent / "05_receipt_io_boundary"
baseline_nibabel = _load_module(
    "scenario05_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario05_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def nib_pair(fixture):
    return to_nibabel(fixture)


def test_baseline_sidecar_round_trip(tmp_path, nib_pair):
    """The hand-rolled JSON sidecar carries provenance across write/read."""
    bold_img, mask_img = nib_pair
    out_nii = tmp_path / "mean.nii.gz"
    out_json = tmp_path / "mean.json"
    baseline_nibabel.write_mean_volume_with_sidecar(
        bold_img, mask_img, out_nii, out_json
    )
    provenance = baseline_nibabel.read_provenance(out_nii, out_json)
    assert provenance["method_name"] == "mean_over_time"
    assert provenance["n_voxels"] == int(mask_img.get_fdata().astype(bool).sum())
    assert "input_space_hash" in provenance
    assert "mask_hash" in provenance


def test_neuroim_in_memory_receipt_is_fully_populated(fixture):
    """ME-9 baseline: the in-memory Receipt is fully populated before write."""
    sl = ni.searchlight_apply(
        fixture.mask,
        radius=4.5,
        method=lambda a: float(np.asarray(a).mean()),
        data=fixture.bold,
        cores=0,
    )
    assert isinstance(sl, SearchlightResult)
    rcpt = sl.provenance
    assert isinstance(rcpt, Receipt)
    assert rcpt.method_name
    assert rcpt.n_voxels > 0
    assert rcpt.radius == 4.5
    assert rcpt.input_space_hash
    assert rcpt.mask_hash


def test_neuroim_round_trip_preserves_receipt(tmp_path, fixture):
    """The Receipt survives the NIfTI write/read round-trip.

    Closes PAIN-6 (mote bd-01KRKR7SX4GKW1QZ9KF6G73ZWR): the fix embeds
    the Receipt as a NIfTI 'comment' header extension prefixed with the
    marker :data:`neuroim.results.RECEIPT_NIFTI_PREFIX`; the read path
    re-hydrates it onto the returned ``NeuroVol.provenance``.
    """
    out_path = tmp_path / "sl_mean.nii.gz"
    in_mem = neuroim_version.write_searchlight_mean(
        fixture.bold, fixture.mask, out_path
    )
    recovered = neuroim_version.read_provenance_from_file(out_path)
    assert recovered is not None, "Receipt not recovered from .nii.gz on disk"
    assert recovered.method_name == in_mem.provenance.method_name
    assert recovered.n_voxels == in_mem.provenance.n_voxels
    assert recovered.input_space_hash == in_mem.provenance.input_space_hash
    assert recovered.mask_hash == in_mem.provenance.mask_hash
