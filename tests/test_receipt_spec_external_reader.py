"""Contract test for ``docs/spec/receipt-nifti-extension.md``.

The body of :func:`read_neuroim_receipt` is the exact 10-line snippet
published in the spec.  If this test breaks, the spec is wrong (or
the embed/extract path was changed in a way that requires a v2
marker bump).
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import Receipt, make_receipt


# ---------------------------------------------------------------------------
# Reference reader — copied verbatim from docs/spec/receipt-nifti-extension.md.
# Do NOT use any neuroim symbol inside this function; the whole point is that
# a third-party tool can read receipts with only nibabel + json.
# ---------------------------------------------------------------------------


def read_neuroim_receipt(path):
    """Return the neuroim v1 receipt dict, or ``None`` if absent."""
    img = nib.load(str(path))
    for ext in getattr(img.header, "extensions", []) or []:
        if int(ext.get_code()) != 6:
            continue
        text = bytes(ext.get_content()).rstrip(b"\x00").decode("utf-8", "replace")
        if text.startswith("neuroim/receipt/v1:"):
            return json.loads(text[len("neuroim/receipt/v1:"):])
    return None


# Reference writer — also copied verbatim from the spec.
def write_neuroim_receipt(img, receipt):
    from nibabel.nifti1 import Nifti1Extension

    payload = ("neuroim/receipt/v1:" + json.dumps(receipt, sort_keys=True)).encode(
        "utf-8"
    )
    img.header.extensions[:] = [
        ext
        for ext in img.header.extensions
        if not (
            int(ext.get_code()) == 6
            and bytes(ext.get_content())
            .rstrip(b"\x00")
            .decode("utf-8", "replace")
            .startswith("neuroim/receipt/v1:")
        )
    ]
    img.header.extensions.append(Nifti1Extension(6, payload))
    return img


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_receipt() -> Receipt:
    return make_receipt(
        input_space=ni.NeuroSpace((4, 4, 4)),
        mask_data=np.array([[0, 0, 0], [1, 1, 1]], dtype=int),
        radius=4.5,
        n_voxels=2,
        method_name="searchlight",
        seed=42,
    )


@pytest.fixture
def neuroim_written_nii(tmp_path: Path) -> Path:
    """A .nii.gz produced by neuroim, carrying a Receipt extension."""
    vol = ni.DenseNeuroVol(
        np.arange(27, dtype=np.float64).reshape(3, 3, 3),
        ni.NeuroSpace((3, 3, 3)),
    )
    vol.provenance = _sample_receipt()
    out = tmp_path / "produced_by_neuroim.nii.gz"
    nib.save(vol.to_nibabel(), str(out))
    return out


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_reference_reader_recovers_receipt_from_neuroim_file(neuroim_written_nii):
    """The 10-line reference reader works without importing neuroim."""
    recovered = read_neuroim_receipt(neuroim_written_nii)
    assert recovered is not None
    # All eight fields documented in the spec must be present.
    assert set(recovered.keys()) == {
        "input_space_hash",
        "mask_hash",
        "radius",
        "n_voxels",
        "method_name",
        "seed",
        "neuroim_version",
        "source_affine_hash",
    }
    expected = _sample_receipt()
    assert recovered["method_name"] == expected.method_name
    assert recovered["n_voxels"] == expected.n_voxels
    assert recovered["radius"] == expected.radius
    assert recovered["seed"] == expected.seed
    assert recovered["input_space_hash"] == expected.input_space_hash
    assert recovered["mask_hash"] == expected.mask_hash


def test_reference_reader_returns_none_when_no_extension(tmp_path: Path):
    """A plain nibabel file with no extensions yields ``None``, not an error."""
    img = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    out = tmp_path / "plain.nii.gz"
    nib.save(img, str(out))
    assert read_neuroim_receipt(out) is None


def test_reference_reader_ignores_foreign_comment_extensions(tmp_path: Path):
    """Other tools' 'comment' (ecode 6) extensions must not be mistaken for a receipt."""
    from nibabel.nifti1 import Nifti1Extension

    img = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    img.header.extensions.append(Nifti1Extension(6, b"AFNI sub-brick label"))
    out = tmp_path / "foreign_comment.nii.gz"
    nib.save(img, str(out))
    assert read_neuroim_receipt(out) is None


def test_reference_writer_produces_a_file_neuroim_can_re_hydrate(tmp_path: Path):
    """A file written via the reference writer must round-trip back into neuroim."""
    receipt_dict = {
        "input_space_hash": "abcd1234abcd1234",
        "mask_hash": "ef01ef01ef01ef01",
        "radius": 6.0,
        "n_voxels": 100,
        "method_name": "external_tool_simulated",
        "seed": None,
        "neuroim_version": "0.0.0-external",
        "source_affine_hash": "0000000000000000",
    }
    img = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    write_neuroim_receipt(img, receipt_dict)
    out = tmp_path / "from_external.nii.gz"
    nib.save(img, str(out))

    recovered_via_neuroim = ni.read_image(str(out))
    assert hasattr(recovered_via_neuroim, "provenance")
    rcpt = recovered_via_neuroim.provenance
    assert rcpt.method_name == receipt_dict["method_name"]
    assert rcpt.n_voxels == receipt_dict["n_voxels"]
    assert rcpt.radius == receipt_dict["radius"]
    assert rcpt.input_space_hash == receipt_dict["input_space_hash"]


def test_v2_marker_is_ignored_by_v1_reader(tmp_path: Path):
    """Future-proofing: a v2 marker must not be mistaken for v1 by this reader.

    The spec promises that v1 readers ignore unknown prefixes without raising.
    """
    from nibabel.nifti1 import Nifti1Extension

    img = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    img.header.extensions.append(
        Nifti1Extension(6, b"neuroim/receipt/v2:{\"future\": true}")
    )
    out = tmp_path / "v2_marker.nii.gz"
    nib.save(img, str(out))
    assert read_neuroim_receipt(out) is None
