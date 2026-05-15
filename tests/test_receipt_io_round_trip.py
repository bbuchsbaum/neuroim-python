"""Unit tests for the Receipt IO boundary (PAIN-6 fix).

Targets the three new surfaces independently of the higher-level
scenario test:

- :meth:`neuroim.results.Receipt.to_json` /
  :meth:`~neuroim.results.Receipt.from_json` round-trip.
- :meth:`~neuroim.results.Receipt.to_nifti_extension_bytes` /
  :meth:`~neuroim.results.Receipt.from_nifti_extension_bytes` round-trip,
  including the marker-prefix guard against foreign comment extensions.
- :meth:`~neuroim.neuro_vol.NeuroVol.to_nibabel` embeds the Receipt; a
  ``nib.save`` + ``neuroim.read_image`` round-trip rehydrates it onto
  ``result.provenance``.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import (
    RECEIPT_NIFTI_PREFIX,
    Receipt,
    make_receipt,
)


def _sample_receipt() -> Receipt:
    return make_receipt(
        input_space=ni.NeuroSpace((4, 4, 4)),
        mask_data=np.array([[0, 0, 0], [1, 1, 1]], dtype=int),
        radius=4.5,
        n_voxels=2,
        method_name="unit-test",
        seed=42,
    )


def test_receipt_json_round_trip_is_lossless():
    receipt = _sample_receipt()
    recovered = Receipt.from_json(receipt.to_json())
    assert recovered == receipt


def test_receipt_extension_bytes_round_trip():
    receipt = _sample_receipt()
    payload = receipt.to_nifti_extension_bytes()
    assert payload.startswith(RECEIPT_NIFTI_PREFIX.encode("utf-8"))
    recovered = Receipt.from_nifti_extension_bytes(payload)
    assert recovered == receipt


def test_extension_bytes_without_marker_returns_none():
    """A foreign 'comment' extension must not be mistaken for a Receipt."""
    assert Receipt.from_nifti_extension_bytes(b"AFNI metadata") is None
    assert Receipt.from_nifti_extension_bytes(b"") is None


def test_extension_bytes_with_corrupt_json_returns_none():
    bad = (RECEIPT_NIFTI_PREFIX + "{not-json").encode("utf-8")
    assert Receipt.from_nifti_extension_bytes(bad) is None


def test_to_nibabel_embeds_receipt_extension():
    vol = ni.DenseNeuroVol(np.ones((3, 3, 3), dtype=np.float64), ni.NeuroSpace((3, 3, 3)))
    vol.provenance = _sample_receipt()

    img = vol.to_nibabel()
    exts = list(img.header.extensions)
    assert len(exts) == 1
    assert int(exts[0].get_code()) == 6
    recovered = Receipt.from_nifti_extension_bytes(bytes(exts[0].get_content()))
    assert recovered == vol.provenance


def test_to_nibabel_without_provenance_does_not_add_extension():
    vol = ni.DenseNeuroVol(np.zeros((2, 2, 2)), ni.NeuroSpace((2, 2, 2)))
    img = vol.to_nibabel()
    assert len(list(img.header.extensions)) == 0


def test_disk_round_trip_via_read_image(tmp_path: Path):
    """Write a NeuroVol carrying a Receipt; read it back in a clean state."""
    vol = ni.DenseNeuroVol(np.arange(27, dtype=np.float64).reshape(3, 3, 3), ni.NeuroSpace((3, 3, 3)))
    vol.provenance = _sample_receipt()

    out = tmp_path / "provenance.nii.gz"
    nib.save(vol.to_nibabel(), str(out))

    recovered_img = ni.io.read_image(str(out))
    assert hasattr(recovered_img, "provenance")
    assert recovered_img.provenance == vol.provenance


def test_to_nibabel_replaces_prior_receipt_rather_than_stacking(tmp_path: Path):
    """A second ``to_nibabel`` after re-assigning provenance keeps one extension."""
    vol = ni.DenseNeuroVol(np.zeros((3, 3, 3)), ni.NeuroSpace((3, 3, 3)))
    vol.provenance = _sample_receipt()
    first = vol.to_nibabel()
    assert len(list(first.header.extensions)) == 1

    # Re-attach the same header to the vol and write again.
    vol._nibabel_header = first.header.copy()
    vol.provenance = _sample_receipt()
    second = vol.to_nibabel()
    receipt_exts = [
        ext
        for ext in second.header.extensions
        if Receipt.from_nifti_extension_bytes(bytes(ext.get_content())) is not None
    ]
    assert len(receipt_exts) == 1


def test_searchlight_result_to_nibabel_chains_through_map_to_volume():
    """SearchlightResult.to_nibabel must propagate the Receipt through map_to_volume."""
    from neuroim.results import SearchlightResult

    space = ni.NeuroSpace((4, 4, 4))
    receipt = _sample_receipt()
    sl = SearchlightResult(
        values=np.array([1.0]),
        centers=np.array([[1, 1, 1]]),
        space=space,
        radius=4.5,
        shape="sphere",
        provenance=receipt,
    )
    img = sl.to_nibabel()
    exts = list(img.header.extensions)
    assert len(exts) == 1
    recovered = Receipt.from_nifti_extension_bytes(bytes(exts[0].get_content()))
    assert recovered == receipt
