"""Unit tests for ROIExtractionResult.map_to_volume / to_nibabel (PAIN-7).

Mirrors the SearchlightResult pattern so ROI-shaped results can carry
their Receipt across the file boundary via the receipt-NIfTI-extension
contract (docs/spec/receipt-nifti-extension.md).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import ROIExtractionResult, Receipt, make_receipt


def _coords_in_corner(n: int = 4) -> np.ndarray:
    """A column of ``n`` voxels along the x-axis (fits in a 4x4x4 space)."""
    return np.column_stack([np.arange(n), np.zeros(n, int), np.zeros(n, int)])


def _sample_receipt() -> Receipt:
    return make_receipt(
        input_space=ni.NeuroSpace((4, 4, 4)),
        mask_data=_coords_in_corner(4),
        n_voxels=4,
        method_name="values_roi",
        seed=None,
    )


def test_map_to_volume_1d_values_returns_3d_densevol_with_provenance():
    coords = _coords_in_corner(4)
    values = np.array([10.0, 20.0, 30.0, 40.0])
    space = ni.NeuroSpace((4, 4, 4))
    result = ROIExtractionResult(
        values=values,
        coords=coords,
        space=space,
        mask_hash="m",
        provenance=_sample_receipt(),
    )
    vol = result.map_to_volume()
    assert isinstance(vol, ni.DenseNeuroVol)
    assert vol.data.shape == (4, 4, 4)
    np.testing.assert_array_equal(vol.data[:4, 0, 0], values)
    # Background voxels are the default fill (0).
    assert vol.data[3, 3, 3] == 0.0
    assert vol.provenance == result.provenance


def test_map_to_volume_2d_values_returns_4d_densevec_with_provenance():
    coords = _coords_in_corner(3)
    # values is (nt, n_voxels) per series_roi convention.
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float64,
    )
    space = ni.NeuroSpace((4, 4, 4))
    result = ROIExtractionResult(
        values=values,
        coords=coords,
        space=space,
        mask_hash="m",
        provenance=_sample_receipt(),
    )
    vec = result.map_to_volume()
    assert isinstance(vec, ni.DenseNeuroVec)
    assert vec.data.shape == (4, 4, 4, 4)
    # Per-voxel time series should be each column of values.
    np.testing.assert_array_equal(vec.data[0, 0, 0, :], values[:, 0])
    np.testing.assert_array_equal(vec.data[1, 0, 0, :], values[:, 1])
    np.testing.assert_array_equal(vec.data[2, 0, 0, :], values[:, 2])
    # Unmapped voxel stays at fill.
    assert vec.data[3, 3, 3, 0] == 0.0
    assert vec.provenance == result.provenance


def test_to_nibabel_chains_through_map_to_volume_and_embeds_receipt():
    coords = _coords_in_corner(4)
    values = np.array([10.0, 20.0, 30.0, 40.0])
    result = ROIExtractionResult(
        values=values,
        coords=coords,
        space=ni.NeuroSpace((4, 4, 4)),
        mask_hash="m",
        provenance=_sample_receipt(),
    )
    img = result.to_nibabel()
    exts = list(img.header.extensions)
    assert len(exts) == 1
    recovered = Receipt.from_nifti_extension_bytes(bytes(exts[0].get_content()))
    assert recovered == result.provenance


def test_disk_round_trip_recovers_provenance(tmp_path: Path):
    coords = _coords_in_corner(4)
    values = np.array([10.0, 20.0, 30.0, 40.0])
    result = ROIExtractionResult(
        values=values,
        coords=coords,
        space=ni.NeuroSpace((4, 4, 4)),
        mask_hash="m",
        provenance=_sample_receipt(),
    )
    out = tmp_path / "roi_map.nii.gz"
    nib.save(result.to_nibabel(), str(out))
    back = ni.read_image(str(out))
    assert hasattr(back, "provenance")
    assert back.provenance == result.provenance


def test_map_to_volume_rejects_higher_rank_values():
    result = ROIExtractionResult(
        values=np.zeros((2, 3, 4)),
        coords=_coords_in_corner(3),
        space=ni.NeuroSpace((4, 4, 4)),
        mask_hash="m",
        provenance=_sample_receipt(),
    )
    with pytest.raises(ValueError, match="values.ndim"):
        result.map_to_volume()
