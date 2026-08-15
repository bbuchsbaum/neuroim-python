"""Regression contracts for exact spatial reorientation.

Reorientation is a lossless relabelling of the voxel grid.  These tests use
nibabel as the differential oracle and also check the stronger invariant that
every output voxel still maps to the same world point as its source voxel.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.compat import as_canonical
from neuroim.resample import reorient


AIR_AFFINE = np.array(
    [
        [0.0, 0.0, 2.0, -10.0],
        [3.0, 0.0, 0.0, -20.0],
        [0.0, -4.0, 0.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _volume(affine: np.ndarray = AIR_AFFINE) -> ni.DenseNeuroVol:
    shape = (3, 4, 5)
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape, order="F")
    return ni.DenseNeuroVol(data, ni.NeuroSpace(shape, trans=affine))


def _nibabel_reorient(data, affine, target):
    start = nib.orientations.io_orientation(affine)
    end = nib.orientations.axcodes2ornt(tuple(target))
    transform = nib.orientations.ornt_transform(start, end)
    expected_data = nib.orientations.apply_orientation(data, transform)
    expected_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    return expected_data, expected_affine, transform


def test_reorient_ras_matches_nibabel_voxel_for_voxel():
    vol = _volume()
    expected = nib.as_closest_canonical(nib.Nifti1Image(vol.data, vol.space.affine))

    observed = reorient(vol, "RAS")

    assert observed.shape == expected.shape == (5, 3, 4)
    assert observed.data.dtype == vol.data.dtype
    np.testing.assert_array_equal(observed.data, np.asanyarray(expected.dataobj))
    np.testing.assert_allclose(observed.space.affine, expected.affine)
    assert nib.aff2axcodes(observed.space.affine) == ("R", "A", "S")


@pytest.mark.parametrize("target", ["RAS", "LPI", "PIR", "ASR"])
def test_reorient_matches_general_nibabel_orientation_oracle(target):
    vol = _volume()
    expected_data, expected_affine, _ = _nibabel_reorient(
        vol.data, vol.space.affine, target
    )

    observed = reorient(vol, target)

    np.testing.assert_array_equal(observed.data, expected_data)
    np.testing.assert_allclose(observed.space.affine, expected_affine)
    assert nib.aff2axcodes(observed.space.affine) == tuple(target)


def test_reorient_preserves_world_location_of_every_voxel():
    vol = _volume()
    observed = reorient(vol, "RAS")
    _, _, transform = _nibabel_reorient(vol.data, vol.space.affine, "RAS")
    new_to_old = nib.orientations.inv_ornt_aff(transform, vol.shape)

    new_grid = np.indices(observed.shape).reshape(3, -1).T
    old_grid = nib.affines.apply_affine(new_to_old, new_grid)
    expected_world = nib.affines.apply_affine(vol.space.affine, old_grid)
    observed_world = nib.affines.apply_affine(observed.space.affine, new_grid)

    np.testing.assert_allclose(observed_world, expected_world)


def test_reorient_noop_and_roundtrip_are_exact():
    vol = _volume()
    source_codes = "".join(nib.aff2axcodes(vol.space.affine))

    no_op = reorient(vol, source_codes)
    roundtrip = reorient(reorient(vol, "RAS"), source_codes)

    np.testing.assert_array_equal(no_op.data, vol.data)
    np.testing.assert_allclose(no_op.space.affine, vol.space.affine)
    np.testing.assert_array_equal(roundtrip.data, vol.data)
    np.testing.assert_allclose(roundtrip.space.affine, vol.space.affine)


def test_space_only_and_compat_paths_share_the_volume_contract():
    vol = _volume()

    space_only = reorient(vol.space, ["R", "A", "S"])
    image = reorient(vol, "RAS")
    compat = as_canonical(vol)

    assert tuple(space_only.dim) == image.shape == (5, 3, 4)
    np.testing.assert_allclose(space_only.affine, image.space.affine)
    np.testing.assert_array_equal(compat.data, image.data)
    np.testing.assert_allclose(compat.space.affine, image.space.affine)


def test_space_only_preserves_nonspatial_dimensions():
    affine = AIR_AFFINE.copy()
    trans = np.eye(5)
    trans[:3, :3] = affine[:3, :3]
    trans[:3, 4] = affine[:3, 3]
    trans[3, 3] = 2.5
    space = ni.NeuroSpace(
        (3, 4, 5, 7),
        spacing=(3.0, 4.0, 2.0, 2.5),
        origin=(-10.0, -20.0, 30.0, 0.0),
        trans=trans,
    )

    observed = reorient(space, "RAS")

    assert tuple(observed.dim) == (5, 3, 4, 7)
    assert observed.spacing[3] == pytest.approx(2.5)
    assert observed.trans[3, 3] == pytest.approx(2.5)
    assert nib.aff2axcodes(observed.affine) == ("R", "A", "S")


def test_4d_neurovec_reorients_each_volume_without_touching_time():
    shape = (3, 4, 5, 3)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape, order="F")
    trans = np.eye(5)
    trans[:3, :3] = AIR_AFFINE[:3, :3]
    trans[:3, 4] = AIR_AFFINE[:3, 3]
    trans[3, 3] = 2.5
    vec = ni.DenseNeuroVec(data, ni.NeuroSpace(shape, trans=trans), label="bold")
    expected, expected_affine, _ = _nibabel_reorient(data, AIR_AFFINE, "RAS")

    observed = as_canonical(vec)

    assert isinstance(observed, ni.DenseNeuroVec)
    assert observed.shape == (5, 3, 4, 3)
    assert observed.label == "bold"
    np.testing.assert_array_equal(observed.data, expected)
    np.testing.assert_allclose(observed.space.affine, expected_affine)
    np.testing.assert_array_equal(observed.data[..., 1] - observed.data[..., 0], 60)
    assert observed.space.trans[3, 3] == pytest.approx(2.5)
