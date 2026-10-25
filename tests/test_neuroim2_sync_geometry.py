"""Regression contracts carried over from neuroim2 fixes (Aug-Oct 2026).

Each test pins a defect that neuroim2 fixed (or that a neuroim2 audit found
in this package): voxel spacing from the transform actually in use, affine
preservation in file-backed vectors, 4-D world coordinates, anatomical axis
labels, ROI world coordinates, downsample alignment, sparse masks and
orientation, and backend matrix layouts.
"""

from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim import NeuroSpace
from neuroim.axis import INF_SUP, LEFT_RIGHT, POST_ANT, RIGHT_LEFT
from neuroim.big_neuro_vec import BigNeuroVec
from neuroim.file_backed_neuro_vec import FileBackedNeuroVec
from neuroim.neuro_hypervec import SparseNeuroHyperVec
from neuroim.neuro_vec import DenseNeuroVec, SparseNeuroVec
from neuroim.operations import downsample
from neuroim.resample import reorient
from neuroim.roi import ROIVol, spherical_roi

LAS = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _save(img, tmp_path, name):
    path = tmp_path / name
    nib.save(img, str(path))
    return str(path)


# --- spacing follows the chosen transform (neuroim2 916866f) ---------------


def test_spacing_comes_from_sform_not_pixdim(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((4.0, 4.0, 4.0))  # pixdim disagrees with the sform
    img.set_qform(np.diag([4.0, 4.0, 4.0, 1.0]), code=1)
    img.set_sform(np.eye(4), code=1)
    vol = ni.read_vol(_save(img, tmp_path, "sform.nii"))

    np.testing.assert_allclose(vol.space.spacing, [1.0, 1.0, 1.0])
    roi = spherical_roi(vol.space, (10, 10, 10), 3.0)
    assert len(roi) > 27  # 1 mm voxels: a 3 mm sphere spans many voxels


def test_4d_spacing_keeps_time_zoom(tmp_path):
    img = nib.Nifti1Image(np.zeros((4, 4, 4, 3), dtype=np.float32), LAS)
    img.header.set_zooms((2.0, 2.0, 2.0, 1.5))
    vec = ni.read_vec(_save(img, tmp_path, "tr.nii"))
    np.testing.assert_allclose(vec.space.spacing, [2.0, 2.0, 2.0, 1.5])


# --- file-backed vectors keep the full affine ------------------------------


def test_file_backed_vec_keeps_affine(tmp_path):
    files = []
    for i in range(2):
        img = nib.Nifti1Image(np.full((3, 4, 5), i, dtype=np.float32), LAS)
        files.append(_save(img, tmp_path, f"v{i}.nii"))
    fb = FileBackedNeuroVec(files)
    np.testing.assert_allclose(fb.space.affine, LAS)


# --- 4-D world coordinates honour rotations and flips (neuroim2 21ac77c) ----


def test_4d_coords_use_full_affine():
    sp3 = NeuroSpace((5, 6, 7), trans=LAS)
    sp4 = sp3.add_dim(size=3)
    grid = np.array([[0, 0, 0], [4, 5, 6], [1, 2, 3]])
    expected = nib.affines.apply_affine(LAS, grid)

    np.testing.assert_allclose(sp4.grid_to_coord(grid), expected)
    np.testing.assert_allclose(
        sp4.grid_to_coord(np.column_stack([grid, [0, 1, 2]]))[:, :3], expected
    )
    idx = sp3.grid_to_index(grid)
    np.testing.assert_allclose(sp4.index_to_coord(idx)[:, :3], expected)
    np.testing.assert_array_equal(sp4.coord_to_index(expected), idx)


# --- anatomical axis labels (neuroim2 21ac77c) ------------------------------


def test_axes_follow_affine():
    ras = NeuroSpace((4, 4, 4), trans=np.diag([2.0, 2.0, 2.0, 1.0]))
    assert (ras.axes.i, ras.axes.j, ras.axes.k) == (LEFT_RIGHT, POST_ANT, INF_SUP)
    las = NeuroSpace((4, 4, 4), trans=LAS)
    assert (las.axes.i, las.axes.j, las.axes.k) == (RIGHT_LEFT, POST_ANT, INF_SUP)


def test_reorient_axes_match_requested_codes():
    sp = NeuroSpace((4, 5, 6), trans=LAS)
    out = reorient(sp, "RAS")
    assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
    assert (out.axes.i, out.axes.j, out.axes.k) == (LEFT_RIGHT, POST_ANT, INF_SUP)


# --- ROI world coordinates have no half-voxel shift -------------------------


def test_roi_real_coords_match_affine():
    sp = NeuroSpace((10, 10, 10), trans=LAS)
    coords = np.array([[0, 0, 0], [3, 4, 5]])
    roi = ROIVol(np.ones(2), sp, coords)
    np.testing.assert_allclose(
        roi.get_coords(real=True), nib.affines.apply_affine(LAS, coords)
    )


# --- downsample keeps data and affine aligned -------------------------------


@pytest.mark.parametrize(
    "shape,kwargs",
    [
        ((12, 12, 12), {"factor": 0.5}),
        ((11, 11, 11), {"factor": 2}),
        ((10, 10, 10), {"factor": 0.3}),
        ((12, 12, 12), {"spacing": 5}),
    ],
)
def test_downsample_values_match_affine(shape, kwargs):
    sp = NeuroSpace(shape, spacing=(2.0, 2.0, 2.0))
    world_x = sp.grid_to_coord(np.indices(shape).reshape(3, -1).T)[:, 0]
    vol = ni.DenseNeuroVol(world_x.reshape(shape), sp)
    out = downsample(vol, **kwargs)
    out_shape = out.shape
    grid = np.indices(out_shape).reshape(3, -1).T
    out_x = out.space.grid_to_coord(grid)[:, 0]
    # Piecewise-constant voxels make a box mean of a ramp exact only up to a
    # small fraction of a voxel at partially covered edges.
    assert np.max(np.abs(out.data.reshape(-1) - out_x)) < 0.25


def test_downsample_spacing_rounds_like_neuroim2():
    sp = NeuroSpace((12, 12, 12), spacing=(2.0, 2.0, 2.0))
    out = downsample(ni.DenseNeuroVol(np.zeros((12, 12, 12)), sp), spacing=5)
    assert out.shape == (5, 5, 5)


def test_downsample_integer_blocks_are_exact_means():
    data = np.arange(64, dtype=float).reshape(4, 4, 4)
    vol = ni.DenseNeuroVol(data, NeuroSpace((4, 4, 4)))
    out = downsample(vol, factor=0.5)
    np.testing.assert_allclose(out.data[0, 0, 0], data[:2, :2, :2].mean())


# --- sparse masks (neuroim2 badd3db) ----------------------------------------


def _dense(nt=2):
    sp = NeuroSpace((3, 3, 1, nt))
    data = np.arange(9 * nt, dtype=float).reshape((3, 3, 1, nt), order="F") + 1
    return DenseNeuroVec(data, sp)


def test_integer_spatial_mask_is_logical():
    v = _dense()
    m = np.zeros((3, 3, 1), dtype=int)
    m[2, 1, 0] = 1
    m[0, 2, 0] = 1
    s = SparseNeuroVec(v.data[m.astype(bool)].T, v.space, m)
    np.testing.assert_array_equal(s.mask.data, m.astype(bool))


def test_unsorted_indices_keep_values_at_their_voxels():
    v = _dense()
    flat = v.data.reshape(-1, 2, order="F")
    idx = np.array([7, 0, 3])
    s = SparseNeuroVec(flat[idx].T, v.space, idx)
    np.testing.assert_allclose(s.to_dense().data, v.data * s.mask.data[..., None])


@pytest.mark.parametrize("bad", [[-1], [9], [1, 1], [0.5]])
def test_bad_indices_rejected(bad):
    v = _dense()
    with pytest.raises(ValueError):
        SparseNeuroVec(np.zeros((2, len(bad))), v.space, np.array(bad))


def test_to_sparse_accepts_indices():
    v = _dense()
    s = v.to_sparse(np.array([7, 0, 7]))
    assert np.flatnonzero(s.mask.data.ravel(order="F")).tolist() == [0, 7]
    np.testing.assert_allclose(s.series_at_indices([7]).ravel(), v.data[1, 2, 0])


# --- square sparse orientation (neuroim2 #31) -------------------------------


def test_square_sparse_warns_and_explicit_orientation_wins():
    v = _dense(nt=3)
    idx = np.array([0, 4, 8])
    vox_x_time = v.data.reshape(-1, 3, order="F")[idx]
    with pytest.warns(UserWarning, match="square"):
        SparseNeuroVec(vox_x_time.T, v.space, idx)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = SparseNeuroVec(vox_x_time, v.space, idx, orientation="voxels_x_time")
    np.testing.assert_allclose(s.data, vox_x_time.T)


def test_hypervec_producers_square():
    n = 3
    sp5 = NeuroSpace((3, 1, 1, n, 2))
    mask = ni.LogicalNeuroVol(np.ones((3, 1, 1), dtype=bool), NeuroSpace((3, 1, 1)))
    data = np.arange(2 * n * 3, dtype=float).reshape(2, n, 3)  # feat, time, vox
    hv = SparseNeuroHyperVec(data, mask, sp5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(hv.mean_features().data, data.mean(axis=0))
        np.testing.assert_allclose(hv.apply_feature_func(np.max).data, data.max(axis=0))


def test_hypervec_as_dense_voxel_order():
    sp5 = NeuroSpace((2, 2, 1, 1, 1))
    m = np.zeros((2, 2, 1), dtype=bool)
    m[1, 0, 0] = m[0, 1, 0] = True
    mask = ni.LogicalNeuroVol(m, NeuroSpace((2, 2, 1)))
    hv = SparseNeuroHyperVec(np.array([[[1.0, 2.0]]]), mask, sp5)
    dense = hv.as_dense().data
    assert dense[1, 0, 0, 0, 0] == 1.0  # F-order: (1,0,0) precedes (0,1,0)
    assert dense[0, 1, 0, 0, 0] == 2.0


# --- backend matrix layouts are voxels x time -------------------------------


def test_file_backed_as_matrix_is_voxels_by_time(tmp_path):
    files = []
    for i in range(3):
        img = nib.Nifti1Image(np.full((2, 2, 2), i, dtype=np.float32), np.eye(4))
        files.append(_save(img, tmp_path, f"m{i}.nii"))
    fb = FileBackedNeuroVec(files)
    np.testing.assert_allclose(fb.as_matrix(), fb.to_dense().as_matrix())
    assert fb.as_matrix().shape == (8, 3)


def test_big_neuro_vec_matrix_and_sparse():
    data = np.random.default_rng(0).normal(size=(3, 2, 2, 4))
    sp = NeuroSpace((3, 2, 2, 4))
    big = BigNeuroVec(data=data, space=sp)
    dense = DenseNeuroVec(data, sp)
    np.testing.assert_allclose(big.as_matrix(), dense.as_matrix())
    np.testing.assert_allclose(big.to_sparse().to_dense().data, data)
