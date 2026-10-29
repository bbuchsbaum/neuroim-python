"""Resampling/spatial-transform contracts from the second correctness audit."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from neuroim import DenseNeuroVol, LogicalNeuroVol, NeuroSpace, SparseNeuroVol
from neuroim.clustered_neuro_vol import ClusteredNeuroVol
from neuroim.compat import add_dim, deoblique, output_aligned_space
from neuroim.neuro_vec import DenseNeuroVec
from neuroim.orientation import axcodes, vox2out_vox
from neuroim.resample import resample, resample_to, resample_vec

SRC = NeuroSpace((6, 6, 6), spacing=(2.0, 2.0, 2.0))
TGT = NeuroSpace((5, 5, 5), spacing=(2.4, 2.4, 2.4))


def test_sparse_and_logical_inputs_and_targets():
    data = np.random.default_rng(0).random((6, 6, 6))
    dense = DenseNeuroVol(data, SRC)
    sparse = dense.as_sparse()
    mask = LogicalNeuroVol(data > 0.5, SRC)
    expected = resample(dense, TGT, 1).data
    np.testing.assert_allclose(resample(sparse, TGT, 1).data, expected)
    assert resample(mask, TGT, 0).data.shape == (5, 5, 5)
    target_mask = LogicalNeuroVol(np.ones((5, 5, 5), bool), TGT)
    np.testing.assert_allclose(resample(dense, target_mask, 1).data, expected)
    target_sparse = SparseNeuroVol([1.0], TGT, indices=[0])
    np.testing.assert_allclose(resample(dense, target_sparse, 1).data, expected)


def test_resample_vec_keeps_source_time_axis():
    sp4 = NeuroSpace((6, 6, 6, 4), spacing=(2, 2, 2, 2.5))
    vec = DenseNeuroVec(np.random.default_rng(1).random((6, 6, 6, 4)), sp4)
    target = DenseNeuroVec(np.zeros((5, 5, 5, 9)), TGT.add_dim(size=9))
    for tgt in (target, TGT, DenseNeuroVol(np.zeros((5, 5, 5)), TGT)):
        out = resample_vec(vec, tgt, 1)
        assert out.shape == (5, 5, 5, 4)
        assert out.space.spacing[3] == pytest.approx(2.5)


def test_resample_to_defaults_to_nearest_for_labels():
    labels = np.zeros((6, 6, 6))
    labels[:3] = 1
    labels[3:] = 7
    out = resample_to(DenseNeuroVol(labels, SRC), TGT)
    assert set(np.unique(out.data)) <= {0.0, 1.0, 7.0}


def test_large_integer_labels_survive_and_label_map_is_pruned():
    big = np.full((6, 6, 6), 16777217.0)
    assert resample(DenseNeuroVol(big, SRC), SRC, 0).data[0, 0, 0] == 16777217.0
    m = np.ones((6, 6, 6), bool)
    clusters = np.ones(216, dtype=int)
    clusters[1] = 2  # voxel (1,0,0): between the coarse grid's sample points
    cv = ClusteredNeuroVol(
        LogicalNeuroVol(m, SRC), clusters, label_map={"big": 1, "tiny": 2}
    )
    out = resample(cv, NeuroSpace((2, 2, 2), spacing=(6, 6, 6)), 0)
    assert "tiny" not in out.label_map


def test_output_aligned_space_on_4d_and_2d():
    aff = np.array(
        [[1.9, 0.3, 0, -20], [-0.3, 1.9, 0, -15], [0, 0, 2, -10], [0, 0, 0, 1.0]]
    )
    sp3 = NeuroSpace((12, 10, 8), trans=aff)
    o3 = output_aligned_space(sp3, voxel_sizes=2.0)
    o4 = output_aligned_space(sp3.add_dim(size=3), voxel_sizes=2.0)
    np.testing.assert_allclose(o4.origin, o3.origin)
    a2 = np.eye(4)
    a2[:3, 3] = [5, 7, 9]
    np.testing.assert_allclose(output_aligned_space(((19, 28), a2)).origin, [5, 7])
    vec = DenseNeuroVec(np.zeros((12, 10, 8, 3)), sp3.add_dim(size=3))
    assert deoblique(vec).shape[3] == 3


def test_vox2out_vox_and_axcodes_4d():
    aff = np.zeros((4, 4))
    aff[3, 3] = 1
    aff[1, 0] = aff[2, 1] = aff[0, 2] = 1  # i->A, j->S, k->R
    _, shape = vox2out_vox(aff, (10, 20, 30))
    canonical = nib.as_closest_canonical(nib.Nifti1Image(np.zeros((10, 20, 30)), aff))
    assert shape == canonical.shape
    sp4 = NeuroSpace((4, 4, 4), trans=aff).add_dim(size=2)
    assert axcodes(sp4) == axcodes(NeuroSpace((4, 4, 4), trans=aff))


def test_compat_add_dim_adds_one_axis_of_length_n():
    assert tuple(add_dim(NeuroSpace((4, 5, 6)), 10).dim) == (4, 5, 6, 10)


def test_space_hash_matches_equality():
    a = NeuroSpace((2, 2, 2), trans=np.diag([1.0, 1.0, 1.0, 1.0]))
    t = np.diag([1.0, 1.0, 1.0, 1.0])
    t[0, 1] = -0.0
    b = NeuroSpace((2, 2, 2), trans=t)
    assert a == b and len({a, b}) == 1


def test_atlas_rejects_fractional_labels():
    from neuroim.atlas import schaefer_200

    lab = np.zeros((10, 10, 10), np.float32)
    lab[:5] = 1
    lab[5:] = 7
    sp = NeuroSpace((10, 10, 10), spacing=(2, 2, 2))
    shifted = NeuroSpace((10, 10, 10), spacing=(2, 2, 2), origin=(0.6, 0, 0))
    linear = resample_to(DenseNeuroVol(lab, sp), shifted, "linear")
    with pytest.raises(ValueError, match="integer"):
        schaefer_200(linear)
