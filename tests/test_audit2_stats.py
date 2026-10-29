"""Statistics/reduction contracts from the second correctness audit."""

from __future__ import annotations

import numpy as np
import pytest

from neuroim import DenseNeuroVol, LogicalNeuroVol, NeuroSpace, SparseNeuroVol
from neuroim.clustered_neuro_vol import ClusteredNeuroVol
from neuroim.indexing import linear_access
from neuroim.neuro_vec import DenseNeuroVec
from neuroim.roi import ROIVec
from neuroim.stats import (
    map_values,
    partition,
    split_blocks,
    split_clusters,
    split_volumes,
    split_scale,
)


def _vec(shape=(8, 3, 2, 6), dtype=float, seed=0):
    data = np.random.default_rng(seed).normal(size=shape) * 10
    return DenseNeuroVec(data.astype(dtype), NeuroSpace(shape))


def test_split_scale_integer_input_is_not_truncated():
    vec = _vec(dtype=np.int32)
    fac = np.array([1, 1, 1, 2, 2, 2])
    out = split_scale(vec, fac)
    assert out.data.dtype == np.float64
    level = vec.data[0, 0, 0, :3].astype(float)
    expected = (level - level.mean()) / level.std(ddof=1)
    np.testing.assert_allclose(out.data[0, 0, 0, :3], expected)


def test_split_volumes_selects_time_points():
    vec = _vec()
    fac = np.array([1, 2, 1, 2, 2, 1])
    out = split_volumes(vec, fac)
    np.testing.assert_array_equal(out[1].data, vec.data[..., fac == 1])
    assert out[2].shape == (8, 3, 2, 3)


def test_split_blocks_keeps_series_with_their_voxels():
    vec = _vec()
    blocks = split_blocks(vec, np.array([7, 0, 5]), np.array([1, 1, 1]))
    blk = blocks[0]
    for idx in (0, 5, 7):
        grid = np.unravel_index(idx, vec.shape[:3], order="F")
        np.testing.assert_allclose(blk.series_at_indices([idx]).ravel(), vec.data[grid])


def test_split_blocks_sparse_input_and_mask():
    vec = _vec()
    m = np.zeros(vec.shape[:3], dtype=bool)
    m[:2] = True
    sparse = vec.to_sparse(np.ones(vec.shape[:3], dtype=bool))
    blocks = split_blocks(
        sparse, nblocks=2, mask=LogicalNeuroVol(m, NeuroSpace(m.shape))
    )
    assert sum(int(b.mask.data.sum()) for b in blocks) == int(m.sum())


def test_split_clusters_on_a_vector_returns_series():
    vec = _vec((3, 3, 3, 5))
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0] = 1
    labels[2] = 2
    rois = split_clusters(vec, DenseNeuroVol(labels, NeuroSpace((3, 3, 3))))
    assert [type(r) for r in rois] == [ROIVec, ROIVec]
    roi = rois[0]
    assert roi.data.shape == (5, 9)
    np.testing.assert_array_equal(roi.data[:, 0], vec.data[tuple(roi.coords[0])])


def test_partition_is_spatial():
    m = np.ones((6, 6, 6), dtype=bool)
    cv = partition(LogicalNeuroVol(m, NeuroSpace(m.shape)), 4)
    sizes = np.bincount(np.asarray(cv.clusters))[1:]
    assert len(sizes) == 4 and sizes.min() > 30  # four compact parcels


def test_map_values_unmapped_to_zero_and_float():
    sp = NeuroSpace((2, 2, 2))
    vol = DenseNeuroVol(np.array([0, 1, 2, 3, 1, 2, 0, 3]).reshape(2, 2, 2), sp)
    out = map_values(vol, {1: 0.5, 2: 20})
    assert out.data.dtype == np.float64
    assert sorted(set(out.values().tolist())) == [0.0, 0.5, 20.0]
    sparse = SparseNeuroVol([1.0], sp, indices=[3])
    assert map_values(sparse, {0: -1.0}).values()[0] == -1.0


def test_linear_access_scalar_and_negative():
    vec = _vec((2, 2, 2, 3))
    np.testing.assert_array_equal(
        linear_access(vec, 3), vec.series_at_indices([3]).ravel()
    )
    with pytest.raises(ValueError):
        linear_access(vec, np.array([-1]))


def test_connectome_receipts_record_metric():
    labels = np.array([1, 1, 2, 2, 1, 2, 1, 2])
    sp = NeuroSpace((2, 2, 2))
    cvol = ClusteredNeuroVol(LogicalNeuroVol(np.ones((2, 2, 2), bool), sp), labels)
    vec = _vec((2, 2, 2, 10))
    cv = vec.parcel_means(cvol)
    corr = cv.connectome(metric="correlation").provenance
    cov = cv.connectome(metric="covariance").provenance
    assert corr.method_name != cov.method_name
