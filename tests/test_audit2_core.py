"""Core container contracts from the second correctness audit.

Each test pins a defect that silently produced wrong values: integer
overflow in arithmetic, lost or misdirected assignments, order mismatches
between mask selection and linear indices, matrix-layout swaps, sparse
arithmetic and indexing, scaling denominators, and index validation.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim import DenseNeuroVol, LogicalNeuroVol, NeuroSpace, SparseNeuroVol
from neuroim.index_lookup_vol import IndexLookupVol
from neuroim.mapped_neuro_vec import MappedNeuroVec
from neuroim.neuro_vec import DenseNeuroVec

SHAPE = (4, 5, 3)


def _vol(order="C"):
    data = np.asarray(np.arange(60, dtype=float).reshape(SHAPE), order=order)
    return DenseNeuroVol(data, NeuroSpace(SHAPE))


# --- arithmetic --------------------------------------------------------------


def test_integer_images_do_not_overflow(tmp_path):
    img = nib.Nifti1Image(np.full((2, 2, 2), 200, dtype=np.uint8), np.eye(4))
    nib.save(img, str(tmp_path / "u8.nii"))
    v = ni.read_vol(str(tmp_path / "u8.nii"))
    assert v.data.dtype == np.uint8
    assert (v + v).data[0, 0, 0] == 400
    assert (v - (v + 1)).data[0, 0, 0] == -1
    vec = DenseNeuroVec(
        np.full((2, 2, 2, 2), 30000, dtype=np.int16), NeuroSpace((2, 2, 2, 2))
    )
    assert (vec + vec).data[0, 0, 0, 0] == 60000


def test_numpy_scalars_in_arithmetic():
    v = _vol()
    assert (v + np.int64(2)).data[0, 0, 0] == 2
    assert int((v > np.int64(57)).data.sum()) == 2


def test_sparse_plus_scalar_touches_every_voxel():
    sp = NeuroSpace((3, 3, 3))
    s = SparseNeuroVol([1.0, 2.0], sp, indices=[0, 5])
    assert (s + 5).values().sum() == pytest.approx(3 + 5 * 27)
    assert ((10 - s).values() == 10).sum() == 25
    doubled = s * 2
    assert isinstance(doubled, SparseNeuroVol)  # 0 * 2 == 0 stays sparse
    np.testing.assert_allclose(doubled.values()[[0, 5]], [2.0, 4.0])


def test_vec_plus_sparse_vol_uses_dense_values():
    sp4 = NeuroSpace((2, 2, 2, 3))
    vec = DenseNeuroVec(np.zeros((2, 2, 2, 3)), sp4)
    s = SparseNeuroVol([7.0], NeuroSpace((2, 2, 2)), indices=[3])
    out = (vec + s).data
    assert int((out != 0).sum()) == 3  # one voxel x 3 time points


# --- dense volume indexing ---------------------------------------------------


@pytest.mark.parametrize("order", ["C", "F"])
def test_linear_assignment_reaches_the_volume(order):
    v = _vol(order)
    v[7] = -2
    v[np.array([8])] = -3
    v[0:2] = -4
    flat = v.values()
    assert flat[7] == -2 and flat[8] == -3 and list(flat[:2]) == [-4, -4]


def test_list_key_reads_and_writes_the_same_voxels():
    v = _vol()
    v[[1, 2]] = -1
    assert int((v.data == -1).sum()) == 2
    np.testing.assert_array_equal(v[[1, 2]], [-1, -1])


def test_mask_selection_is_fortran_order():
    v = _vol()
    m = v.data > 30
    flat_idx = np.flatnonzero(m.ravel(order="F"))
    np.testing.assert_array_equal(v[m], v.values()[flat_idx])
    np.testing.assert_array_equal(
        v[LogicalNeuroVol(m, v.space)], v.as_sparse(LogicalNeuroVol(m, v.space)).data
    )
    w = DenseNeuroVol(np.zeros(SHAPE), v.space)
    w[m] = v[m]
    np.testing.assert_array_equal(w.data[m], v.data[m])


def test_numpy_integer_key_is_scalar_and_coords_pair_with_values():
    v = _vol()
    assert np.ndim(v[np.int64(5)]) == 0 and v[np.int64(5)] == v[5]
    coords = v.coords()
    np.testing.assert_array_equal(coords[1], v.space.index_to_grid(1)[0])


def test_which_max_ignores_nan_and_sd_uses_n_minus_1():
    v = _vol()
    v.data[0, 0, 0] = np.nan
    assert v.which_max() == int(np.nanargmax(v.values()))
    assert v.which_min() == int(np.nanargmin(v.values()))
    w = _vol()
    assert w.vol_sd() == pytest.approx(np.std(w.values(), ddof=1))


# --- vector construction and comparison --------------------------------------


def test_dense_vec_from_matrix_round_trips():
    sp = NeuroSpace((2, 3, 4, 5))
    vec = DenseNeuroVec(np.random.default_rng(0).random((2, 3, 4, 5)), sp)
    np.testing.assert_array_equal(DenseNeuroVec(vec.as_matrix(), sp).data, vec.data)
    np.testing.assert_array_equal(DenseNeuroVec(vec.as_matrix().T, sp).data, vec.data)


def test_only_single_time_selection_is_a_volume():
    vec = DenseNeuroVec(np.zeros((2, 3, 4, 5)), NeuroSpace((2, 3, 4, 5)))
    assert isinstance(vec[..., 1], DenseNeuroVol)
    assert not isinstance(vec[0], DenseNeuroVol)


def test_vec_comparisons_are_elementwise():
    sp = NeuroSpace((2, 2, 2, 3))
    d = np.arange(24.0).reshape(2, 2, 2, 3)
    v, w = DenseNeuroVec(d, sp), DenseNeuroVec(d.copy(), sp)
    assert bool(np.all((v == w).data))
    assert int((v > 10).data.sum()) == 13


def test_scale_series_matches_neuroim2():
    d = np.random.default_rng(1).normal(size=(2, 2, 2, 8)) + 3
    vec = DenseNeuroVec(d, NeuroSpace(d.shape))
    z = vec.scale_series(center=True, scale=True).data
    np.testing.assert_allclose(np.std(z, axis=3, ddof=1), 1.0)
    s = vec.scale_series(center=False, scale=True).data
    np.testing.assert_allclose(np.sqrt((s**2).sum(axis=3) / 7), 1.0)


# --- index validation and sparse volumes -------------------------------------


@pytest.mark.parametrize("bad", [[-1], [2.7], [60]])
def test_constructor_indices_validated(bad):
    sp = NeuroSpace(SHAPE)
    with pytest.raises(ValueError):
        DenseNeuroVol([1.0], sp, indices=bad)
    with pytest.raises(ValueError):
        LogicalNeuroVol(None, sp, indices=bad)


def test_sparse_vol_rejects_duplicates_and_indexes_like_dense():
    sp = NeuroSpace(SHAPE)
    with pytest.raises(ValueError):
        SparseNeuroVol([1.0, 2.0], sp, indices=[4, 4])
    s = SparseNeuroVol([1.0, 9.0], sp, indices=[0, 59])
    flat = np.zeros(60, dtype=bool)
    flat[59] = True
    s[flat] = 5.0
    assert s.values()[59] == 5.0
    assert isinstance(s[0:3], np.ndarray)


def test_index_lookup_rejects_negative_and_fractional():
    sp = NeuroSpace(SHAPE)
    lut = IndexLookupVol(sp, np.array([0, 59]))
    with pytest.raises(KeyError):
        lut.grid_to_table(np.array([-1]))
    with pytest.raises(ValueError):
        IndexLookupVol(sp, np.array([2.7]))


def test_mapped_vec_over_sparse_source():
    sp4 = NeuroSpace((2, 2, 2, 3))
    vec = DenseNeuroVec(np.arange(24, dtype=np.int64).reshape(2, 2, 2, 3), sp4)
    sparse = vec.to_sparse(np.ones((2, 2, 2), dtype=bool))
    mapped = MappedNeuroVec(sparse, lambda x: x * 0.5)
    np.testing.assert_allclose(mapped.data, vec.data * 0.5)
    assert mapped.dtype == np.float64
