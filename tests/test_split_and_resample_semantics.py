"""split_fill / split_reduce follow neuroim2; split_* work on every vector
backend; a volume resamples onto a 4-D target's spatial grid."""

import nibabel as nib
import numpy as np
import pytest

from neuroim import DenseNeuroVol, NeuroSpace
from neuroim.big_neuro_vec import BigNeuroVec
from neuroim.file_backed_neuro_vec import FileBackedNeuroVec
from neuroim.mapped_neuro_vec import MappedNeuroVec
from neuroim.neuro_vec import DenseNeuroVec
from neuroim.resample import resample, resample_to
from neuroim.stats import split_fill, split_reduce, split_scale, split_volumes

SHAPE = (3, 4, 5)


def _vol():
    data = np.random.default_rng(0).normal(size=SHAPE)
    return DenseNeuroVol(data, NeuroSpace(SHAPE))


# --- split_fill (neuroim2: fill each voxel set with FUN's result) -------------


def test_split_fill_scalar_fun_fills_each_set():
    vol = _vol()
    fac = np.arange(60) % 6  # flat, linear (Fortran) voxel order
    out = split_fill(vol, fac, np.mean)
    flat = vol.values()
    expected = np.array([flat[fac == lev].mean() for lev in range(6)])[fac]
    np.testing.assert_allclose(out.values(), expected)
    assert out.space == vol.space


def test_split_fill_vector_fun_and_3d_factor():
    vol = _vol()
    fac3d = (np.arange(60) % 3).reshape(SHAPE, order="F")
    out = split_fill(vol, fac3d, lambda v: v[::-1])  # R example: rev
    flat, fac = vol.values(), fac3d.ravel(order="F")
    for lev in range(3):
        np.testing.assert_allclose(out.values()[fac == lev], flat[fac == lev][::-1])


def test_split_fill_errors():
    vol = _vol()
    with pytest.raises(ValueError, match="one entry per voxel"):
        split_fill(vol, np.zeros(7), np.mean)
    with pytest.raises(ValueError, match="FUN returned"):
        split_fill(vol, np.zeros(60), lambda v: v[:2])


# --- split_reduce with a per-voxel factor (parcel time series) ----------------


def test_split_reduce_per_voxel_factor_gives_level_time_series():
    data = np.random.default_rng(1).normal(size=SHAPE + (7,))
    vec = DenseNeuroVec(data, NeuroSpace(SHAPE + (7,)))
    fac = np.arange(60) % 4
    out = split_reduce(vec, fac)
    mat = data.reshape(60, 7, order="F")
    assert out.shape == (4, 7)
    for lev in range(4):
        np.testing.assert_allclose(out[lev], mat[fac == lev].mean(axis=0))
    np.testing.assert_allclose(
        split_reduce(vec, fac.reshape(SHAPE, order="F"), np.median)[2],
        np.median(mat[fac == 2], axis=0),
    )


# --- every vector backend ------------------------------------------------------


@pytest.fixture
def backends(tmp_path):
    rng = np.random.default_rng(2)
    data = rng.normal(size=SHAPE + (6,)).astype(np.float32)
    space = NeuroSpace(SHAPE + (6,))
    dense = DenseNeuroVec(data.astype(np.float64), space)
    files = []
    for t in range(6):
        f = str(tmp_path / f"v{t}.nii")
        nib.save(nib.Nifti1Image(data[..., t], np.eye(4)), f)
        files.append(f)
    return dense, {
        "big": BigNeuroVec(data, space),
        "file_backed": FileBackedNeuroVec(files),
        "mapped": MappedNeuroVec(dense, lambda x: x),
    }


@pytest.mark.parametrize("name", ["big", "file_backed", "mapped"])
def test_split_functions_on_non_dense_backends(backends, name):
    dense, vecs = backends
    vec = vecs[name]
    fac = np.array([1, 1, 2, 2, 2, 3])

    parts = split_volumes(vec, fac)
    for lev, part in parts.items():
        np.testing.assert_allclose(
            np.asarray(part.to_dense().data), dense.data[..., fac == lev], rtol=1e-6
        )
    np.testing.assert_allclose(
        split_reduce(vec, fac, np.max), split_reduce(dense, fac, np.max), rtol=1e-6
    )
    np.testing.assert_allclose(
        split_scale(vec, fac).data, split_scale(dense, fac).data, rtol=1e-4, atol=1e-5
    )


# --- resample onto a 4-D target -----------------------------------------------


def test_volume_resamples_onto_spatial_grid_of_4d_target():
    src = DenseNeuroVol(
        np.random.default_rng(3).random((6, 6, 6)), NeuroSpace((6, 6, 6), (2, 2, 2))
    )
    grid3 = NeuroSpace((5, 5, 5), (2.4, 2.4, 2.4), origin=(1, 2, 3))
    run = DenseNeuroVec(np.zeros((5, 5, 5, 4)), grid3.add_dim(size=4))
    expected = resample(src, grid3, 1).data

    for target in (run, run.space):
        out = resample_to(src, target, "linear")
        assert out.shape == (5, 5, 5)
        np.testing.assert_allclose(out.data, expected)
        np.testing.assert_allclose(out.space.affine, grid3.affine)
