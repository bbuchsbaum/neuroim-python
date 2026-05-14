import numpy as np

import neuroim as ni
from neuroim.file_backed_neuro_vec import FileBackedNeuroVec
from neuroim.storage import NeuroVecStoreAdapter, VoxelSeriesStore


def test_neurovec_store_adapter_delegates_series_and_matrix():
    space = ni.NeuroSpace((3, 4, 2, 5))
    data = np.arange(np.prod(space.dim)).reshape(tuple(space.dim), order="F")
    vec = ni.DenseNeuroVec(data, space)

    store = vec.store

    assert isinstance(store, NeuroVecStoreAdapter)
    assert isinstance(store, VoxelSeriesStore)
    assert store.shape == vec.shape
    assert store.dtype == vec.dtype
    np.testing.assert_array_equal(
        store.series(np.array([[0, 0, 0]])), vec.series(np.array([[0, 0, 0]]))
    )
    np.testing.assert_array_equal(store.as_matrix(), vec.as_matrix())


def test_sparse_dtype_and_oob_series_contract():
    space = ni.NeuroSpace((3, 3, 2, 4))
    data = np.arange(np.prod(space.dim), dtype=np.float32).reshape(
        tuple(space.dim), order="F"
    )
    dense = ni.DenseNeuroVec(data, space)
    sparse = dense.as_sparse(np.ones(tuple(space.dim[:3]), dtype=bool))

    assert sparse.dtype == np.dtype(np.float32)

    result = sparse.series(np.array([[0, 0, 0], [99, 99, 99]], dtype=int))
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result[:, 0], data[0, 0, 0, :])
    np.testing.assert_array_equal(result[:, 1], np.zeros(4))


def test_file_backed_store_series_does_not_materialize_full_data(tmp_path, monkeypatch):
    import nibabel as nib

    files = []
    affine = np.eye(4)
    for i in range(3):
        data = np.full((3, 3, 2), i, dtype=np.float32)
        path = tmp_path / f"vol_{i}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        files.append(str(path))

    vec = FileBackedNeuroVec(files)

    def fail_full_data(_self):
        raise AssertionError("full data materialization was called")

    monkeypatch.setattr(FileBackedNeuroVec, "data", property(fail_full_data))

    series = vec.store.series(np.array([[1, 1, 1]], dtype=int))
    assert series.shape == (3, 1)
    np.testing.assert_array_equal(series[:, 0], [0, 1, 2])
