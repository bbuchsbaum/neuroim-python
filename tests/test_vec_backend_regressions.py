"""Vector-backend defects surfaced while type-checking."""

import nibabel as nib
import numpy as np
import pytest

from neuroim import NeuroSpace
from neuroim.file_backed_neuro_vec import FileBackedNeuroVec
from neuroim.mapped_neuro_vec import MappedNeuroVec
from neuroim.neuro_vec import DenseNeuroVec


def _vec():
    data = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    return DenseNeuroVec(data, NeuroSpace(data.shape))


def test_series_rejects_badly_shaped_array_instead_of_returning_none():
    vec = _vec()
    for v in (vec, vec.to_sparse(np.ones((2, 3, 4), dtype=bool))):
        with pytest.raises(ValueError, match="Invalid input"):
            v.series(np.zeros((2, 2, 2), dtype=int))


def test_mapped_vec_as_matrix_over_sparse_source():
    vec = _vec()
    sparse = vec.to_sparse(np.ones((2, 3, 4), dtype=bool))
    mapped = MappedNeuroVec(sparse, lambda x: x * 0.5)
    np.testing.assert_allclose(mapped.as_matrix(), vec.as_matrix() * 0.5)


def test_file_backed_sub_vector_accepts_an_int(tmp_path):
    vols = np.random.default_rng(0).random((3, 4, 5, 3))
    files = []
    for t in range(3):
        f = str(tmp_path / f"v{t}.nii")
        nib.save(nib.Nifti1Image(vols[..., t], np.eye(4)), f)
        files.append(f)
    sub = FileBackedNeuroVec(files).sub_vector(1)
    assert sub.shape[3] == 1
    np.testing.assert_allclose(np.asarray(sub.data)[..., 0], vols[..., 1])
