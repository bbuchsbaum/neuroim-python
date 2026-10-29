"""read_vec(..., mask=<array>) builds a 3-D mask space from the 4-D file."""

import nibabel as nib
import numpy as np

import neuroim as ni
from neuroim.neuro_vec import SparseNeuroVec


def test_read_vec_with_array_mask(tmp_path):
    data = np.random.default_rng(0).random((4, 5, 6, 3)).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-4, -5, -6]
    path = str(tmp_path / "v.nii")
    nib.save(nib.Nifti1Image(data, affine), path)
    mask = np.zeros((4, 5, 6), dtype=bool)
    mask[1:3, 2:4, :] = True

    vec = ni.read_vec(path, mask=mask)

    assert isinstance(vec, SparseNeuroVec)
    dense = ni.read_vec(path)
    np.testing.assert_allclose(vec.to_dense().data[mask], dense.data[mask])
    np.testing.assert_allclose(vec.space.affine, affine)
