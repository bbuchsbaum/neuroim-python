"""Regression: SparseNeuroVol(arr, space, mask=full_mask).as_dense().data == arr.

The bug surfaced by the ME-7 round-trip work was an F/C-order disagreement
between ``SparseNeuroVol.__init__`` (which uses Fortran-order linear indices)
and the ravel() of the supplied 3-D array (which defaults to C-order),
causing voxel values to scramble during the sparse <-> dense round trip.
"""

import numpy as np

from neuroim import DenseNeuroVol, NeuroSpace, SparseNeuroVol


def test_sparse_dense_roundtrip_preserves_3d_arr():
    arr = np.arange(8, dtype=np.float64).reshape((2, 2, 2))
    space = NeuroSpace(dim=(2, 2, 2))
    mask = np.ones((2, 2, 2), dtype=bool)

    sparse_vol = SparseNeuroVol(arr, space, mask=mask)
    dense = sparse_vol.as_dense()

    assert isinstance(dense, DenseNeuroVol)
    np.testing.assert_array_equal(dense.data, arr)


def test_sparse_dense_roundtrip_non_cubic_shape():
    arr = np.arange(2 * 3 * 4, dtype=np.float64).reshape((2, 3, 4))
    space = NeuroSpace(dim=(2, 3, 4))
    mask = np.ones((2, 3, 4), dtype=bool)

    sparse_vol = SparseNeuroVol(arr, space, mask=mask)
    dense = sparse_vol.as_dense()

    np.testing.assert_array_equal(dense.data, arr)


def test_sparse_dense_roundtrip_preserves_per_voxel_value():
    """A spot-check that each individual voxel survives the round trip."""
    arr = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    space = NeuroSpace(dim=(2, 3, 4))
    mask = np.ones((2, 3, 4), dtype=bool)

    dense = SparseNeuroVol(arr, space, mask=mask).as_dense()

    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert dense.data[i, j, k] == arr[i, j, k], (
                    f"mismatch at ({i},{j},{k}): "
                    f"dense={dense.data[i, j, k]} arr={arr[i, j, k]}"
                )
