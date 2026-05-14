"""Backend dtype round-trip parity invariant for write/read cycles.

ME-7 / bd-01KRKJVA45WRCNWNYME5HDWE06: a permanent regression gate against
the dtype-drift class of bugs that produced post-01KRKJC0YP Finding 3
(FileBackedNeuroVec float64-vs-float32 silently flipped after read).  The
invariant we promise: when a caller writes a neuroimaging container to
disk and reads it back through neuroim's I/O entry points, the dtype of
the recovered array matches the dtype of the original.

The test parametrizes over:
- Volume backends: DenseNeuroVol, SparseNeuroVol
- Vector backends: DenseNeuroVec, SparseNeuroVec, MappedNeuroVec,
  FileBackedNeuroVec
- Numeric dtypes representing the realistic working set: float32, float64,
  int16, int32, uint8

Sparse and Mapped backends materialize to dense on write; their dtype
contract is the input dtype, not the storage backend's internal dtype.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from neuroim import (
    DenseNeuroVec,
    DenseNeuroVol,
    FileBackedNeuroVec,
    LogicalNeuroVol,
    MappedNeuroVec,
    NeuroSpace,
    SparseNeuroVec,
    SparseNeuroVol,
)
from neuroim.io import read_vec, read_vol, write_vec, write_vol


_VOL_SHAPE: Tuple[int, int, int] = (6, 6, 6)
_VEC_SHAPE: Tuple[int, int, int, int] = (6, 6, 6, 4)
_DTYPES = ("float32", "float64", "int16", "int32", "uint8")

_NUMPY_TO_NIFTI = {
    np.dtype("float32"): "FLOAT32",
    np.dtype("float64"): "FLOAT64",
    np.dtype("int16"): "INT16",
    np.dtype("int32"): "INT32",
    np.dtype("uint8"): "UINT8",
}


def _to_nifti_dtype(dtype: np.dtype) -> str:
    try:
        return _NUMPY_TO_NIFTI[np.dtype(dtype)]
    except KeyError as exc:
        raise AssertionError(
            f"Test fixture extended to dtype {dtype} without a NIfTI mapping"
        ) from exc


def _vol_space() -> NeuroSpace:
    return NeuroSpace(dim=_VOL_SHAPE, spacing=(2.0, 2.0, 2.0))


def _vec_space() -> NeuroSpace:
    return NeuroSpace(dim=_VEC_SHAPE, spacing=(2.0, 2.0, 2.0, 1.0))


def _vol_array(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(seed=20260514)
    if dtype.startswith("int") or dtype.startswith("uint"):
        info = np.iinfo(np.dtype(dtype))
        return rng.integers(
            low=max(info.min, -100), high=min(info.max, 100), size=_VOL_SHAPE
        ).astype(dtype)
    return rng.standard_normal(size=_VOL_SHAPE).astype(dtype)


def _vec_array(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(seed=20260515)
    if dtype.startswith("int") or dtype.startswith("uint"):
        info = np.iinfo(np.dtype(dtype))
        return rng.integers(
            low=max(info.min, -100), high=min(info.max, 100), size=_VEC_SHAPE
        ).astype(dtype)
    return rng.standard_normal(size=_VEC_SHAPE).astype(dtype)


def _vol_factory(name: str, arr: np.ndarray, space: NeuroSpace):
    if name == "dense":
        return DenseNeuroVol(arr, space)
    if name == "sparse":
        full_mask = LogicalNeuroVol(
            np.ones(arr.shape, dtype=bool), space
        )
        return SparseNeuroVol(arr, space, mask=full_mask)
    raise ValueError(name)


def _vec_factory(name: str, arr: np.ndarray, space: NeuroSpace, tmp_path: Path):
    if name == "dense":
        return DenseNeuroVec(arr, space)
    if name == "sparse":
        full_mask = np.ones(arr.shape[:3], dtype=bool)
        return DenseNeuroVec(arr, space).as_sparse(full_mask)
    if name == "mapped":
        return MappedNeuroVec(
            DenseNeuroVec(arr, space), map_fun=lambda x: x, inverse_fun=lambda x: x
        )
    if name == "file_backed":
        import nibabel as nib

        affine = np.asarray(space.trans[:4, :4], dtype=float)
        filenames = []
        for t in range(arr.shape[-1]):
            path = tmp_path / f"vol_{t:03d}.nii.gz"
            nib.save(nib.Nifti1Image(arr[..., t], affine), str(path))
            filenames.append(str(path))
        return FileBackedNeuroVec(filenames)
    raise ValueError(name)


@pytest.mark.parametrize("backend_name", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_vol_write_read_preserves_dtype(tmp_path, backend_name, dtype):
    """Primary ME-7 invariant: explicit-dtype write_vol/read_vol round-trips
    recover the on-disk dtype for every Vol backend.
    """
    space = _vol_space()
    arr = _vol_array(dtype)
    vol = _vol_factory(backend_name, arr, space)

    target = tmp_path / f"{backend_name}_{dtype}.nii.gz"
    write_vol(vol, str(target), data_type=_to_nifti_dtype(arr.dtype))
    recovered = read_vol(str(target))

    assert np.dtype(recovered.data.dtype) == np.dtype(dtype), (
        f"Vol round-trip dtype drift: backend={backend_name} "
        f"input={dtype} recovered={recovered.data.dtype}"
    )


@pytest.mark.parametrize("backend_name", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_vol_write_read_preserves_values(tmp_path, backend_name, dtype):
    """Value-parity round-trip across Vol backends.  Originally held sparse
    out pending bd-01KRKKM7AMS57HPMG7MHEMP48V; that path now ravels data in
    F-order to match the mask-path index ordering, so sparse rejoins the
    matrix.
    """
    space = _vol_space()
    arr = _vol_array(dtype)
    vol = _vol_factory(backend_name, arr, space)

    target = tmp_path / f"{backend_name}_{dtype}.nii.gz"
    write_vol(vol, str(target), data_type=_to_nifti_dtype(arr.dtype))
    recovered = read_vol(str(target))

    np.testing.assert_array_equal(
        recovered.data.astype(dtype), arr.astype(dtype),
        err_msg=f"Vol round-trip value drift: backend={backend_name} dtype={dtype}",
    )


@pytest.mark.parametrize(
    "backend_name", ["dense", "sparse", "mapped", "file_backed"]
)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_vec_write_read_preserves_dtype(tmp_path, backend_name, dtype):
    space = _vec_space()
    arr = _vec_array(dtype)
    vec = _vec_factory(backend_name, arr, space, tmp_path)

    target = tmp_path / f"vec_{backend_name}_{dtype}.nii.gz"
    write_vec(vec, str(target), data_type=_to_nifti_dtype(arr.dtype))
    recovered = read_vec(str(target))

    assert np.dtype(recovered.data.dtype) == np.dtype(dtype), (
        f"Vec round-trip dtype drift: backend={backend_name} "
        f"input={dtype} recovered={recovered.data.dtype}"
    )
    np.testing.assert_array_equal(
        recovered.data.astype(dtype), arr.astype(dtype),
        err_msg=f"Vec round-trip value drift: backend={backend_name} dtype={dtype}",
    )


# A second matrix asserting the parity claim WITHOUT explicit data_type:
# this is the form the issue body literally writes (`read_vol(write_vol(x))`).
# Behavior here is the "default-preservation" contract — when the caller does
# not specify a target on-disk dtype, the round-trip preserves the input
# dtype.  Today's NIfTI default is FLOAT32, so float64 silently downcasts;
# this matrix documents that drift class through xfail markers until the
# defaults are corrected in a follow-up.


@pytest.mark.parametrize("dtype", _DTYPES)
def test_vol_default_write_dtype_drift_documented(tmp_path, dtype):
    """Documents which dtypes survive the default NIfTI write path.

    float32 round-trips cleanly today.  float64 silently downcasts to
    float32 because :func:`write_vol` defaults ``data_type='FLOAT'``.
    Integer dtypes round-trip cleanly when ``read_vol`` uses ``get_fdata``
    only because the writer chose float32 — so the recovered array is
    float32, not the input integer dtype.
    """
    space = _vol_space()
    arr = _vol_array(dtype)
    vol = DenseNeuroVol(arr, space)

    target = tmp_path / f"default_{dtype}.nii.gz"
    write_vol(vol, str(target))
    recovered = read_vol(str(target))

    if dtype == "float32":
        assert np.dtype(recovered.data.dtype) == np.dtype("float32")
    else:
        # documented drift: the default write path casts to float32
        assert np.dtype(recovered.data.dtype) != np.dtype(dtype), (
            "default write path no longer drifts dtype; lift the xfail in "
            "test_vol_default_write_dtype_drift_documented and tighten the "
            "default behaviour."
        )
