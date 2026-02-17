"""Indexing utilities for neuroimaging data.

This module provides functions for linear and matricized access to
NeuroVol and NeuroVec data, as well as conversions between matrix
representations and NeuroVec objects.
"""

import numpy as np
from typing import Tuple, Union

from .neuro_vol import NeuroVol, DenseNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .neuro_space import NeuroSpace


def linear_access(vol_or_vec: Union[NeuroVol, NeuroVec], indices: np.ndarray) -> np.ndarray:
    """Extract values using linear (1D) indices into the 3D spatial grid.

    For a NeuroVol the returned array has one value per index.
    For a NeuroVec the returned array has shape ``(n_timepoints, n_indices)``
    -- one time series per index.

    Parameters
    ----------
    vol_or_vec : NeuroVol or NeuroVec
        The neuroimaging object to index into.
    indices : array-like
        1D array of linear indices (Fortran / column-major order) into the
        spatial grid.

    Returns
    -------
    np.ndarray
        Extracted values.
    """
    indices = np.asarray(indices, dtype=int)

    if isinstance(vol_or_vec, NeuroVec):
        # Use the series() method which accepts linear indices
        return vol_or_vec.series(indices)
    elif isinstance(vol_or_vec, NeuroVol):
        flat = vol_or_vec.as_dense().data.ravel(order='F')
        return flat[indices]
    else:
        raise TypeError(f"Expected NeuroVol or NeuroVec, got {type(vol_or_vec)}")


def matricized_access(vec: NeuroVec, row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
    """Extract values using a time x space index pair.

    Selects specific time points (rows) and specific voxels (columns) from
    the underlying time x voxel matrix.

    Parameters
    ----------
    vec : NeuroVec
        A 4D neuroimaging vector.
    row_indices : array-like
        Indices along the time dimension (axis 3).
    col_indices : array-like
        Linear spatial indices (Fortran order) into the 3D grid.

    Returns
    -------
    np.ndarray
        2D array of shape ``(len(row_indices), len(col_indices))``.
    """
    row_indices = np.asarray(row_indices, dtype=int)
    col_indices = np.asarray(col_indices, dtype=int)

    # Get the full time x space matrix
    if isinstance(vec, DenseNeuroVec):
        mat = vec.data.reshape(-1, vec.shape[3], order='F').T  # (time, voxels)
    else:
        mat = vec.as_dense().data.reshape(-1, vec.shape[3], order='F').T

    return mat[np.ix_(row_indices, col_indices)]


def from_matvec(mat: np.ndarray, space: NeuroSpace) -> DenseNeuroVec:
    """Convert a time x voxels matrix to a DenseNeuroVec.

    Parameters
    ----------
    mat : np.ndarray
        2D array of shape ``(n_timepoints, n_voxels)`` where
        ``n_voxels == prod(space.dim[:3])`` and ``n_timepoints == space.dim[3]``.
    space : NeuroSpace
        A 4D NeuroSpace describing the target geometry.

    Returns
    -------
    DenseNeuroVec
        Dense 4D vector.

    Raises
    ------
    ValueError
        If *mat* dimensions do not match *space*.
    """
    if mat.ndim != 2:
        raise ValueError(f"mat must be 2D, got {mat.ndim}D")

    n_time, n_vox = mat.shape
    expected_vox = int(np.prod(space.dim[:3]))
    expected_time = int(space.dim[3])

    if n_vox != expected_vox:
        raise ValueError(
            f"Number of columns ({n_vox}) does not match spatial volume "
            f"({expected_vox})"
        )
    if n_time != expected_time:
        raise ValueError(
            f"Number of rows ({n_time}) does not match time dimension "
            f"({expected_time})"
        )

    # Reshape: each row is a flattened volume in Fortran order
    data_4d = np.zeros(tuple(int(d) for d in space.dim), dtype=mat.dtype)
    for t in range(n_time):
        data_4d[..., t] = mat[t].reshape(space.dim[:3], order='F')

    return DenseNeuroVec(data_4d, space)


def to_matvec(vec: NeuroVec) -> Tuple[np.ndarray, NeuroSpace]:
    """Convert a NeuroVec to a ``(matrix, space)`` tuple.

    The matrix has shape ``(n_timepoints, n_voxels)`` with voxels
    stored in Fortran (column-major) order.

    Parameters
    ----------
    vec : NeuroVec
        A 4D neuroimaging vector.

    Returns
    -------
    mat : np.ndarray
        2D array ``(n_timepoints, n_voxels)``.
    space : NeuroSpace
        The 4D space of the input vector.
    """
    if isinstance(vec, DenseNeuroVec):
        data = vec.data
    else:
        data = vec.as_dense().data

    n_time = data.shape[3]
    n_vox = int(np.prod(data.shape[:3]))
    mat = np.zeros((n_time, n_vox), dtype=data.dtype)
    for t in range(n_time):
        mat[t] = data[..., t].ravel(order='F')

    return mat, vec.space


def dot_reduce(vec: NeuroVec, weights: np.ndarray) -> DenseNeuroVol:
    """Dot-product reduction: multiply time series by weights and sum across time.

    For each voxel, computes ``sum(time_series * weights)`` producing a
    single 3D volume.

    Parameters
    ----------
    vec : NeuroVec
        A 4D neuroimaging vector.
    weights : array-like
        1D array of length ``n_timepoints``.

    Returns
    -------
    DenseNeuroVol
        3D volume containing the weighted sums.

    Raises
    ------
    ValueError
        If length of *weights* does not match the time dimension.
    """
    weights = np.asarray(weights, dtype=float)
    n_time = vec.shape[3]

    if weights.shape != (n_time,):
        raise ValueError(
            f"weights length ({weights.size}) must match time dimension ({n_time})"
        )

    if isinstance(vec, DenseNeuroVec):
        data = vec.data
    else:
        data = vec.as_dense().data

    # Weighted sum along time axis: data is (x, y, z, t), weights is (t,)
    result = np.tensordot(data, weights, axes=([3], [0]))

    vol_space = NeuroSpace(
        dim=vec.shape[:3],
        spacing=vec.spacing[:3],
        origin=vec.origin[:3],
    )
    return DenseNeuroVol(result, vol_space)
