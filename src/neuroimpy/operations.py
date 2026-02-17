"""Data manipulation operations for neuroimaging data.

This module provides functions for concatenating, scaling, mapping,
and downsampling neuroimaging volumes and vectors.
"""

import numpy as np
from typing import Union, Callable

from .neuro_vol import NeuroVol, DenseNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .neuro_space import NeuroSpace


def concat(*vecs: NeuroVec) -> DenseNeuroVec:
    """Concatenate NeuroVec objects along the time dimension.

    Parameters
    ----------
    *vecs : NeuroVec
        Two or more NeuroVec objects to concatenate. All must share the
        same spatial dimensions (first 3 dims).

    Returns
    -------
    DenseNeuroVec
        A new dense vector with concatenated time points.

    Raises
    ------
    ValueError
        If fewer than two vectors are given or spatial dimensions differ.
    """
    if len(vecs) < 2:
        raise ValueError("concat requires at least two NeuroVec objects")

    first = vecs[0]
    spatial_shape = first.shape[:3]

    for v in vecs[1:]:
        if v.shape[:3] != spatial_shape:
            raise ValueError(
                f"All NeuroVecs must share the same spatial dimensions. "
                f"Expected {spatial_shape}, got {v.shape[:3]}"
            )

    # Collect 4D arrays, converting sparse to dense as needed
    arrays = []
    for v in vecs:
        if isinstance(v, DenseNeuroVec):
            arrays.append(v.data)
        else:
            # SparseNeuroVec or other subclass -- materialize dense array
            arrays.append(v.as_dense().data)

    concat_data = np.concatenate(arrays, axis=3)
    total_time = concat_data.shape[3]

    concat_space = NeuroSpace(
        dim=(*spatial_shape, total_time),
        spacing=first.spacing,
        origin=first.origin,
        axes=first.space.axes,
    )
    return DenseNeuroVec(concat_data, concat_space)


def scale_series(vec: NeuroVec, method: str = "zscore") -> DenseNeuroVec:
    """Per-voxel time-series normalization.

    Parameters
    ----------
    vec : NeuroVec
        A 4D neuroimaging vector.
    method : str
        Normalization method. One of:
        - ``"zscore"``: subtract mean, divide by standard deviation
        - ``"mean_center"``: subtract mean only
        - ``"unit_scale"``: scale to [0, 1] range per voxel

    Returns
    -------
    DenseNeuroVec
        A new vector with normalized time series.

    Raises
    ------
    ValueError
        If *method* is not recognized.
    """
    valid_methods = ("zscore", "mean_center", "unit_scale")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    # Get dense 4D data
    if isinstance(vec, DenseNeuroVec):
        data = vec.data.copy()
    else:
        data = vec.as_dense().data.copy()

    # Operate along axis=3 (time)
    if method == "zscore":
        mean = np.mean(data, axis=3, keepdims=True)
        std = np.std(data, axis=3, keepdims=True)
        std[std == 0] = 1.0
        data = (data - mean) / std
    elif method == "mean_center":
        mean = np.mean(data, axis=3, keepdims=True)
        data = data - mean
    elif method == "unit_scale":
        vmin = np.min(data, axis=3, keepdims=True)
        vmax = np.max(data, axis=3, keepdims=True)
        denom = vmax - vmin
        denom[denom == 0] = 1.0
        data = (data - vmin) / denom

    return DenseNeuroVec(data, vec.space)


def mapf(vol_or_vec: Union[NeuroVol, NeuroVec], func: Callable) -> Union[DenseNeuroVol, DenseNeuroVec]:
    """Apply a function element-wise to the data of a volume or vector.

    Parameters
    ----------
    vol_or_vec : NeuroVol or NeuroVec
        The input neuroimaging object.
    func : callable
        A function that accepts and returns an ndarray (element-wise).

    Returns
    -------
    DenseNeuroVol or DenseNeuroVec
        A new object of the corresponding dense type with transformed data.
    """
    if isinstance(vol_or_vec, NeuroVec):
        if isinstance(vol_or_vec, DenseNeuroVec):
            new_data = func(vol_or_vec.data)
        else:
            new_data = func(vol_or_vec.as_dense().data)
        return DenseNeuroVec(new_data, vol_or_vec.space)
    elif isinstance(vol_or_vec, NeuroVol):
        if isinstance(vol_or_vec, DenseNeuroVol):
            new_data = func(vol_or_vec.data)
        else:
            new_data = func(vol_or_vec.as_dense().data)
        return DenseNeuroVol(new_data, vol_or_vec.space)
    else:
        raise TypeError(f"Expected NeuroVol or NeuroVec, got {type(vol_or_vec)}")


def downsample(vol: NeuroVol, factor: int) -> DenseNeuroVol:
    """Reduce spatial resolution of a volume by an integer factor using block averaging.

    Parameters
    ----------
    vol : NeuroVol
        The 3D volume to downsample.
    factor : int
        Integer downsampling factor (must be >= 1).

    Returns
    -------
    DenseNeuroVol
        Downsampled volume with updated NeuroSpace.

    Raises
    ------
    ValueError
        If *factor* is less than 1.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return DenseNeuroVol(vol.as_dense().data.copy(), vol.space)

    data = vol.as_dense().data
    sx, sy, sz = data.shape

    # Trim to multiple of factor
    nx = sx // factor
    ny = sy // factor
    nz = sz // factor

    trimmed = data[:nx * factor, :ny * factor, :nz * factor]

    # Reshape and average over block axes
    reshaped = trimmed.reshape(nx, factor, ny, factor, nz, factor)
    downsampled = reshaped.mean(axis=(1, 3, 5))

    new_spacing = vol.space.spacing[:3] * factor
    new_origin = vol.space.origin[:3]
    new_space = NeuroSpace(
        dim=(nx, ny, nz),
        spacing=new_spacing,
        origin=new_origin,
    )
    return DenseNeuroVol(downsampled, new_space)
