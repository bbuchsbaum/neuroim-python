"""Data manipulation operations for neuroimaging data.

This module provides functions for concatenating, scaling, mapping,
and downsampling neuroimaging volumes and vectors.
"""

import warnings

import numpy as np
from scipy import ndimage
from typing import Union, Callable

from .neuro_vol import NeuroVol, DenseNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .neuro_space import NeuroSpace
from .typing import NeuroVecLike, NeuroVolLike
from .verify import assert_same_space


def concat(*vecs: NeuroVecLike) -> DenseNeuroVec:
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
        try:
            assert_same_space(first.space, v.space)
        except ValueError as exc:
            raise ValueError(
                "spatial contract mismatch: NeuroVecs must have same "
                f"spatial dimensions and affine; {exc}"
            ) from None

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
        trans=first.space.trans,
    )
    out = DenseNeuroVec(concat_data, concat_space)

    # Provenance threading (ME-9): attach a Receipt that composes any
    # upstream Receipts on the inputs.  When inputs lack provenance, fall
    # back to a fresh "concat" Receipt anchored on the first input's space.
    from .results import make_receipt

    upstream_receipts = [
        getattr(v, "provenance", None) for v in vecs
    ]
    upstream_receipts = [r for r in upstream_receipts if r is not None]
    base = make_receipt(
        input_space=concat_space,
        mask_data=None,
        n_voxels=int(np.prod(spatial_shape)),
        method_name="concat",
        source_affine=concat_space.trans,
    )
    if upstream_receipts:
        merged = upstream_receipts[0]
        for r in upstream_receipts[1:]:
            try:
                merged = merged.merge(r)
            except ValueError:
                # Inputs disagree on space/mask: keep the conflict surfaced in
                # the receipt by leaving the merge field bare; concat itself
                # already asserts spatial-shape compatibility above.
                break
        try:
            base = base.merge(merged, method_name="concat")
        except ValueError:
            pass
    out.provenance = base
    return out


def scale_series(vec: NeuroVecLike, method: str = "zscore") -> DenseNeuroVec:
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


def mapf(vol_or_vec: Union[NeuroVolLike, NeuroVecLike], func: Callable) -> Union[DenseNeuroVol, DenseNeuroVec]:
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


def _spatial_affine(space: NeuroSpace) -> np.ndarray:
    """Return a 4x4 affine for the spatial axes of a 3D or 4D space."""
    affine = np.eye(4)
    affine[:3, :3] = space.trans[:3, :3]
    affine[:3, 3] = space.trans[:3, -1]
    return affine


def _space_with_downsampled_grid(space: NeuroSpace, out_shape: tuple[int, int, int]) -> NeuroSpace:
    from .orientation import rescale_affine

    in_shape = tuple(int(d) for d in space.dim[:3])
    old_spacing = np.asarray(space.spacing[:3], dtype=float)
    new_spacing = old_spacing * (np.asarray(in_shape, dtype=float) / np.asarray(out_shape, dtype=float))
    new_affine = rescale_affine(_spatial_affine(space), in_shape, new_spacing, out_shape)

    if space.ndim == 4:
        new_dim = (*out_shape, int(space.dim[3]))
        spacing = np.concatenate([new_spacing, np.asarray(space.spacing[3:4], dtype=float)])
        origin_tail = np.asarray(space.origin[3:4], dtype=float) if len(space.origin) > 3 else np.array([0.0])
        origin = np.concatenate([new_affine[:3, 3], origin_tail])
        return NeuroSpace(new_dim, spacing=spacing, origin=origin, axes=space.axes, trans=new_affine)

    return NeuroSpace(out_shape, spacing=new_spacing, origin=new_affine[:3, 3], axes=space.axes, trans=new_affine)


def _resolve_downsample_shape(
    shape: tuple[int, int, int],
    current_spacing: np.ndarray,
    factor=None,
    spacing=None,
    outdim=None,
) -> tuple[int, int, int]:
    requested = sum(x is not None for x in (factor, spacing, outdim))
    if requested != 1:
        raise ValueError("Exactly one of factor, spacing, or outdim must be specified")

    shape_arr = np.asarray(shape, dtype=int)

    if factor is not None:
        fac = np.asarray(factor, dtype=float)
        if fac.ndim == 0:
            fac = np.repeat(float(fac), 3)
        if fac.size != 3:
            raise ValueError("factor must be a scalar or have length 3")
        if np.any(~np.isfinite(fac)) or np.any(fac <= 0):
            raise ValueError("factor must be positive")
        if np.all(fac <= 1):
            out = np.floor(shape_arr * fac).astype(int)
        else:
            if not np.allclose(fac, np.round(fac)):
                raise ValueError("factor values greater than 1 must be integer block sizes")
            out = np.floor(shape_arr / np.round(fac).astype(int)).astype(int)
    elif spacing is not None:
        target_spacing = np.asarray(spacing, dtype=float)
        if target_spacing.ndim == 0:
            target_spacing = np.repeat(float(target_spacing), 3)
        if target_spacing.size != 3:
            raise ValueError("spacing must have length 3")
        if np.any(~np.isfinite(target_spacing)) or np.any(target_spacing <= 0):
            raise ValueError("spacing values must be positive")
        out = np.floor(shape_arr * current_spacing / target_spacing).astype(int)
    else:
        out = np.asarray(outdim, dtype=int)
        if out.size != 3:
            raise ValueError("outdim must contain exactly 3 values")
        ratios = shape_arr / out.astype(float)
        if np.ptp(ratios) > 1e-6:
            warnings.warn("downsample outdim changes the spatial aspect ratio", UserWarning, stacklevel=2)

    if np.any(out <= 0):
        raise ValueError("downsample output dimensions must be positive")
    if np.any(out > shape_arr):
        raise ValueError("downsample output dimensions must not exceed input dimensions")
    return tuple(int(x) for x in out)


def _downsample_array(data: np.ndarray, out_shape: tuple[int, int, int]) -> np.ndarray:
    in_shape = np.asarray(data.shape[:3], dtype=int)
    out = np.asarray(out_shape, dtype=int)
    block = in_shape / out

    if np.allclose(block, np.round(block)):
        block = np.round(block).astype(int)
        trimmed = data[: out[0] * block[0], : out[1] * block[1], : out[2] * block[2], ...]
        trailing = data.shape[3:]
        reshaped = trimmed.reshape(
            out[0], block[0], out[1], block[1], out[2], block[2], *trailing
        )
        return reshaped.mean(axis=(1, 3, 5))

    zoom = tuple(out / in_shape) + (1.0,) * (data.ndim - 3)
    return ndimage.zoom(data, zoom=zoom, order=1)


def downsample(
    vol: Union[NeuroVolLike, NeuroVecLike],
    factor=None,
    spacing=None,
    outdim=None,
    method: str = "box",
) -> Union[DenseNeuroVol, DenseNeuroVec]:
    """Reduce spatial resolution using neuroim2-style downsampling controls.

    Parameters
    ----------
    vol : NeuroVol or NeuroVec
        The 3D volume or 4D vector to downsample spatially.
    factor : scalar or length-3 sequence, optional
        A ratio in ``(0, 1]`` or a legacy integer block size ``>= 1``.
    spacing : scalar or length-3 sequence, optional
        Requested output voxel spacing.
    outdim : length-3 sequence, optional
        Requested spatial output dimensions.
    method : str
        Currently only ``"box"`` is supported.

    Returns
    -------
    DenseNeuroVol or DenseNeuroVec
        Downsampled object with updated NeuroSpace. The time dimension of
        NeuroVec inputs is preserved.

    Raises
    ------
    ValueError
        If the request is invalid.
    """
    if method != "box":
        raise ValueError("Only 'box' downsampling is supported")

    if not isinstance(vol, (NeuroVol, NeuroVec)):
        raise TypeError(f"Expected NeuroVol or NeuroVec, got {type(vol)}")

    if isinstance(vol, (DenseNeuroVol, DenseNeuroVec)):
        data = vol.data
    else:
        data = vol.as_dense().data
    out_shape = _resolve_downsample_shape(
        tuple(int(d) for d in data.shape[:3]),
        np.asarray(vol.space.spacing[:3], dtype=float),
        factor=factor,
        spacing=spacing,
        outdim=outdim,
    )

    if tuple(data.shape[:3]) == out_shape:
        downsampled = data.copy()
    else:
        downsampled = _downsample_array(data, out_shape)

    new_space = _space_with_downsampled_grid(vol.space, out_shape)
    if isinstance(vol, NeuroVec):
        return DenseNeuroVec(downsampled, new_space)
    return DenseNeuroVol(downsampled, new_space)
