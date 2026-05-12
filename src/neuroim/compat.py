"""Compatibility helpers for neuroim2-style generic functions.

The core Python classes expose idiomatic attributes and methods.  This module
adds thin dispatch wrappers for the R package's exported S4 generic names so
migration code can call ``neuroim.space(x)`` or ``neuroim.values(x)``
without duplicating logic already implemented on the objects.
"""

from __future__ import annotations

from typing import Any, Protocol, Union, runtime_checkable

import numpy as np


@runtime_checkable
class NeuroObj(Protocol):
    """Protocol for objects carrying a :class:`NeuroSpace`.

    R's ``NeuroObj`` is an S4 virtual superclass.  In Python this is a
    structural protocol because concrete objects already expose ``.space``.
    """

    space: Any


@runtime_checkable
class ArrayLike3D(Protocol):
    """Structural equivalent of neuroim2's virtual ``ArrayLike3D`` class."""

    shape: tuple[int, int, int]

    def __getitem__(self, key: Any) -> Any:
        ...


@runtime_checkable
class ArrayLike4D(Protocol):
    """Structural equivalent of neuroim2's virtual ``ArrayLike4D`` class."""

    shape: tuple[int, int, int, int]

    def __getitem__(self, key: Any) -> Any:
        ...


@runtime_checkable
class ArrayLike5D(Protocol):
    """Structural equivalent of neuroim2's virtual ``ArrayLike5D`` class."""

    shape: tuple[int, ...]

    def __getitem__(self, key: Any) -> Any:
        ...


numericOrMatrix = Union[int, float, np.ndarray]


def _member(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    if not hasattr(obj, name):
        raise TypeError(f"{type(obj).__name__!s} does not provide {name!r}")
    value = getattr(obj, name)
    return value(*args, **kwargs) if callable(value) else value


def _maybe_member(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) else value
    raise TypeError(f"{type(obj).__name__!s} does not provide any of {names!r}")


def space(x: Any) -> Any:
    return _maybe_member(x, "space")


def ndim(x: Any) -> int:
    return int(_maybe_member(x, "ndim"))


def spacing(x: Any) -> np.ndarray:
    return _maybe_member(x, "spacing")


def origin(x: Any) -> np.ndarray:
    return _maybe_member(x, "origin")


def axes(x: Any) -> Any:
    return _maybe_member(x, "axes")


def trans(x: Any) -> np.ndarray:
    return _maybe_member(x, "trans")


def inverse_trans(x: Any) -> np.ndarray:
    sp = space(x) if not hasattr(x, "inverse") else x
    return _member(sp, "inverse")


def bounds(x: Any) -> np.ndarray:
    return _member(x, "bounds")


def centroid(x: Any) -> np.ndarray:
    return _member(x, "centroid")


def dim_of(x: Any, axis: Any) -> int:
    return _member(x, "dim_of", axis)


def which_dim(x: Any, axis: Any) -> int:
    return _member(x, "which_dim", axis)


def add_dim(x: Any, n: int = 1, size: int = 1) -> Any:
    return _member(x, "add_dim", n, size)


def drop_dim(x: Any, dimnum: int) -> Any:
    return _member(x, "drop_dim", dimnum)


def coord_to_grid(x: Any, coords: np.ndarray) -> np.ndarray:
    return _member(x, "coord_to_grid", coords)


def grid_to_coord(x: Any, grid: np.ndarray) -> np.ndarray:
    return _member(x, "grid_to_coord", grid)


def coord_to_index(x: Any, coords: np.ndarray) -> np.ndarray:
    return _member(x, "coord_to_index", coords)


def index_to_coord(x: Any, idx: Any) -> np.ndarray:
    return _member(x, "index_to_coord", idx)


def grid_to_index(x: Any, grid: np.ndarray) -> np.ndarray:
    return _member(x, "grid_to_index", grid)


def grid_to_grid(x: Any, grid: np.ndarray, target_space: Any) -> np.ndarray:
    return _member(x, "grid_to_grid", grid, target_space)


def index_to_grid(x: Any, idx: Any) -> np.ndarray:
    return _member(x, "index_to_grid", idx)


def values(x: Any) -> np.ndarray:
    if hasattr(x, "values"):
        value = getattr(x, "values")
        return value() if callable(value) else value
    if hasattr(x, "data"):
        return x.data
    raise TypeError(f"{type(x).__name__!s} does not provide values")


def coords(x: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    return _member(x, "coords", *args, **kwargs)


def voxels(x: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    return coords(x, *args, **kwargs)


def indices(x: Any) -> np.ndarray:
    return _member(x, "indices")


def mask(x: Any) -> Any:
    return _maybe_member(x, "mask", "as_mask")


def as_dense(x: Any) -> Any:
    return _member(x, "as_dense")


def as_sparse(x: Any, mask: Any = None) -> Any:
    if mask is None:
        return _member(x, "as_sparse")
    return _member(x, "as_sparse", mask)


def as_mask(x: Any, indices: Any = None) -> Any:
    if indices is None:
        return _member(x, "as_mask")
    return _member(x, "as_mask", indices)


def as_matrix(x: Any) -> np.ndarray:
    if hasattr(x, "as_matrix"):
        return x.as_matrix()
    return np.asarray(x)


def as_array(x: Any) -> np.ndarray:
    if hasattr(x, "as_dense"):
        return np.asarray(x.as_dense().data)
    if hasattr(x, "data"):
        return np.asarray(x.data)
    return np.asarray(x)


def series(x: Any, i: Any, *args: Any) -> np.ndarray:
    if args:
        return _member(x, "series", i, *args)
    return _member(x, "series", i)


def series_roi(x: Any, roi: Any) -> np.ndarray:
    return _member(x, "series_roi", roi)


def vols(x: Any, indices: Any = None) -> Any:
    if indices is None:
        return _member(x, "vols")
    return _member(x, "vols", indices)


def vectors(x: Any, subset: Any = None) -> Any:
    if subset is None:
        return _member(x, "vectors")
    return _member(x, "vectors", subset)


def sub_vector(x: Any, i: Any, *args: Any, **kwargs: Any) -> Any:
    return _member(x, "sub_vector", i, *args, **kwargs)


def temporal_access(x: Any, i: Any) -> np.ndarray:
    data = values(x)
    if data.ndim == 4:
        mat = data.reshape(-1, data.shape[3], order="F").T
        return mat[np.atleast_1d(i), :]
    if data.ndim == 2:
        return data[np.atleast_1d(i), :]
    raise TypeError("temporal_access expects a NeuroVec-like object")


def volume_labels(x: Any) -> list[str]:
    labels = getattr(x, "volume_labels", None)
    if labels is None:
        return []
    return list(labels)


def lookup(x: Any, i: Any, *args: Any) -> Any:
    if hasattr(x, "lookup"):
        return x.lookup(i, *args)
    if hasattr(x, "lookup_index"):
        return x.lookup_index(i)
    return x[i]


def num_clusters(x: Any) -> int:
    return int(_member(x, "num_clusters"))


def sub_clusters(x: Any, ids: Any) -> Any:
    return _member(x, "sub_clusters", ids)


def file_matches(x: Any, file_name: Any) -> bool:
    return bool(_member(x, "file_matches", file_name))


def header_file_matches(x: Any, file_name: Any) -> bool:
    return bool(_member(x, "header_file_matches", file_name))


def data_file_matches(x: Any, file_name: Any) -> bool:
    return bool(_member(x, "data_file_matches", file_name))


def header_file(x: Any, file_name: Any) -> str:
    return _member(x, "header_file", file_name)


def data_file(x: Any, file_name: Any) -> str:
    return _member(x, "data_file", file_name)


def strip_extension(x: Any, file_name: Any) -> str:
    return _member(x, "strip_extension", file_name)


def read_elements(x: Any, num_elements: int) -> np.ndarray:
    return _member(x, "read_elements", num_elements)


def read_columns(x: Any, column_indices: Any) -> np.ndarray:
    return _member(x, "read_columns", column_indices)


def write_elements(x: Any, els: Any) -> Any:
    return _member(x, "write_elements", els)


def header(x: Any) -> Any:
    return _maybe_member(x, "header")


def extensions(x: Any) -> Any:
    return _maybe_member(x, "extensions")


def has_extensions(x: Any) -> bool:
    if hasattr(x, "has_extensions"):
        value = getattr(x, "has_extensions")
        return bool(value() if callable(value) else value)
    return bool(extensions(x))


def extension(x: Any, code: Any = None) -> Any:
    exts = extensions(x)
    if code is None:
        return exts
    return [ext for ext in exts if getattr(ext, "code", None) == code]


def data_reader(x: Any, offset: int = 0) -> Any:
    return _member(x, "data_reader", offset)


def meta_info(x: Any) -> Any:
    return _member(x, "meta_info") if hasattr(x, "meta_info") else x


def load_data(x: Any, *args: Any, **kwargs: Any) -> Any:
    return _member(x, "load_data", *args, **kwargs)


def scale(x: Any, *args: Any, **kwargs: Any) -> Any:
    if hasattr(x, "scale_series"):
        return x.scale_series(*args, **kwargs)
    return np.asarray(x) * 1


def drop(x: Any) -> Any:
    if hasattr(x, "drop"):
        return x.drop()
    return np.squeeze(x)


def as_canonical(x: Any, target: Any = ("R", "A", "S")) -> Any:
    from .resample import reorient

    if isinstance(target, str):
        orient = target
    else:
        orient = "".join(target)
    return reorient(x, orient)


def slice_to_volume_affine(index: int, axis: int, shape: Any = None,
                           index_base: str = "R") -> np.ndarray:
    """Affine from 2D slice coordinates to 3D volume coordinates."""
    if index_base not in {"R", "zero"}:
        raise ValueError("index_base must be 'R' or 'zero'")
    if axis in (0, 1, 2):
        axis0 = int(axis)
    elif axis in (1, 2, 3):
        axis0 = int(axis) - 1
    else:
        raise ValueError("axis must be in 1..3 or 0..2")

    index = int(index)
    if index_base == "R":
        if index < 1:
            raise ValueError("For index_base='R', index must be >= 1")
        index0 = index - 1
    else:
        if index < 0:
            raise ValueError("For index_base='zero', index must be >= 0")
        index0 = index

    if shape is not None:
        shape_arr = np.asarray(shape, dtype=int)
        if shape_arr.size < 3 or np.any(shape_arr[:3] <= 0):
            raise ValueError("shape must provide at least 3 positive dimensions")
        if index0 >= shape_arr[axis0]:
            raise ValueError("index exceeds the provided shape along axis")

    keep = [i for i in range(4) if i != axis0]
    out = np.eye(4)[:, keep]
    out[axis0, 2] = index0
    return out


def slice2volume(index: int, axis: int, shape: Any = None,
                 index_base: str = "R") -> np.ndarray:
    return slice_to_volume_affine(index, axis, shape=shape, index_base=index_base)


def vec_from_vols(vols: Any, mask: Any = None) -> Any:
    from .neuro_vec import neurovecseq

    vec = neurovecseq(list(vols))
    if mask is not None:
        return vec.as_sparse(mask)
    return vec


def apply_mask(x: Any, mask: Any) -> Any:
    """Zero values outside a spatial mask."""
    from .neuro_vec import DenseNeuroVec, NeuroVec
    from .neuro_vol import DenseNeuroVol, LogicalNeuroVol, NeuroVol

    mask_vol = mask if isinstance(mask, LogicalNeuroVol) else LogicalNeuroVol(np.asarray(mask, dtype=bool), space(x).drop_dim(3) if getattr(x, "ndim", 0) == 4 else space(x))
    keep = np.asarray(mask_vol.data, dtype=bool)
    if isinstance(x, NeuroVec):
        data = x.as_dense().data.copy() if hasattr(x, "as_dense") else np.asarray(x.data).copy()
        data[~keep, :] = 0
        return DenseNeuroVec(data, x.space, getattr(x, "label", ""))
    if isinstance(x, NeuroVol):
        data = x.as_dense().data.copy() if hasattr(x, "as_dense") else np.asarray(x.data).copy()
        data[~keep] = 0
        return DenseNeuroVol(data, x.space)
    arr = np.asarray(x).copy()
    arr[~keep] = 0
    return arr


def _afni_clip_level_numeric(x: Any, mfrac: float = 0.5, nhist: int = 10000) -> float:
    vals = np.asarray(x, dtype=float).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size <= 222:
        return 0.0
    if not np.isfinite(mfrac) or mfrac <= 0 or mfrac >= 0.99:
        mfrac = 0.5
    vmax = float(np.max(vals))
    if not np.isfinite(vmax) or vmax < 1e-100:
        return 0.0

    integerish = np.all(np.abs(vals - np.round(vals)) < 1e-8)
    if integerish and vmax <= 32767:
        nhist_eff = 255 if vmax <= 255 else 32767
        sfac = 1.0
        bins = np.round(vals).astype(int)
    else:
        nhist_eff = int(nhist)
        sfac = nhist_eff / vmax
        bins = np.floor(sfac * vals + 0.499).astype(int)
    bins = bins[(bins >= 0) & (bins <= nhist_eff)]
    if bins.size <= 222:
        return 0.0

    hist = np.bincount(bins, minlength=nhist_eff + 1)
    npos = int(hist.sum())
    dsum = float(np.sum((np.arange(hist.size) ** 2) * hist))
    qq = int(0.65 * npos)
    ib = int(round(0.5 * np.sqrt(dsum / npos)))
    acc = 0
    ii = nhist_eff - 1
    while ii >= ib and acc < qq:
        acc += int(hist[ii])
        ii -= 1

    ncut = max(ii, 0)
    for _ in range(66):
        start = max(ncut, 0) + 1
        npos_cut = int(hist[start - 1:nhist_eff].sum()) if start <= nhist_eff else 0
        nhalf = npos_cut // 2
        acc = 0
        ii = max(ncut, 0)
        while ii < nhist_eff and acc < nhalf:
            acc += int(hist[ii])
            ii += 1
        old = ncut
        ncut = int(mfrac * ii)
        if ncut == old:
            break
    return min(float(ncut) / sfac, 1e38)


def _center_of_mass_zero_based(arr: np.ndarray) -> np.ndarray:
    weights = np.asarray(arr, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if not np.any(weights > 0):
        return (np.asarray(weights.shape, dtype=float) - 1.0) / 2.0
    grids = np.indices(weights.shape, dtype=float)
    total = float(np.sum(weights))
    return np.array([float(np.sum(grids[i] * weights) / total) for i in range(3)])


def _afni_gradual_clip_array(arr: Any, mfrac: float = 0.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError("gradual clip expects a 3D array")

    nx, ny, nz = arr.shape
    cm = _center_of_mass_zero_based(arr)
    ic, jc, kc = [
        min(max(int(round(cm[i])), 0), arr.shape[i] - 1)
        for i in range(3)
    ]
    it, jt, kt = nx - 1, ny - 1, nz - 1
    val_floor = 0.333 * _afni_clip_level_numeric(arr, mfrac=mfrac)

    ox = max(1, int(round(0.01 * nx)))
    oy = max(1, int(round(0.01 * ny)))
    oz = max(1, int(round(0.01 * nz)))

    icm, icp = max(ic - ox, 0), min(ic + ox, it)
    jcm, jcp = max(jc - oy, 0), min(jc + oy, jt)
    kcm, kcp = max(kc - oz, 0), min(kc + oz, kt)

    def octclip(xa: int, xb: int, ya: int, yb: int, za: int, zb: int) -> float:
        return max(
            _afni_clip_level_numeric(arr[xa:xb + 1, ya:yb + 1, za:zb + 1], mfrac=mfrac),
            val_floor,
        )

    c000 = octclip(0, icp, 0, jcp, 0, kcp)
    c100 = octclip(icm, it, 0, jcp, 0, kcp)
    c010 = octclip(0, icp, jcm, jt, 0, kcp)
    c110 = octclip(icm, it, jcm, jt, 0, kcp)
    c001 = octclip(0, icp, 0, jcp, kcm, kt)
    c101 = octclip(icm, it, 0, jcp, kcm, kt)
    c011 = octclip(0, icp, jcm, jt, kcm, kt)
    c111 = octclip(icm, it, jcm, jt, kcm, kt)

    x0, x1 = 0.5 * ic, 0.5 * (ic + it)
    y0, y1 = 0.5 * jc, 0.5 * (jc + jt)
    z0, z1 = 0.5 * kc, 0.5 * (kc + kt)

    xw1 = np.clip((np.arange(nx) - x0) / (x1 - x0), 0, 1) if x1 > x0 else np.zeros(nx)
    yw1 = np.clip((np.arange(ny) - y0) / (y1 - y0), 0, 1) if y1 > y0 else np.zeros(ny)
    zw1 = np.clip((np.arange(nz) - z0) / (z1 - z0), 0, 1) if z1 > z0 else np.zeros(nz)
    xw0, yw0, zw0 = 1 - xw1, 1 - yw1, 1 - zw1

    def outer3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return x[:, None, None] * y[None, :, None] * z[None, None, :]

    return (
        c000 * outer3(xw0, yw0, zw0)
        + c100 * outer3(xw1, yw0, zw0)
        + c010 * outer3(xw0, yw1, zw0)
        + c110 * outer3(xw1, yw1, zw0)
        + c001 * outer3(xw0, yw0, zw1)
        + c101 * outer3(xw1, yw0, zw1)
        + c011 * outer3(xw0, yw1, zw1)
        + c111 * outer3(xw1, yw1, zw1)
    )


def _connectivity_structure(connect: str) -> np.ndarray:
    if connect not in {"26-connect", "18-connect", "6-connect"}:
        raise ValueError("connect must be '26-connect', '18-connect', or '6-connect'")
    offsets = np.indices((3, 3, 3)).reshape(3, -1).T - 1
    shell = np.sum(np.abs(offsets), axis=1)
    if connect == "6-connect":
        keep = shell <= 1
    elif connect == "18-connect":
        keep = shell <= 2
    else:
        keep = shell <= 3
    return keep.reshape((3, 3, 3))


def _largest_component_mask(mask: np.ndarray, connect: str) -> np.ndarray:
    from scipy import ndimage

    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    labels, nlab = ndimage.label(mask, structure=_connectivity_structure(connect))
    if nlab == 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _fill_holes_mask(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_fill_holes(mask, structure=_connectivity_structure("6-connect"))


def _neighbor_count_18(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    offsets = np.indices((3, 3, 3)).reshape(3, -1).T - 1
    shell = np.sum(np.abs(offsets), axis=1)
    kernel = ((shell > 0) & (shell <= 2)).reshape((3, 3, 3)).astype(int)
    return ndimage.convolve(mask.astype(int), kernel, mode="nearest")


def _peel_restore_mask(mask: np.ndarray, peels: int = 1, peel_threshold: int = 17) -> np.ndarray:
    peels = int(peels)
    if peels < 1 or not np.any(mask):
        return mask

    real_peel_threshold = min(18, max(1, int(peel_threshold)))
    marks = np.zeros(mask.shape, dtype=int)
    out = mask.copy()

    for pp in range(1, peels + 1):
        counts = _neighbor_count_18(out)
        to_erode = out & (counts < real_peel_threshold)
        marks[to_erode] = pp
        out[to_erode] = False

    for pp in range(peels, 0, -1):
        counts = _neighbor_count_18(out)
        bth = 0 if pp == peels else 1
        to_restore = (marks >= pp) & ~out & (counts > bth)
        out[to_restore] = True

    return out


def _automask_array(arr: Any, mfrac: float = 0.5, gradual: bool = True,
                    peels: int = 1, peel_threshold: int = 17,
                    connect: str = "26-connect") -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError("automask expects a 3D array")

    clip = _afni_clip_level_numeric(arr, mfrac=mfrac)
    if not np.isfinite(clip) or clip <= 0:
        return np.zeros(arr.shape, dtype=bool)

    threshold = _afni_gradual_clip_array(arr, mfrac=mfrac) if gradual else np.full(arr.shape, clip)
    mask = arr >= threshold
    if not np.any(mask):
        return mask

    mask = _largest_component_mask(mask, connect=connect)
    if int(peels) > 0 and np.any(mask):
        mask = _peel_restore_mask(mask, peels=peels, peel_threshold=peel_threshold)
        if np.any(mask):
            mask = _largest_component_mask(mask, connect=connect)

    if np.any(mask):
        mask = _fill_holes_mask(mask)
        mask = _largest_component_mask(mask, connect=connect)
    return mask


def _image_array(x: Any) -> np.ndarray:
    if hasattr(x, "as_dense"):
        dense = x.as_dense()
        if hasattr(dense, "data"):
            return np.asarray(dense.data)
    if hasattr(x, "data"):
        return np.asarray(x.data)
    return np.asarray(x)


def clip_level(x: Any, mfrac: float = 0.5, gradual: bool = False,
               representative: str = "median") -> Any:
    """AFNI-like clip level estimate for volumes or vector representatives."""
    from .neuro_vol import DenseNeuroVol

    data = _image_array(x)
    if np.asarray(data).ndim == 4:
        if representative == "mean_abs":
            arr = np.mean(np.abs(data), axis=3)
        elif representative == "mean":
            arr = np.mean(data, axis=3)
        else:
            arr = np.median(data, axis=3)
    else:
        arr = np.asarray(data)
    level = _afni_clip_level_numeric(arr, mfrac=mfrac)
    if gradual:
        return DenseNeuroVol(_afni_gradual_clip_array(arr, mfrac=mfrac), space(x).drop_dim(3) if getattr(x, "ndim", 0) == 4 else space(x))
    return level


def automask(x: Any, mfrac: float = 0.5, gradual: bool = True,
             representative: str = "mean_abs", peels: int = 1,
             peel_threshold: int = 17, connect: str = "26-connect") -> Any:
    """Create a simple logical mask using an AFNI-style clip threshold."""
    from .neuro_vol import LogicalNeuroVol

    data = _image_array(x)
    if np.asarray(data).ndim == 4:
        if representative == "median":
            arr = np.median(np.abs(data), axis=3)
        else:
            arr = np.mean(np.abs(data), axis=3)
        sp = space(x).drop_dim(3)
    else:
        arr = np.abs(np.asarray(data))
        sp = space(x)
    return LogicalNeuroVol(
        _automask_array(
            arr,
            mfrac=mfrac,
            gradual=gradual,
            peels=peels,
            peel_threshold=peel_threshold,
            connect=connect,
        ),
        sp,
    )


def as_mmap(x: Any, file: Any = None, **kwargs: Any) -> Any:
    """Convert a NeuroVec-like object to a memory-mapped BigNeuroVec."""
    from .big_neuro_vec import BigNeuroVec

    filename = None if file is None else str(file)
    return BigNeuroVec(values(x), space(x), filename=filename)


def _parse_mapped_voxels(mapped_voxels: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(mapped_voxels, "dim") and hasattr(mapped_voxels, "trans"):
        return np.asarray(mapped_voxels.dim), np.asarray(mapped_voxels.trans)
    if hasattr(mapped_voxels, "space"):
        sp = space(mapped_voxels)
        return np.asarray(sp.dim), np.asarray(sp.trans)
    if isinstance(mapped_voxels, dict):
        return np.asarray(mapped_voxels["shape"]), np.asarray(mapped_voxels["affine"])
    shape, affine = mapped_voxels
    return np.asarray(shape), np.asarray(affine)


def output_aligned_space(mapped_voxels: Any, voxel_sizes: Any = None) -> Any:
    """Compute an axis-aligned output NeuroSpace covering mapped voxels."""
    from itertools import product
    from .neuro_space import NeuroSpace
    from .orientation import apply_affine

    shape, affine = _parse_mapped_voxels(mapped_voxels)
    n_axes = min(3, len(shape))
    spatial_shape = shape[:n_axes].astype(int)
    if voxel_sizes is None:
        out_vox = np.ones(n_axes, dtype=float)
    else:
        out_vox = np.asarray(voxel_sizes, dtype=float)
        if out_vox.ndim == 0:
            out_vox = np.repeat(float(out_vox), n_axes)
        else:
            out_vox = out_vox[:n_axes]
        if out_vox.size != n_axes or np.any(~np.isfinite(out_vox)) or np.any(out_vox <= 0):
            raise ValueError("voxel_sizes must be a positive scalar or match the number of spatial axes")
    corners = np.array(list(product(*[(0, int(n) - 1) for n in spatial_shape])), dtype=float)
    world = apply_affine(affine[:n_axes + 1, :n_axes + 1], corners)
    out_min = np.min(world, axis=0)
    out_max = np.max(world, axis=0)
    out_shape = np.ceil((out_max - out_min) / out_vox).astype(int) + 1
    trans_mat = np.eye(4)
    trans_mat[:n_axes, :n_axes] = np.diag(out_vox)
    trans_mat[:n_axes, 3] = out_min
    return NeuroSpace(out_shape, spacing=out_vox, origin=out_min, trans=trans_mat)


def deoblique(x: Any, method: str = "linear", engine: str = "nibabel") -> Any:
    """Resample an object to an axis-aligned space covering its current bounds."""
    from .neuro_space import NeuroSpace
    from .resample import resample_to

    target = output_aligned_space(x, voxel_sizes=float(np.min(spacing(x)[:3])))
    if isinstance(x, NeuroSpace):
        return target
    return resample_to(x, target, method=method, engine=engine)


def plot(x: Any, *args: Any, **kwargs: Any) -> Any:
    from .plotting import plot_neuro_vol

    return plot_neuro_vol(x, *args, **kwargs)


def image(x: Any, *args: Any, **kwargs: Any) -> Any:
    return plot(x, *args, **kwargs)


def plot_checkerboard(*args: Any, **kwargs: Any) -> Any:
    from .plotting import plot_overlay

    return plot_overlay(*args, **kwargs)


def plot_edge_overlay(*args: Any, **kwargs: Any) -> Any:
    from .plotting import plot_overlay

    return plot_overlay(*args, **kwargs)


def scale_fill_neuro(cmap: str = "viridis", limits: Any = None, **kwargs: Any) -> dict:
    return {"cmap": cmap, "limits": limits, **kwargs}


def theme_neuro(**kwargs: Any) -> dict:
    return dict(kwargs)


def annotate_orientation(*args: Any, **kwargs: Any) -> dict:
    return {"args": args, **kwargs}


from .neuro_vec import SparseNeuroVec as AbstractSparseNeuroVec

__all__ = [
    name for name in globals()
    if not name.startswith("_") and name not in {"Any", "Protocol", "Union", "runtime_checkable", "np"}
]
