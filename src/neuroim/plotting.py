"""Matplotlib plotting helpers for neuroim spatial containers.

The implementation is intentionally matplotlib-native.  neuroim2's plotting
code is used as a behavioral reference for slice orientation, display ranges,
alpha maps, and registration-QC panels, but this module keeps Python indexing
and object boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from nibabel.orientations import io_orientation

from .neuro_slice import NeuroSlice
from .neuro_vol import NeuroVol

AxisLike = Union[int, str]
RangeArg = Union[str, Tuple[float, float], Sequence[float], None]
CmapLike = Union[str, Sequence[str], Colormap]


@dataclass(frozen=True)
class _DisplaySlice:
    data: np.ndarray
    extent: Tuple[float, float, float, float]
    axis: int
    index: int
    plane: str
    alpha: Optional[np.ndarray] = None
    native_axes: Tuple[int, int] = (0, 1)
    input_order: Tuple[int, int] = (0, 1)
    flipped: Tuple[bool, bool] = (False, False)
    display_dim: Tuple[int, int] = (0, 0)
    display_spacing: Tuple[float, float] = (1.0, 1.0)


_CMAP_ALIASES = {
    "neuroimaging": "gray",
    "activation": "hot",
    "coolwarm": "coolwarm",
    "diverging": "RdBu_r",
    "stat": "RdYlBu_r",
    # neuroim2/R-style aliases
    "grays": "gray",
    "grey": "gray",
    "greys": "gray",
    "blue-red": "RdBu_r",
    "coldhot": "coolwarm",
    "red-yellow": "hot",
}

_AXIS_ALIASES = {
    "x": 0,
    "i": 0,
    "sagittal": 0,
    "sag": 0,
    "y": 1,
    "j": 1,
    "coronal": 1,
    "cor": 1,
    "z": 2,
    "k": 2,
    "axial": 2,
    "axi": 2,
}

_PLANE_BY_AXIS = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
_AXIS_NAME = {0: "x", 1: "y", 2: "z"}

# Typographic defaults.  Panel titles are deliberately small so the imaging
# data, not the labels, dominates each montage cell.
_TITLE_FONTSIZE = 10
_SUPTITLE_KW = dict(fontsize=13, fontweight="medium")

# Fraction of the content bounding box added as breathing room when cropping.
_DEFAULT_CROP_MARGIN = 0.08


def resolve_cmap(name: CmapLike) -> Colormap:
    """Resolve a colormap name, alias, color list, or Colormap object."""
    if isinstance(name, Colormap):
        return name
    if isinstance(name, str):
        resolved = _CMAP_ALIASES.get(name.lower(), name)
        return plt.colormaps[resolved]
    if isinstance(name, Sequence):
        colors = list(name)
        if not colors:
            raise ValueError("color sequence must contain at least one color")
        return LinearSegmentedColormap.from_list("neuroim_custom", colors)
    raise TypeError("cmap must be a string, color sequence, or matplotlib Colormap")


def map_to_colors(
    data: np.ndarray,
    cmap: CmapLike = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    *,
    alpha: float = 1.0,
    alpha_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert numeric data to an RGBA array.

    NaN and infinite values are rendered transparent.
    """
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = resolve_cmap(cmap)(norm(arr))
    rgba[..., 3] = np.clip(float(alpha), 0.0, 1.0)

    if alpha_map is not None:
        amap = np.asarray(alpha_map, dtype=float)
        if amap.shape != arr.shape:
            raise ValueError("alpha_map must have the same shape as data")
        rgba[..., 3] *= np.clip(np.nan_to_num(amap, nan=0.0), 0.0, 1.0)

    rgba[..., 3] = np.where(np.isfinite(arr), rgba[..., 3], 0.0)
    return rgba


def _as_3d_array(vol) -> np.ndarray:
    """Extract a 3-D ndarray through neuroim's explicit array boundary."""
    if isinstance(vol, NeuroVol):
        data = vol.as_dense().as_array()
    elif hasattr(vol, "as_array") and callable(vol.as_array):
        data = vol.as_array()
    elif hasattr(vol, "data"):
        data = vol.data
    else:
        data = np.asarray(vol)

    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError("plotting functions require 3D volumes")
    return arr


def _space_of(vol):
    return getattr(vol, "space", None)


def _validate_same_grid(
    reference, *others, message: str = "same NeuroSpace grid"
) -> None:
    ref_shape = _as_3d_array(reference).shape
    ref_space = _space_of(reference)
    for other in others:
        if _as_3d_array(other).shape != ref_shape:
            raise ValueError(f"volumes must occupy the {message}")
        other_space = _space_of(other)
        if ref_space is not None and other_space is not None:
            try:
                ref_space.compatible_with(other_space)
            except Exception as exc:
                raise ValueError(f"volumes must occupy the {message}") from exc


_PLANE_NAMES = {"sagittal", "sag", "coronal", "cor", "axial", "axi"}


def _normalize_axis(
    axis: AxisLike = 2, *, along: Optional[int] = None, vol=None
) -> int:
    """Resolve ``axis``/``along`` to a native array axis.

    Integers and x/y/z (i/j/k) name native voxel axes. Plane names
    (sagittal/coronal/axial) name anatomical planes and are resolved through
    the volume's affine, so ``"axial"`` is the axial plane however the data
    are stored.
    """
    if along is not None:
        value = int(along) - 1
    elif isinstance(axis, str):
        key = axis.lower()
        if key not in _AXIS_ALIASES:
            raise ValueError("axis must be 0, 1, 2 or x/y/z, sagittal/coronal/axial")
        value = _AXIS_ALIASES[key]
        space = _space_of(vol) if vol is not None else None
        if key in _PLANE_NAMES and space is not None:
            world_axes = _spatial_orientation(space)[:, 0].astype(int)
            value = int(np.flatnonzero(world_axes == value)[0])
    else:
        value = int(axis)
    if value not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    return value


def _slice_title(vol, axis: int, index: int) -> str:
    """Panel label naming the world axis and slice position, e.g. ``z = -12 mm``."""
    space = _space_of(vol)
    if space is None:
        return f"{_AXIS_NAME[axis]} = {int(index)}"
    try:
        world_axis = int(_spatial_orientation(space)[axis, 0])
    except ValueError:
        return f"{_AXIS_NAME[axis]} = {int(index)}"
    center = (np.asarray(space.dim[:3], dtype=float) - 1) / 2
    center[axis] = index
    mm = float(space.grid_to_coord(center[None, :])[0, world_axis])
    return f"{_AXIS_NAME[world_axis]} = {mm:.0f} mm"


def _normalize_indices(
    indices: Optional[Iterable[int]],
    size: int,
    *,
    n_default: int,
    name: str,
) -> np.ndarray:
    if indices is None:
        n = min(int(n_default), int(size))
        out = np.unique(np.round(np.linspace(0, size - 1, n)).astype(int))
    else:
        out = np.asarray(list(indices), dtype=int)
    if out.size == 0:
        raise ValueError(f"`{name}` must contain at least one slice index")
    if np.any(out < 0) or np.any(out >= size):
        raise ValueError(f"`{name}` contains slices outside the volume")
    return out


def _raster_extent_from_centers(centers: np.ndarray) -> Tuple[float, float]:
    vals = np.unique(np.asarray(centers, dtype=float))
    vals = vals[np.isfinite(vals)]
    vals.sort()
    if vals.size == 0:
        return (0.0, 1.0)
    if vals.size == 1:
        return (float(vals[0] - 0.5), float(vals[0] + 0.5))
    step = float(np.median(np.diff(vals)))
    return (float(vals[0] - step / 2.0), float(vals[-1] + step / 2.0))


def _space_grid_to_coord(space, grid: np.ndarray) -> np.ndarray:
    """Project grid coordinates for display, tolerating 2-D slice spaces.

    Some slice spaces are represented by a 4x4 spatial transform even when
    their logical ``ndim`` is 2.  The core ``NeuroSpace.grid_to_coord`` is
    stricter than plotting needs, so we project through the available affine
    directly and return the first ``ndim`` display coordinates.
    """
    grid = np.atleast_2d(np.asarray(grid, dtype=float))
    try:
        return np.asarray(space.grid_to_coord(grid), dtype=float)
    except ValueError:
        trans = np.asarray(space.trans, dtype=float)
        if trans.shape == (4, 4) and space.ndim < 3:
            homo = np.zeros((grid.shape[0], 4), dtype=float)
            homo[:, : space.ndim] = grid[:, : space.ndim]
            homo[:, 3] = 1.0
            return (homo @ trans.T)[:, : space.ndim]
        raise


def _slice_raw(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    selector = [slice(None)] * 3
    selector[axis] = int(index)
    return np.asarray(data[tuple(selector)])


def _slice_object(vol, axis: int, index: int) -> Optional[NeuroSlice]:
    if not isinstance(vol, NeuroVol):
        return None
    from .neuro_slice import slice as extract_slice

    return extract_slice(vol, int(index), int(axis))


def _orient_matrix(
    mat: np.ndarray,
    *,
    slc: Optional[NeuroSlice] = None,
    alpha_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], Optional[np.ndarray]]:
    values = np.asarray(mat)
    if values.ndim != 2:
        raise ValueError("slice data must be 2D")

    if alpha_map is not None:
        alpha_map = np.asarray(alpha_map, dtype=float)
        if alpha_map.shape != values.shape:
            raise ValueError("alpha_map must have the same shape as slice data")

    if slc is None:
        xvals = np.arange(values.shape[0], dtype=float)
        yvals = np.arange(values.shape[1], dtype=float)
        out = values.T
        out_alpha = None if alpha_map is None else alpha_map.T
    else:
        flat_idx = np.arange(values.size)
        grid = slc.space.index_to_grid(flat_idx)
        coords = _space_grid_to_coord(slc.space, grid)
        xvals = np.unique(coords[:, 0])
        yvals = np.unique(coords[:, 1])
        xvals.sort()
        yvals.sort()

        # Exact rectilinear path.  Oblique/sheared spaces remain safe by falling
        # back to a bounded pixel grid below.
        if xvals.size == values.shape[0] and yvals.size == values.shape[1]:
            out = np.full((yvals.size, xvals.size), np.nan, dtype=float)
            out_alpha = (
                np.full_like(out, np.nan, dtype=float)
                if alpha_map is not None
                else None
            )
            col_idx = np.searchsorted(xvals, coords[:, 0])
            row_idx = np.searchsorted(yvals, coords[:, 1])
            out[row_idx, col_idx] = values.ravel(order="F")
            if out_alpha is not None:
                out_alpha[row_idx, col_idx] = alpha_map.ravel(order="F")
        else:
            out = values.T
            out_alpha = None if alpha_map is None else alpha_map.T
            xvals = coords[:, 0]
            yvals = coords[:, 1]

    xmin, xmax = _raster_extent_from_centers(xvals)
    ymin, ymax = _raster_extent_from_centers(yvals)
    return out, (xmin, xmax, ymin, ymax), out_alpha


def _spatial_orientation(space) -> np.ndarray:
    """Return a validated native-axis to anatomical-axis orientation."""
    orientation = io_orientation(np.asarray(space.affine, dtype=float))
    world_axes = orientation[:, 0]
    if (
        orientation.shape != (3, 2)
        or not np.all(np.isfinite(orientation))
        or len(set(world_axes.astype(int))) != 3
    ):
        raise ValueError(
            "volume affine must map its three voxel axes to distinct "
            "anatomical directions"
        )
    return orientation


def _display_from_native_matrix(
    mat: np.ndarray,
    *,
    space,
    axis: int,
    index: int,
    alpha_map: Optional[np.ndarray] = None,
) -> _DisplaySlice:
    """Orient one native voxel plane on a regular anatomical raster.

    The affine supplies a signed permutation from native voxel axes to their
    nearest world axes.  Oblique data remain on their native pixel lattice;
    this function only permutes/flips pixels and scales the regular display
    grid by the physical native-axis spacing.
    """
    values = np.asarray(mat)
    if values.ndim != 2:
        raise ValueError("slice data must be 2D")
    if alpha_map is not None:
        alpha_map = np.asarray(alpha_map, dtype=float)
        if alpha_map.shape != values.shape:
            raise ValueError("alpha_map must have the same shape as slice data")

    orientation = _spatial_orientation(space)
    native_axes = tuple(i for i in range(3) if i != axis)
    anatomical_axes = orientation[list(native_axes), 0].astype(int)
    input_order = tuple(int(i) for i in np.argsort(anatomical_axes))
    directions = orientation[np.asarray(native_axes, dtype=int)[list(input_order)], 1]
    flipped = tuple(bool(direction < 0) for direction in directions)

    canonical = np.transpose(values, input_order)
    canonical_alpha = (
        None if alpha_map is None else np.transpose(alpha_map, input_order)
    )
    for display_axis, should_flip in enumerate(flipped):
        if should_flip:
            canonical = np.flip(canonical, axis=display_axis)
            if canonical_alpha is not None:
                canonical_alpha = np.flip(canonical_alpha, axis=display_axis)

    display_dim = tuple(int(d) for d in canonical.shape)
    native_spacing = np.linalg.norm(np.asarray(space.affine)[:3, :3], axis=0)
    retained_spacing = native_spacing[list(native_axes)]
    display_spacing = tuple(float(retained_spacing[i]) for i in input_order)
    xvals = np.arange(display_dim[0], dtype=float) * display_spacing[0]
    yvals = np.arange(display_dim[1], dtype=float) * display_spacing[1]
    xmin, xmax = _raster_extent_from_centers(xvals)
    ymin, ymax = _raster_extent_from_centers(yvals)
    world_normal = int(orientation[axis, 0])

    return _DisplaySlice(
        data=canonical.T,
        extent=(xmin, xmax, ymin, ymax),
        axis=axis,
        index=int(index),
        plane=_PLANE_BY_AXIS[world_normal],
        alpha=None if canonical_alpha is None else canonical_alpha.T,
        native_axes=native_axes,
        input_order=input_order,
        flipped=flipped,
        display_dim=display_dim,
        display_spacing=display_spacing,
    )


def _display_slice(
    vol,
    *,
    axis: int,
    index: int,
    alpha_map: Optional[np.ndarray] = None,
) -> _DisplaySlice:
    arr = _as_3d_array(vol)
    raw = _slice_raw(arr, axis, index)
    space = _space_of(vol)
    if space is not None:
        return _display_from_native_matrix(
            raw,
            space=space,
            axis=axis,
            index=index,
            alpha_map=alpha_map,
        )

    oriented, extent, oriented_alpha = _orient_matrix(raw, alpha_map=alpha_map)
    return _DisplaySlice(
        data=oriented,
        extent=extent,
        axis=axis,
        index=int(index),
        plane=_PLANE_BY_AXIS[axis],
        alpha=oriented_alpha,
    )


def _slice_grid_to_display(
    display: _DisplaySlice, grid_coord: Sequence[float]
) -> np.ndarray:
    """Map a 3-D native grid coordinate into a displayed slice raster."""
    grid = np.asarray(grid_coord, dtype=float)
    if grid.shape != (3,) or not np.all(np.isfinite(grid)):
        raise ValueError("grid_coord must contain three finite voxel coordinates")
    retained = grid[list(display.native_axes)]
    shown = retained[list(display.input_order)].copy()
    for axis, should_flip in enumerate(display.flipped):
        if should_flip:
            shown[axis] = display.display_dim[axis] - 1 - shown[axis]
    return shown * np.asarray(display.display_spacing)


def _anatomical_specs(vol, coord: Sequence[int]) -> list[Tuple[int, int]]:
    """Native slice axes for Axial, Sagittal, Coronal display order."""
    space = _space_of(vol)
    if space is None:
        return [(2, int(coord[2])), (0, int(coord[0])), (1, int(coord[1]))]
    orientation = _spatial_orientation(space)
    native_for_world = {
        int(world_axis): int(native_axis)
        for native_axis, world_axis in enumerate(orientation[:, 0])
    }
    return [
        (native_for_world[world_axis], int(coord[native_for_world[world_axis]]))
        for world_axis in (2, 0, 1)
    ]


def _resolve_display_limits(values, range_arg: RangeArg = "robust", probs=(0.02, 0.98)):
    if range_arg is None:
        range_arg = "data"
    if isinstance(range_arg, str):
        mode = range_arg.lower()
        if mode not in {"robust", "data"}:
            raise ValueError("range must be 'robust', 'data', or a length-2 range")
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (0.0, 1.0)
        if mode == "robust":
            lo, hi = np.quantile(vals, probs)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = np.nanmin(vals), np.nanmax(vals)
        else:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
    else:
        rng = np.asarray(range_arg, dtype=float)
        if rng.shape != (2,) or not np.all(np.isfinite(rng)) or rng[0] == rng[1]:
            raise ValueError("numeric range must be two distinct finite values")
        lo, hi = float(np.min(rng)), float(np.max(rng))

    if lo == hi:
        hi = lo + 1.0
    return (float(lo), float(hi))


def _foreground_threshold(values) -> float:
    """Otsu threshold separating head/tissue from air (neuroim2 port)."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if x.size > 200_000:
        x = x[np.linspace(0, x.size - 1, 200_000).astype(int)]
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return float("nan")
    edges = np.linspace(lo, hi, 257)
    counts, _ = np.histogram(x, bins=edges)
    p = counts / counts.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    mu = np.cumsum(p * mids)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    between[~np.isfinite(between)] = -np.inf
    return float(edges[int(np.argmax(between)) + 1])


def _background_display_limits(
    values, range_arg: RangeArg = "robust", probs=(0.02, 0.98)
):
    """Display window for a structural background (neuroim2 #19).

    For ``"robust"`` the window is computed over foreground (head) voxels, so
    air does not drag the upper limit down and clip bright tissue in
    skull-stripped images; air maps to black.
    """
    if not (isinstance(range_arg, str) and range_arg.lower() == "robust"):
        return _resolve_display_limits(values, range_arg, probs=probs)
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    thr = _foreground_threshold(v)
    fg = v[v > thr] if np.isfinite(thr) else v
    if fg.size < 10:
        fg = v
    lo = float(np.quantile(v, probs[0]))
    if v.min() >= 0:
        lo = float(v.min())
    if np.isfinite(thr):
        air = v[v <= thr]
        if air.size >= 10:
            lo = max(lo, min(float(np.quantile(air, 0.9)), lo + (thr - lo) / 4))
    hi = float(np.quantile(fg, max(probs[1], 0.995)))
    hi = lo + (hi - lo) * 1.08
    if not np.isfinite(hi) or hi <= lo:
        hi = float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    return (float(lo), float(hi))


def _nice_ceiling(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return x
    p = 10 ** (np.floor(np.log10(x)) - 1)
    return float(np.ceil(x / p - 1e-9) * p)


def _scalar_threshold(threshold) -> float:
    if threshold is None:
        return 0.0
    if np.isscalar(threshold):
        return float(threshold)
    return 0.0


def _overlay_scale(
    values,
    ov_range: RangeArg = "robust",
    probs=(0.02, 0.98),
    threshold=None,
    symmetric: Optional[bool] = None,
    ov_cap: Optional[float] = None,
):
    """Overlay colour limits from the whole map (neuroim2 ``overlay_scale``).

    The scale does not depend on which slices are shown, always contains the
    threshold, starts at 0 for non-negative maps and rounds the cap up to a
    readable value. Returns ``(limits, cap, signed, symmetric)``.
    """
    v_all = np.asarray(values, dtype=float).ravel()
    fin = v_all[np.isfinite(v_all)]
    signed = fin.size > 0 and fin.min() < 0 < fin.max()
    symmetric = signed if symmetric is None else bool(symmetric)
    thresh = _scalar_threshold(threshold)
    auto_range = isinstance(ov_range, str)

    if not auto_range:
        lim = list(_resolve_display_limits(fin, ov_range, probs=probs))
    else:
        mode = str(ov_range).lower()
        if mode not in {"robust", "data"}:
            raise ValueError("range must be 'robust', 'data', or a length-2 range")
        v = fin[fin != 0]
        if v.size == 0:
            lim = [0.0, 1.0]
        elif mode == "data":
            lim = [float(v.min()), float(v.max())]
        else:
            if v.size > 500_000:
                v = v[np.linspace(0, v.size - 1, 500_000).astype(int)]
            hi_p = max(probs[1], 0.99)
            supra = v[np.abs(v) >= thresh] if thresh > 0 else v
            vv = supra if thresh > 0 and supra.size >= 10 else v
            lim = [float(np.quantile(vv, 1 - hi_p)), float(np.quantile(vv, hi_p))]
            if vv.min() >= 0:
                lim[0] = min(0.0, lim[0])
            if lim[0] == lim[1]:
                lim = [float(v.min()), float(v.max())]
        if thresh > 0 and v.size:
            if 0 < lim[1] < thresh:
                lim[1] = max(float(v.max()), thresh * 1.05)
            if -thresh < lim[0] < 0:
                lim[0] = min(float(v.min()), -thresh * 1.05)
        if lim[0] == lim[1]:
            lim = [lim[0] - 0.5, lim[1] + 0.5]

    cap = abs(float(ov_cap)) if ov_cap is not None else max(abs(lim[0]), abs(lim[1]))
    if auto_range and ov_cap is None:
        cap = _nice_ceiling(cap)
    if thresh > 0 and np.isfinite(cap) and cap <= thresh:
        cap = thresh * 1.25
    if symmetric and np.isfinite(cap) and cap > 0:
        lim = [-cap, cap]
    elif ov_cap is not None or auto_range:
        lim[1] = cap
        if auto_range and lim[0] > 0:
            lim[0] = 0.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0
    return (float(lim[0]), float(lim[1])), float(cap), bool(signed), bool(symmetric)


def _values_for_slices(vol, *, axis: int, indices: Sequence[int]) -> np.ndarray:
    arr = _as_3d_array(vol)
    return np.concatenate([_slice_raw(arr, axis, int(idx)).ravel() for idx in indices])


def _threshold_mask(values: np.ndarray, threshold) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if threshold is None:
        return ~np.isfinite(arr)
    if np.isscalar(threshold):
        return (~np.isfinite(arr)) | (np.abs(arr) < float(threshold))
    lo, hi = np.asarray(threshold, dtype=float)
    return (~np.isfinite(arr)) | ((arr > lo) & (arr < hi))


def _soft_alpha_params(mags, threshold=0.0, cap=None, gamma=None):
    vals = np.asarray(mags, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if threshold and threshold > 0:
        knee = float(threshold)
    elif vals.size:
        knee = float(np.median(vals))
    else:
        knee = 0.0
    hi = (
        float(cap)
        if cap is not None and np.isfinite(cap)
        else (float(np.max(vals)) if vals.size else knee + 1.0)
    )
    if not np.isfinite(hi) or hi <= knee:
        hi = knee + 1.0
    if gamma is None:
        supra = vals[vals > knee]
        if supra.size >= 10:
            t_med = float(np.median((supra - knee) / (hi - knee)))
            t_med = min(max(t_med, 1e-3), 0.999)
            gamma = np.log(0.2) / np.log(t_med)
        else:
            gamma = 2.0
        gamma = min(max(float(gamma), 1.5), 5.0)
    return knee, hi, float(gamma)


def _alpha_for_overlay(
    values: np.ndarray,
    *,
    mode: str,
    threshold,
    cap: float,
    gamma: Optional[float] = None,
    soft: Optional[Tuple[float, float, float]] = None,
    floor: Optional[float] = None,
) -> np.ndarray:
    """Per-voxel overlay opacity (neuroim2 ``overlay_alpha_fun``).

    With a threshold, every supra-threshold voxel keeps at least ``floor``
    opacity (default 0.6) in the graded modes, so nothing that passes the
    threshold disappears. ``soft`` is the ``(knee, hi, gamma)`` curve fitted
    once for the whole map; it is only re-fitted here when omitted.
    """
    arr = np.asarray(values, dtype=float)
    mags = np.abs(arr)
    mode = mode.lower()
    thresh = (
        0.0
        if threshold is None
        else (
            float(threshold)
            if np.isscalar(threshold)
            else float(max(abs(t) for t in threshold))
        )
    )
    thresh = thresh if thresh > 0 else 0.0
    cap = cap if np.isfinite(cap) and cap > 0 else 1.0
    fl = (0.6 if floor is None else float(floor)) if thresh > 0 else 0.0

    def graded(t):
        return fl + (1.0 - fl) * np.clip(t, 0.0, 1.0)

    span = max(cap - thresh, 1e-12)
    visible_floor = thresh
    with np.errstate(invalid="ignore"):
        if mode == "binary":
            alpha = np.ones_like(mags, dtype=float)
        elif mode == "proportional":
            alpha = (
                graded((mags - thresh) / span)
                if thresh > 0
                else np.minimum(mags / cap, 1.0)
            )
        elif mode == "ramp":
            alpha = (
                graded((mags - thresh) / span)
                if thresh > 0
                else np.clip(mags / cap, 0.0, 1.0)
            )
        elif mode == "soft":
            if soft is None:
                soft = _soft_alpha_params(mags, threshold=thresh, cap=cap, gamma=gamma)
            lo, hi, g = soft
            t = np.clip((mags - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
            alpha = fl + (1.0 - fl) * t**g
            visible_floor = max(thresh, lo)
        else:
            raise ValueError(
                "alpha_mode must be 'binary', 'proportional', 'ramp', or 'soft'"
            )
        visible = np.isfinite(mags) & (mags >= visible_floor) & (mags > 0)

    alpha = np.where(visible, np.clip(np.nan_to_num(alpha, nan=0.0), 0.0, 1.0), 0.0)
    alpha[_threshold_mask(arr, threshold)] = 0.0
    return alpha


def _imshow(
    ax,
    display: _DisplaySlice,
    *,
    cmap: CmapLike,
    limits,
    alpha: float = 1.0,
    interpolation: str = "nearest",
):
    if display.alpha is None:
        return ax.imshow(
            display.data,
            cmap=resolve_cmap(cmap),
            vmin=limits[0],
            vmax=limits[1],
            alpha=alpha,
            origin="lower",
            extent=display.extent,
            aspect="equal",
            interpolation=interpolation,
        )

    rgba = map_to_colors(
        display.data,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        alpha=alpha,
        alpha_map=display.alpha,
    )
    return ax.imshow(
        rgba,
        origin="lower",
        extent=display.extent,
        aspect="equal",
        interpolation=interpolation,
    )


def _layout(n_panels: int, ncol: int, figsize=None):
    if int(ncol) <= 0:
        raise ValueError("`ncol` must be positive")
    nrow = int(np.ceil(n_panels / int(ncol)))
    if figsize is None:
        figsize = (3.2 * int(ncol), 3.2 * nrow)
    fig, axes = plt.subplots(nrow, int(ncol), figsize=figsize, squeeze=False)
    return fig, axes.ravel()


def _finalize_grid(fig, axes, used: int, title=None):
    for ax in axes[used:]:
        ax.axis("off")
    if title is not None:
        fig.suptitle(title, **_SUPTITLE_KW)
        fig.tight_layout(pad=0.5, w_pad=0.4, h_pad=0.9, rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout(pad=0.5, w_pad=0.4, h_pad=0.9)


def _crop_margin(crop) -> Optional[float]:
    """Resolve the ``crop`` argument to a margin fraction (or ``None``)."""
    if crop is None or crop is False:
        return None
    if crop is True:
        return _DEFAULT_CROP_MARGIN
    margin = float(crop)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("`crop` must be True/False or a non-negative margin fraction")
    return margin


def _content_world_bbox(display: _DisplaySlice, floor: float):
    """World-coordinate bbox of finite, above-``floor`` pixels in a slice."""
    data = np.asarray(display.data, dtype=float)
    mask = np.isfinite(data) & (data > floor)
    if not mask.any():
        return None
    ny, nx = data.shape
    x0, x1, y0, y1 = display.extent
    dx = (x1 - x0) / nx if nx else 0.0
    dy = (y1 - y0) / ny if ny else 0.0
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    return (
        x0 + cols[0] * dx,
        x0 + (cols[-1] + 1) * dx,
        y0 + rows[0] * dy,
        y0 + (rows[-1] + 1) * dy,
    )


def _union_bbox(bboxes):
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return None
    return (
        min(b[0] for b in valid),
        max(b[1] for b in valid),
        min(b[2] for b in valid),
        max(b[3] for b in valid),
    )


def _apply_panel_crop(ax, bbox, margin, full_extent) -> None:
    """Zoom ``ax`` to ``bbox`` (plus an isotropic margin), clamped to data."""
    if bbox is None or margin is None:
        return
    x0, x1, y0, y1 = bbox
    ex0, ex1, ey0, ey1 = full_extent
    pad = margin * max(x1 - x0, y1 - y0)
    lo_x = max(min(ex0, ex1), x0 - pad)
    hi_x = min(max(ex0, ex1), x1 + pad)
    lo_y = max(min(ey0, ey1), y0 - pad)
    hi_y = min(max(ey0, ey1), y1 + pad)
    if hi_x > lo_x and hi_y > lo_y:
        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(lo_y, hi_y)


def _crop_montage_axes(axes, displays, used: int, floor: float, margin) -> None:
    """Crop every montage panel to one shared content bbox (consistent scale)."""
    if margin is None or not displays:
        return
    bbox = _union_bbox([_content_world_bbox(d, floor) for d in displays])
    if bbox is None:
        return
    full = _union_bbox([d.extent for d in displays]) or displays[0].extent
    for ax in axes[:used]:
        _apply_panel_crop(ax, bbox, margin, full)


def _ortho_width_ratios(displays) -> list:
    """Width ratios proportional to each plane's world width (tidy filmstrip)."""
    return [max(float(d.extent[1] - d.extent[0]), 1e-6) for d in displays]


def _coord_to_voxel(vol, coords, coord_space: str) -> Tuple[int, int, int]:
    data = _as_3d_array(vol)
    if coords is None:
        return tuple(int(s // 2) for s in data.shape[:3])
    coord_space = coord_space.lower()
    arr = np.asarray(coords, dtype=float)
    if arr.shape != (3,):
        raise ValueError("coords must be a length-3 coordinate")
    if coord_space in {"world", "mm"}:
        space = _space_of(vol)
        if space is None:
            raise ValueError("world/mm coordinates require a NeuroVol with NeuroSpace")
        arr = space.coord_to_grid(arr.reshape(1, -1))[0]
    elif coord_space not in {"voxel", "index"}:
        raise ValueError("coord_space must be 'voxel'/'index' or 'world'/'mm'")
    idx = tuple(int(i) for i in np.round(arr))
    if any(i < 0 or i >= data.shape[axis] for axis, i in enumerate(idx)):
        raise ValueError("coords must be inside the volume")
    return idx


def plot_neuro_vol(
    vol: NeuroVol,
    cmap: CmapLike = "gray",
    zlevels=None,
    irange: RangeArg = None,
    thresh=(0, 0),
    alpha=1,
    bgvol=None,
    bgcmap: CmapLike = "gray",
    figsize=(12, 8),
    *,
    axis: AxisLike = 2,
    along: Optional[int] = None,
    ncol: int = 3,
    colorbar: bool = True,
    range: RangeArg = None,
    title: Optional[str] = None,
    crop: Union[bool, float] = True,
):
    """Plot a volume as a world-oriented slice montage.

    ``crop`` (default ``True``) zooms every panel to a shared bounding box
    around the imaged content so the brain fills each cell instead of floating
    in empty background.  Pass ``False`` to keep the full field of view, or a
    float to set the margin fraction.
    """
    axis = _normalize_axis(axis, along=along, vol=vol)
    data = _as_3d_array(vol)
    indices = _normalize_indices(zlevels, data.shape[axis], n_default=6, name="zlevels")
    crop_margin = _crop_margin(crop)
    display_range = irange if irange is not None else range
    limits = _resolve_display_limits(
        _values_for_slices(vol, axis=axis, indices=indices), display_range
    )
    if bgvol is not None:
        _validate_same_grid(vol, bgvol)
        bg_limits = _background_display_limits(
            _values_for_slices(bgvol, axis=axis, indices=indices), "robust"
        )
    else:
        bg_limits = None

    fig, axes = _layout(len(indices), ncol, figsize)
    last = None
    crop_displays = []
    for ax, idx in zip(axes, indices):
        if bgvol is not None:
            bg_display = _display_slice(bgvol, axis=axis, index=int(idx))
            _imshow(ax, bg_display, cmap=bgcmap, limits=bg_limits)

        display = _display_slice(vol, axis=axis, index=int(idx))
        crop_displays.append(bg_display if bgvol is not None else display)
        panel_data = display.data
        panel_alpha = display.alpha
        if thresh[1] > thresh[0]:
            # As in neuroim2: values *inside* [lo, hi] are made transparent.
            masked = np.ma.masked_where(
                (panel_data >= thresh[0]) & (panel_data <= thresh[1]), panel_data
            )
            display = _DisplaySlice(
                data=masked,
                extent=display.extent,
                axis=display.axis,
                index=display.index,
                plane=display.plane,
                alpha=panel_alpha,
            )
        last = _imshow(ax, display, cmap=cmap, limits=limits, alpha=float(alpha))
        ax.set_title(f"Slice {int(idx)}", fontsize=_TITLE_FONTSIZE)
        ax.axis("off")

    crop_floor = bg_limits[0] if bgvol is not None else limits[0]
    _crop_montage_axes(axes, crop_displays, len(indices), crop_floor, crop_margin)
    _finalize_grid(fig, axes, len(indices), title=title)
    if colorbar and last is not None:
        fig.colorbar(last, ax=list(axes[: len(indices)]), shrink=0.8, label="Intensity")
    return fig


def plot_ortho(
    vol,
    coords: Optional[Tuple[int, int, int]] = None,
    title: Optional[str] = None,
    cmap: CmapLike = "gray",
    figsize: Tuple[float, float] = (12, 4),
    axes: Optional[np.ndarray] = None,
    *,
    coord_space: str = "voxel",
    unit: Optional[str] = None,
    range: RangeArg = "robust",
    probs=(0.02, 0.98),
    crosshair: bool = False,
    colorbar: bool = False,
    crop: Union[bool, float] = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Orthogonal three-plane view at a voxel or world coordinate.

    When this function creates the axes itself, the three planes are sized in
    proportion to their world extent and top-aligned, and (with ``crop=True``,
    the default) each panel is zoomed to its imaged content.
    """
    if unit is not None:
        coord_space = unit
    coord = _coord_to_voxel(vol, coords, coord_space)
    limits = _background_display_limits(_as_3d_array(vol), range, probs=probs)
    crop_margin = _crop_margin(crop)

    specs = _anatomical_specs(vol, coord)
    displays = [_display_slice(vol, axis=axis, index=idx) for axis, idx in specs]

    created = axes is None
    if axes is not None:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize,
            gridspec_kw={"width_ratios": _ortho_width_ratios(displays)},
        )
        axes = np.asarray(axes).ravel()
    if len(axes) < 3:
        raise ValueError("axes must contain at least three Axes")

    last = None
    for ax, (axis, idx), display in zip(axes, specs, displays):
        last = _imshow(ax, display, cmap=cmap, limits=limits)
        ax.set_title(display.plane, fontsize=_TITLE_FONTSIZE)
        if crosshair:
            center = _slice_grid_to_display(display, coord)
            ax.axvline(float(center[0]), color="white", linewidth=0.6, alpha=0.75)
            ax.axhline(float(center[1]), color="white", linewidth=0.6, alpha=0.75)
        ax.axis("off")
        if crop_margin is not None:
            _apply_panel_crop(
                ax,
                _content_world_bbox(display, limits[0]),
                crop_margin,
                display.extent,
            )
        if created:
            ax.set_anchor("N")

    if title is not None:
        fig.suptitle(title, **_SUPTITLE_KW)
    fig.tight_layout()
    if colorbar and last is not None:
        fig.colorbar(last, ax=list(axes[:3]), shrink=0.8, label="Intensity")
    return fig, axes[:3]


def plot_montage(
    vol,
    axis: AxisLike = 2,
    n_slices: int = 16,
    cmap: CmapLike = "gray",
    ncols: int = 4,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    axes: Optional[np.ndarray] = None,
    *,
    zlevels=None,
    along: Optional[int] = None,
    range: RangeArg = "robust",
    probs=(0.02, 0.98),
    colorbar: bool = False,
    crop: Union[bool, float] = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Show multiple world-oriented slices along an axis.

    With ``crop=True`` (default) the panels share a single content bounding box
    so the brain fills each cell at a consistent scale.
    """
    axis = _normalize_axis(axis, along=along, vol=vol)
    data = _as_3d_array(vol)
    indices = _normalize_indices(
        zlevels, data.shape[axis], n_default=n_slices, name="zlevels"
    )
    limits = _background_display_limits(
        _values_for_slices(vol, axis=axis, indices=indices), range, probs=probs
    )
    crop_margin = _crop_margin(crop)

    if axes is not None:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
    else:
        fig, axes = _layout(len(indices), ncols, figsize)
    if len(axes) < len(indices):
        raise ValueError("axes must contain at least n_slices Axes")

    last = None
    displays = []
    for ax, idx in zip(axes, indices):
        display = _display_slice(vol, axis=axis, index=int(idx))
        displays.append(display)
        last = _imshow(ax, display, cmap=cmap, limits=limits)
        ax.set_title(_slice_title(vol, axis, int(idx)), fontsize=_TITLE_FONTSIZE)
        ax.axis("off")

    _crop_montage_axes(axes, displays, len(indices), limits[0], crop_margin)
    _finalize_grid(fig, axes, len(indices), title=title)
    if colorbar and last is not None:
        fig.colorbar(last, ax=list(axes[: len(indices)]), shrink=0.8, label="Intensity")
    return fig, axes


def plot_overlay(
    base_vol=None,
    overlay_vol=None,
    alpha: float = 0.5,
    base_cmap: CmapLike = "gray",
    overlay_cmap: CmapLike = "hot",
    threshold: Optional[Union[float, Tuple[float, float]]] = None,
    coords: Optional[Tuple[int, int, int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    axes: Optional[np.ndarray] = None,
    *,
    coord_space: str = "voxel",
    unit: Optional[str] = None,
    background=None,
    overlay=None,
    bg_cmap: Optional[CmapLike] = None,
    ov_cmap: Optional[CmapLike] = None,
    zlevels=None,
    axis: AxisLike = 2,
    along: Optional[int] = None,
    ncol: int = 3,
    bg_range: RangeArg = "robust",
    ov_range: RangeArg = "robust",
    probs=(0.02, 0.98),
    ov_thresh: Optional[float] = None,
    alpha_mode: str = "binary",
    ov_alpha_mode: Optional[str] = None,
    symmetric: Optional[bool] = None,
    ov_symmetric: Optional[bool] = None,
    alpha_gamma: Optional[float] = None,
    ov_cap: Optional[float] = None,
    title: Optional[str] = None,
    colorbar: bool = False,
    crop: Union[bool, float] = True,
    alpha_floor: Optional[float] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Overlay a statistical map on a background volume.

    With ``zlevels=None`` the function preserves the historical Python ortho
    view.  Supplying ``zlevels`` activates the neuroim2-style slice montage.
    ``crop=True`` (default) zooms each panel to the background anatomy so the
    overlay is shown in context instead of floating in empty field of view.

    The overlay colour scale is computed from the whole map (not just the
    slices shown), always contains the threshold and starts at 0 for
    non-negative maps. With a threshold, graded ``alpha_mode`` values keep
    every supra-threshold voxel at least ``alpha_floor`` opaque (default
    0.6), and the ``"soft"`` curve is fitted once for the whole map.
    """
    if unit is not None:
        coord_space = unit
    if background is not None:
        base_vol = background
    if overlay is not None:
        overlay_vol = overlay
    if base_vol is None or overlay_vol is None:
        raise TypeError("plot_overlay requires base/background and overlay volumes")
    if bg_cmap is not None:
        base_cmap = bg_cmap
    if ov_cmap is not None:
        overlay_cmap = ov_cmap
    if ov_alpha_mode is not None:
        alpha_mode = ov_alpha_mode
    if ov_thresh is not None:
        threshold = ov_thresh
    if ov_symmetric is not None:
        symmetric = ov_symmetric

    _validate_same_grid(base_vol, overlay_vol)
    axis = _normalize_axis(axis, along=along, vol=base_vol)
    crop_margin = _crop_margin(crop)
    base_data = _as_3d_array(base_vol)
    overlay_data = _as_3d_array(overlay_vol)

    if zlevels is None:
        coord = _coord_to_voxel(base_vol, coords, coord_space)
        specs = _anatomical_specs(base_vol, coord)
        if axes is not None:
            axes = np.asarray(axes).ravel()
            fig = axes[0].figure
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize or (12, 4))
            axes = np.asarray(axes).ravel()
    else:
        indices = _normalize_indices(
            zlevels, base_data.shape[axis], n_default=9, name="zlevels"
        )
        specs = [(axis, int(idx)) for idx in indices]
        fig, axes = _layout(len(specs), ncol, figsize)

    bg_vals = np.concatenate(
        [_slice_raw(base_data, ax, idx).ravel() for ax, idx in specs]
    )
    bg_limits = _background_display_limits(bg_vals, bg_range, probs=probs)
    ov_limits, cap, signed, symmetric = _overlay_scale(
        overlay_data,
        ov_range,
        probs=probs,
        threshold=threshold,
        symmetric=symmetric,
        ov_cap=ov_cap,
    )
    if signed and isinstance(overlay_cmap, str) and overlay_cmap == "hot":
        overlay_cmap = "blue-red"
    soft = None
    if alpha_mode.lower() == "soft":
        soft = _soft_alpha_params(
            np.abs(overlay_data),
            threshold=_scalar_threshold(threshold),
            cap=cap,
            gamma=alpha_gamma,
        )

    last = None
    base_displays = []
    for ax_obj, (ax_axis, idx) in zip(axes, specs):
        base_display = _display_slice(base_vol, axis=ax_axis, index=idx)
        base_displays.append(base_display)
        _imshow(ax_obj, base_display, cmap=base_cmap, limits=bg_limits)

        raw_overlay = _slice_raw(overlay_data, ax_axis, idx)
        alpha_map = _alpha_for_overlay(
            raw_overlay,
            mode=alpha_mode,
            threshold=threshold,
            cap=cap,
            gamma=alpha_gamma,
            soft=soft,
            floor=alpha_floor,
        )
        overlay_display = _display_slice(
            overlay_vol, axis=ax_axis, index=idx, alpha_map=alpha_map
        )
        last = _imshow(
            ax_obj,
            overlay_display,
            cmap=overlay_cmap,
            limits=ov_limits,
            alpha=float(alpha),
        )
        if zlevels is None:
            ax_obj.set_title(overlay_display.plane, fontsize=_TITLE_FONTSIZE)
        else:
            ax_obj.set_title(
                _slice_title(base_vol, ax_axis, idx), fontsize=_TITLE_FONTSIZE
            )
        ax_obj.axis("off")

    used = len(specs)
    if crop_margin is not None:
        if zlevels is None:
            for ax_obj, base_display in zip(axes, base_displays):
                _apply_panel_crop(
                    ax_obj,
                    _content_world_bbox(base_display, bg_limits[0]),
                    crop_margin,
                    base_display.extent,
                )
        else:
            _crop_montage_axes(axes, base_displays, used, bg_limits[0], crop_margin)
    _finalize_grid(fig, axes, used, title=title)
    if colorbar and last is not None:
        sm = ScalarMappable(norm=Normalize(*ov_limits), cmap=resolve_cmap(overlay_cmap))
        sm.set_array([])
        fig.colorbar(sm, ax=list(axes[:used]), shrink=0.8, label="Overlay")
    return fig, axes[:used]


def plot_checkerboard(
    base_vol,
    overlay_vol,
    zlevels=None,
    tile: int = 8,
    ncol: int = 3,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    draw: bool = True,
    *,
    axis: AxisLike = 2,
    along: Optional[int] = None,
    cmap: CmapLike = "gray",
    bg_range: RangeArg = "robust",
    ov_range: RangeArg = "robust",
    probs=(0.02, 0.98),
    crop: Union[bool, float] = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Registration-QC checkerboard slices for two aligned 3D volumes."""
    _validate_same_grid(base_vol, overlay_vol)
    axis = _normalize_axis(axis, along=along, vol=base_vol)
    base = _as_3d_array(base_vol)
    overlay = _as_3d_array(overlay_vol)
    indices = _normalize_indices(zlevels, base.shape[axis], n_default=6, name="zlevels")
    if int(tile) <= 0:
        raise ValueError("`tile` must be positive")
    crop_margin = _crop_margin(crop)

    # One window per image across all panels, so contrast does not change
    # from tile to tile.
    bg_limits = _background_display_limits(
        _values_for_slices(base_vol, axis=axis, indices=indices), bg_range, probs=probs
    )
    ov_limits = _background_display_limits(
        _values_for_slices(overlay_vol, axis=axis, indices=indices),
        ov_range,
        probs=probs,
    )

    fig, axes = _layout(len(indices), int(ncol), figsize)
    displays = []
    for ax_obj, idx in zip(axes, indices):
        base_raw = _slice_raw(base, axis, int(idx))
        overlay_raw = _slice_raw(overlay, axis, int(idx))
        bg01 = np.clip((base_raw - bg_limits[0]) / (bg_limits[1] - bg_limits[0]), 0, 1)
        ov01 = np.clip(
            (overlay_raw - ov_limits[0]) / (ov_limits[1] - ov_limits[0]), 0, 1
        )
        grid = np.indices(base_raw.shape)
        use_base = ((grid[0] // int(tile)) + (grid[1] // int(tile))) % 2 == 0
        panel = np.where(use_base, bg01, ov01)

        space = _space_of(base_vol)
        if space is not None:
            display = _display_from_native_matrix(
                panel, space=space, axis=axis, index=int(idx)
            )
        else:
            oriented, extent, _ = _orient_matrix(panel)
            display = _DisplaySlice(
                oriented, extent, axis, int(idx), _PLANE_BY_AXIS[axis]
            )
        displays.append(display)
        _imshow(ax_obj, display, cmap=cmap, limits=(0.0, 1.0))
        ax_obj.set_title(
            _slice_title(base_vol, axis, int(idx)), fontsize=_TITLE_FONTSIZE
        )
        ax_obj.axis("off")

    # The panel is normalised to [0, 1] with background near 0, so a small
    # positive floor crops away the surrounding air.
    _crop_montage_axes(axes, displays, len(indices), 0.02, crop_margin)
    _finalize_grid(fig, axes, len(indices), title=title)
    if draw:
        fig.canvas.draw_idle()
    return fig, axes[: len(indices)]


def _single_color_cmap(color: str) -> Colormap:
    return LinearSegmentedColormap.from_list("edge", [color, color])


def _edge_contour_level(edge_vals: np.ndarray, edge_thresh, edge_cap: float) -> float:
    """Iso-level for edge contours: the threshold, else half the median edge."""
    if edge_thresh and float(edge_thresh) > 0:
        return float(edge_thresh)
    pos = edge_vals[np.isfinite(edge_vals) & (edge_vals > 0)]
    level = 0.5 * float(np.median(pos)) if pos.size else 0.5 * edge_cap
    if not np.isfinite(level) or level <= 0:
        level = 0.5 * edge_cap
    return level


def plot_edge_overlay(
    base_vol,
    edge_vol1,
    edge_vol2,
    zlevels=None,
    edge_thresh: float = 0.0,
    ncol: int = 3,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    draw: bool = True,
    *,
    axis: AxisLike = 2,
    along: Optional[int] = None,
    bg_cmap: CmapLike = "gray",
    fixed_color: str = "#00d5ff",
    moving_color: str = "#ff3b30",
    bg_range: RangeArg = "robust",
    edge_range: RangeArg = "robust",
    probs=(0.02, 0.98),
    edge_alpha: float = 0.9,
    style: str = "contour",
    edge_linewidth: float = 1.2,
    crop: Union[bool, float] = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Registration-QC edge overlay for fixed/moving edge maps.

    ``style="contour"`` (default) draws crisp iso-contours of each edge map,
    which reads as registration outlines even when the edge masks are coarse.
    ``style="fill"`` keeps the older proportional-alpha shading.
    """
    if style not in {"contour", "fill"}:
        raise ValueError("style must be 'contour' or 'fill'")
    _validate_same_grid(base_vol, edge_vol1, edge_vol2)
    axis = _normalize_axis(axis, along=along, vol=base_vol)
    crop_margin = _crop_margin(crop)
    base = _as_3d_array(base_vol)
    edge1 = _as_3d_array(edge_vol1)
    edge2 = _as_3d_array(edge_vol2)
    indices = _normalize_indices(zlevels, base.shape[axis], n_default=6, name="zlevels")
    bg_limits = _background_display_limits(
        np.concatenate([_slice_raw(base, axis, int(i)).ravel() for i in indices]),
        bg_range,
        probs=probs,
    )
    edge_vals = np.concatenate(
        [np.abs(_slice_raw(edge1, axis, int(i))).ravel() for i in indices]
        + [np.abs(_slice_raw(edge2, axis, int(i))).ravel() for i in indices]
    )
    edge_limits = _resolve_display_limits(edge_vals, edge_range, probs=probs)
    edge_cap = max(abs(edge_limits[0]), abs(edge_limits[1]))
    if not np.isfinite(edge_cap) or edge_cap <= 0:
        edge_cap = 1.0
    level = _edge_contour_level(edge_vals, edge_thresh, edge_cap)

    fig, axes = _layout(len(indices), int(ncol), figsize)
    base_displays = []
    for ax_obj, idx in zip(axes, indices):
        base_display = _display_slice(base_vol, axis=axis, index=int(idx))
        base_displays.append(base_display)
        _imshow(ax_obj, base_display, cmap=bg_cmap, limits=bg_limits)

        for edge_vol, edge_arr, color in (
            (edge_vol1, edge1, fixed_color),
            (edge_vol2, edge2, moving_color),
        ):
            if style == "contour":
                raw = np.abs(_slice_raw(edge_arr, axis, int(idx)))
                space = _space_of(edge_vol)
                if space is not None:
                    oriented_display = _display_from_native_matrix(
                        raw, space=space, axis=axis, index=int(idx)
                    )
                    oriented = oriented_display.data
                    extent = oriented_display.extent
                else:
                    oriented, extent, _ = _orient_matrix(raw)
                if np.any(np.isfinite(oriented) & (oriented >= level)):
                    ax_obj.contour(
                        np.nan_to_num(oriented, nan=0.0),
                        levels=[level],
                        extent=extent,
                        origin="lower",
                        colors=[color],
                        linewidths=float(edge_linewidth),
                        alpha=float(edge_alpha),
                    )
                continue

            raw_edge = np.abs(_slice_raw(edge_arr, axis, int(idx)))
            alpha_map = _alpha_for_overlay(
                raw_edge,
                mode="proportional",
                threshold=edge_thresh,
                cap=edge_cap,
            )
            edge_display = _display_slice(
                edge_vol, axis=axis, index=int(idx), alpha_map=alpha_map
            )
            _imshow(
                ax_obj,
                edge_display,
                cmap=_single_color_cmap(color),
                limits=edge_limits,
                alpha=float(edge_alpha),
            )
        ax_obj.set_title(
            _slice_title(base_vol, axis, int(idx)), fontsize=_TITLE_FONTSIZE
        )
        ax_obj.axis("off")

    _crop_montage_axes(axes, base_displays, len(indices), bg_limits[0], crop_margin)
    _finalize_grid(fig, axes, len(indices), title=title)
    if draw:
        fig.canvas.draw_idle()
    return fig, axes[: len(indices)]


def _plot_method(self, **kwargs):
    """Plot this NeuroVol using :func:`plot_neuro_vol`."""
    return plot_neuro_vol(self, **kwargs)


NeuroVol.plot = _plot_method
