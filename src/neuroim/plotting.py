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


def _validate_same_grid(reference, *others, message: str = "same NeuroSpace grid") -> None:
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


def _normalize_axis(axis: AxisLike = 2, *, along: Optional[int] = None) -> int:
    if along is not None:
        value = int(along) - 1
    elif isinstance(axis, str):
        key = axis.lower()
        if key not in _AXIS_ALIASES:
            raise ValueError("axis must be 0, 1, 2 or x/y/z, sagittal/coronal/axial")
        value = _AXIS_ALIASES[key]
    else:
        value = int(axis)
    if value not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    return value


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
                np.full_like(out, np.nan, dtype=float) if alpha_map is not None else None
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


def _display_slice(
    vol,
    *,
    axis: int,
    index: int,
    alpha_map: Optional[np.ndarray] = None,
) -> _DisplaySlice:
    arr = _as_3d_array(vol)
    slc = _slice_object(vol, axis, index)
    raw = np.asarray(slc.data if slc is not None else _slice_raw(arr, axis, index))
    oriented, extent, oriented_alpha = _orient_matrix(raw, slc=slc, alpha_map=alpha_map)
    return _DisplaySlice(
        data=oriented,
        extent=extent,
        axis=axis,
        index=int(index),
        plane=_PLANE_BY_AXIS[axis],
        alpha=oriented_alpha,
    )


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
    hi = float(cap) if cap is not None and np.isfinite(cap) else (
        float(np.max(vals)) if vals.size else knee + 1.0
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
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mags = np.abs(arr)
    mode = mode.lower()
    thresh = 0.0 if threshold is None else (
        float(threshold) if np.isscalar(threshold) else float(max(abs(t) for t in threshold))
    )

    if mode == "binary":
        alpha = np.ones_like(mags, dtype=float)
    elif mode == "proportional":
        denom = cap if np.isfinite(cap) and cap > 0 else 1.0
        alpha = mags / denom
    elif mode == "ramp":
        denom = cap - thresh
        if not np.isfinite(denom) or denom <= 0:
            denom = cap if np.isfinite(cap) and cap > 0 else 1.0
        alpha = (mags - thresh) / denom
    elif mode == "soft":
        lo, hi, gamma = _soft_alpha_params(mags, threshold=thresh, cap=cap, gamma=gamma)
        t = np.clip((mags - lo) / (hi - lo), 0.0, 1.0)
        alpha = t**gamma
    else:
        raise ValueError("alpha_mode must be 'binary', 'proportional', 'ramp', or 'soft'")

    alpha = np.clip(np.nan_to_num(alpha, nan=0.0), 0.0, 1.0)
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
        fig.suptitle(title)
    fig.tight_layout()


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
):
    """Plot a volume as a world-oriented slice montage."""
    axis = _normalize_axis(axis, along=along)
    data = _as_3d_array(vol)
    indices = _normalize_indices(
        zlevels, data.shape[axis], n_default=6, name="zlevels"
    )
    display_range = irange if irange is not None else range
    limits = _resolve_display_limits(
        _values_for_slices(vol, axis=axis, indices=indices), display_range
    )
    if bgvol is not None:
        _validate_same_grid(vol, bgvol)
        bg_limits = _resolve_display_limits(
            _values_for_slices(bgvol, axis=axis, indices=indices), "robust"
        )
    else:
        bg_limits = None

    fig, axes = _layout(len(indices), ncol, figsize)
    last = None
    for ax, idx in zip(axes, indices):
        if bgvol is not None:
            bg_display = _display_slice(bgvol, axis=axis, index=int(idx))
            _imshow(ax, bg_display, cmap=bgcmap, limits=bg_limits)

        display = _display_slice(vol, axis=axis, index=int(idx))
        panel_data = display.data
        panel_alpha = display.alpha
        if thresh != (0, 0):
            masked = np.ma.masked_where(
                (panel_data < thresh[0]) | (panel_data > thresh[1]), panel_data
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
        ax.set_title(f"Slice {int(idx)}")
        ax.axis("off")

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
) -> Tuple[plt.Figure, np.ndarray]:
    """Orthogonal three-plane view at a voxel or world coordinate."""
    if unit is not None:
        coord_space = unit
    coord = _coord_to_voxel(vol, coords, coord_space)
    limits = _resolve_display_limits(_as_3d_array(vol), range, probs=probs)

    if axes is not None:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes = np.asarray(axes).ravel()
    if len(axes) < 3:
        raise ValueError("axes must contain at least three Axes")

    specs = [(2, coord[2]), (0, coord[0]), (1, coord[1])]
    last = None
    for ax, (axis, idx) in zip(axes, specs):
        display = _display_slice(vol, axis=axis, index=idx)
        last = _imshow(ax, display, cmap=cmap, limits=limits)
        ax.set_title(display.plane)
        if crosshair:
            keep = [i for i in (0, 1, 2) if i != axis]
            center = np.asarray(coord)[keep]
            if _space_of(vol) is not None:
                center = _space_grid_to_coord(_space_of(vol).drop_dim(axis), center)[0]
            ax.axvline(float(center[0]), color="white", linewidth=0.6, alpha=0.75)
            ax.axhline(float(center[1]), color="white", linewidth=0.6, alpha=0.75)
        ax.axis("off")

    if title is not None:
        fig.suptitle(title)
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
) -> Tuple[plt.Figure, np.ndarray]:
    """Show multiple world-oriented slices along an axis."""
    axis = _normalize_axis(axis, along=along)
    data = _as_3d_array(vol)
    indices = _normalize_indices(
        zlevels, data.shape[axis], n_default=n_slices, name="zlevels"
    )
    limits = _resolve_display_limits(
        _values_for_slices(vol, axis=axis, indices=indices), range, probs=probs
    )

    if axes is not None:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
    else:
        fig, axes = _layout(len(indices), ncols, figsize)
    if len(axes) < len(indices):
        raise ValueError("axes must contain at least n_slices Axes")

    last = None
    for ax, idx in zip(axes, indices):
        display = _display_slice(vol, axis=axis, index=int(idx))
        last = _imshow(ax, display, cmap=cmap, limits=limits)
        ax.set_title(f"{_AXIS_NAME[axis]} = {int(idx)}")
        ax.axis("off")

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
    figsize: Tuple[float, float] = (12, 4),
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
) -> Tuple[plt.Figure, np.ndarray]:
    """Overlay a statistical map on a background volume.

    With ``zlevels=None`` the function preserves the historical Python ortho
    view.  Supplying ``zlevels`` activates the neuroim2-style slice montage.
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
    axis = _normalize_axis(axis, along=along)
    base_data = _as_3d_array(base_vol)
    overlay_data = _as_3d_array(overlay_vol)

    if zlevels is None:
        coord = _coord_to_voxel(base_vol, coords, coord_space)
        specs = [(2, coord[2]), (0, coord[0]), (1, coord[1])]
        if axes is not None:
            axes = np.asarray(axes).ravel()
            fig = axes[0].figure
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            axes = np.asarray(axes).ravel()
    else:
        indices = _normalize_indices(
            zlevels, base_data.shape[axis], n_default=9, name="zlevels"
        )
        specs = [(axis, int(idx)) for idx in indices]
        fig, axes = _layout(len(specs), ncol, figsize)

    bg_vals = np.concatenate([
        _slice_raw(base_data, ax, idx).ravel() for ax, idx in specs
    ])
    ov_vals = np.concatenate([
        _slice_raw(overlay_data, ax, idx).ravel() for ax, idx in specs
    ])
    bg_limits = _resolve_display_limits(bg_vals, bg_range, probs=probs)
    ov_limits = _resolve_display_limits(ov_vals, ov_range, probs=probs)

    finite_ov = ov_vals[np.isfinite(ov_vals)]
    signed = finite_ov.size > 0 and np.nanmin(finite_ov) < 0 < np.nanmax(finite_ov)
    if symmetric is None:
        symmetric = signed
    if signed and isinstance(overlay_cmap, str) and overlay_cmap == "hot":
        overlay_cmap = "blue-red"
    if symmetric:
        cap = abs(float(ov_cap)) if ov_cap is not None else max(abs(ov_limits[0]), abs(ov_limits[1]))
        if not np.isfinite(cap) or cap <= 0:
            cap = 1.0
        ov_limits = (-cap, cap)
    cap = abs(float(ov_cap)) if ov_cap is not None else max(abs(ov_limits[0]), abs(ov_limits[1]))
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    last = None
    for ax_obj, (ax_axis, idx) in zip(axes, specs):
        base_display = _display_slice(base_vol, axis=ax_axis, index=idx)
        _imshow(ax_obj, base_display, cmap=base_cmap, limits=bg_limits)

        raw_overlay = _slice_raw(overlay_data, ax_axis, idx)
        alpha_map = _alpha_for_overlay(
            raw_overlay,
            mode=alpha_mode,
            threshold=threshold,
            cap=cap,
            gamma=alpha_gamma,
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
            ax_obj.set_title(overlay_display.plane)
        else:
            ax_obj.set_title(f"{_AXIS_NAME[ax_axis]} = {idx}")
        ax_obj.axis("off")

    used = len(specs)
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
) -> Tuple[plt.Figure, np.ndarray]:
    """Registration-QC checkerboard slices for two aligned 3D volumes."""
    _validate_same_grid(base_vol, overlay_vol)
    axis = _normalize_axis(axis, along=along)
    base = _as_3d_array(base_vol)
    overlay = _as_3d_array(overlay_vol)
    indices = _normalize_indices(
        zlevels, base.shape[axis], n_default=6, name="zlevels"
    )
    if int(tile) <= 0:
        raise ValueError("`tile` must be positive")

    fig, axes = _layout(len(indices), int(ncol), figsize)
    for ax_obj, idx in zip(axes, indices):
        base_raw = _slice_raw(base, axis, int(idx))
        overlay_raw = _slice_raw(overlay, axis, int(idx))
        bg_limits = _resolve_display_limits(base_raw, bg_range, probs=probs)
        ov_limits = _resolve_display_limits(overlay_raw, ov_range, probs=probs)
        bg01 = np.clip((base_raw - bg_limits[0]) / (bg_limits[1] - bg_limits[0]), 0, 1)
        ov01 = np.clip((overlay_raw - ov_limits[0]) / (ov_limits[1] - ov_limits[0]), 0, 1)
        grid = np.indices(base_raw.shape)
        use_base = ((grid[0] // int(tile)) + (grid[1] // int(tile))) % 2 == 0
        panel = np.where(use_base, bg01, ov01)

        slc = _slice_object(base_vol, axis, int(idx))
        oriented, extent, _ = _orient_matrix(panel, slc=slc)
        display = _DisplaySlice(oriented, extent, axis, int(idx), _PLANE_BY_AXIS[axis])
        _imshow(ax_obj, display, cmap=cmap, limits=(0.0, 1.0))
        ax_obj.set_title(f"{_AXIS_NAME[axis]} = {int(idx)}")
        ax_obj.axis("off")

    _finalize_grid(fig, axes, len(indices), title=title)
    if draw:
        fig.canvas.draw_idle()
    return fig, axes[: len(indices)]


def _single_color_cmap(color: str) -> Colormap:
    return LinearSegmentedColormap.from_list("edge", [color, color])


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
    edge_alpha: float = 0.85,
) -> Tuple[plt.Figure, np.ndarray]:
    """Registration-QC edge overlay for fixed/moving edge maps."""
    _validate_same_grid(base_vol, edge_vol1, edge_vol2)
    axis = _normalize_axis(axis, along=along)
    base = _as_3d_array(base_vol)
    edge1 = _as_3d_array(edge_vol1)
    edge2 = _as_3d_array(edge_vol2)
    indices = _normalize_indices(
        zlevels, base.shape[axis], n_default=6, name="zlevels"
    )
    bg_limits = _resolve_display_limits(
        np.concatenate([_slice_raw(base, axis, int(i)).ravel() for i in indices]),
        bg_range,
        probs=probs,
    )
    edge_vals = np.concatenate([
        np.abs(_slice_raw(edge1, axis, int(i))).ravel() for i in indices
    ] + [
        np.abs(_slice_raw(edge2, axis, int(i))).ravel() for i in indices
    ])
    edge_limits = _resolve_display_limits(edge_vals, edge_range, probs=probs)
    edge_cap = max(abs(edge_limits[0]), abs(edge_limits[1]))
    if not np.isfinite(edge_cap) or edge_cap <= 0:
        edge_cap = 1.0

    fig, axes = _layout(len(indices), int(ncol), figsize)
    for ax_obj, idx in zip(axes, indices):
        base_display = _display_slice(base_vol, axis=axis, index=int(idx))
        _imshow(ax_obj, base_display, cmap=bg_cmap, limits=bg_limits)

        for edge_vol, edge_arr, color in (
            (edge_vol1, edge1, fixed_color),
            (edge_vol2, edge2, moving_color),
        ):
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
        ax_obj.set_title(f"{_AXIS_NAME[axis]} = {int(idx)}")
        ax_obj.axis("off")

    _finalize_grid(fig, axes, len(indices), title=title)
    if draw:
        fig.canvas.draw_idle()
    return fig, axes[: len(indices)]


def _plot_method(self, **kwargs):
    """Plot this NeuroVol using :func:`plot_neuro_vol`."""
    return plot_neuro_vol(self, **kwargs)


NeuroVol.plot = _plot_method
