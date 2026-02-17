import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, Union, Tuple, List
from .neuro_vol import NeuroVol

def plot_neuro_vol(vol: NeuroVol, cmap='gray', zlevels=None, irange=None, thresh=(0, 0), 
                   alpha=1, bgvol=None, bgcmap='gray', figsize=(12, 8)):
    """
    Plot a NeuroVol as a series of 2D slices.

    :param vol: The NeuroVol object to display.
    :param cmap: A colormap name or matplotlib colormap object.
    :param zlevels: The series of slice indices to display. If None, 6 evenly spaced slices are chosen.
    :param irange: The intensity range for color scaling. If None, the full range of the data is used.
    :param thresh: A 2-element tuple indicating the lower and upper transparency thresholds.
    :param alpha: The level of alpha transparency.
    :param bgvol: A background volume that serves as an image underlay (optional).
    :param bgcmap: A colormap name or matplotlib colormap object for the background layer.
    :param figsize: The size of the figure in inches (width, height).
    """
    if zlevels is None:
        zlevels = np.unique(np.round(np.linspace(0, vol.shape[2]-1, 6))).astype(int)
    
    if irange is None:
        irange = (np.min(vol.data), np.max(vol.data))

    n_slices = len(zlevels)
    n_rows = int(np.ceil(n_slices / 3))
    n_cols = min(n_slices, 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, z in zip(axes, zlevels):
        # Extract 2D slice data at position z
        # Note: using 1-based indexing for slice method
        slice_data = vol.data[:, :, z] if hasattr(vol, 'data') else vol[..., z]
        
        # For compatibility with tests that expect slice method to be called
        if hasattr(vol, 'slice'):
            slice_obj = vol.slice(z + 1)  # 1-based indexing
            slice_data = slice_obj.data if hasattr(slice_obj, 'data') else slice_obj
        
        if bgvol is not None:
            bg_slice_data = bgvol.data[:, :, z] if hasattr(bgvol, 'data') else bgvol[..., z]
            if hasattr(bgvol, 'slice'):
                bg_slice_obj = bgvol.slice(z + 1)
                bg_slice_data = bg_slice_obj.data if hasattr(bg_slice_obj, 'data') else bg_slice_obj
            ax.imshow(bg_slice_data.T, cmap=bgcmap, aspect='equal', origin='lower')
        
        # Apply thresholding if specified
        if thresh != (0, 0):
            mask = (slice_data < thresh[0]) | (slice_data > thresh[1])
            slice_data = np.ma.masked_where(mask, slice_data)
        
        im = ax.imshow(slice_data.T, cmap=cmap, aspect='equal', 
                       vmin=irange[0], vmax=irange[1], alpha=alpha, origin='lower')
        
        ax.set_title(f'Slice {z}')
        ax.axis('off')

    # Remove any unused subplots
    for ax in axes[n_slices:]:
        ax.remove()

    plt.tight_layout()
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Intensity')
    
    return fig

# ---------------------------------------------------------------------------
# Colormap utilities
# ---------------------------------------------------------------------------

_CMAP_ALIASES = {
    "neuroimaging": "gray",
    "activation": "hot",
    "coolwarm": "coolwarm",
    "diverging": "RdBu_r",
    "stat": "RdYlBu_r",
}


def resolve_cmap(name: str) -> plt.cm.ScalarMappable.__class__:
    """Resolve a colormap name string to a matplotlib Colormap object.

    Supports all standard matplotlib names as well as convenience aliases:
    ``"neuroimaging"`` -> ``"gray"``, ``"activation"`` -> ``"hot"``,
    ``"diverging"`` -> ``"RdBu_r"``, ``"stat"`` -> ``"RdYlBu_r"``.

    Parameters
    ----------
    name : str
        Colormap name or alias.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    resolved = _CMAP_ALIASES.get(name, name)
    return plt.colormaps[resolved]


def map_to_colors(
    data: np.ndarray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Convert a numeric array to an RGBA colour array.

    Parameters
    ----------
    data : np.ndarray
        Numeric array of any shape.
    cmap : str
        Matplotlib colormap name or alias accepted by :func:`resolve_cmap`.
    vmin, vmax : float, optional
        Value range for normalisation.  Defaults to data min/max.

    Returns
    -------
    np.ndarray
        Array of shape ``(*data.shape, 4)`` with RGBA float values in [0, 1].
    """
    colormap = resolve_cmap(cmap)
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = colormap(norm(data))
    return rgba


# ---------------------------------------------------------------------------
# Orthogonal (3-plane) plot
# ---------------------------------------------------------------------------


def plot_ortho(
    vol: np.ndarray,
    coords: Optional[Tuple[int, int, int]] = None,
    title: Optional[str] = None,
    cmap: str = "gray",
    figsize: Tuple[float, float] = (12, 4),
    axes: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Orthogonal 3-plane view (axial, sagittal, coronal).

    Parameters
    ----------
    vol : np.ndarray
        3-D volume array.
    coords : tuple of int, optional
        ``(x, y, z)`` voxel coordinates for the crosshair.  If *None* the
        volume centre is used.
    title : str, optional
        Super-title for the figure.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size in inches.
    axes : array-like of Axes, optional
        Three pre-existing axes to draw into.  When provided *figsize* is
        ignored.

    Returns
    -------
    (fig, axes) : tuple
        Matplotlib Figure and ndarray of three Axes.
    """
    data = np.asarray(vol.data if hasattr(vol, "data") else vol)
    if coords is None:
        coords = tuple(s // 2 for s in data.shape[:3])
    x, y, z = coords

    if axes is not None:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    # axial  – x-y plane at z
    axes[0].imshow(data[:, :, z].T, cmap=cmap, origin="lower", aspect="equal")
    axes[0].set_title("Axial")
    axes[0].axis("off")

    # sagittal – y-z plane at x
    axes[1].imshow(data[x, :, :].T, cmap=cmap, origin="lower", aspect="equal")
    axes[1].set_title("Sagittal")
    axes[1].axis("off")

    # coronal – x-z plane at y
    axes[2].imshow(data[:, y, :].T, cmap=cmap, origin="lower", aspect="equal")
    axes[2].set_title("Coronal")
    axes[2].axis("off")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------


def plot_montage(
    vol: np.ndarray,
    axis: int = 2,
    n_slices: int = 16,
    cmap: str = "gray",
    ncols: int = 4,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    axes: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Show multiple evenly-spaced slices along *axis* in a grid.

    Parameters
    ----------
    vol : np.ndarray
        3-D volume array.
    axis : int
        Axis along which to slice (0, 1, or 2).  Default 2 (axial).
    n_slices : int
        Number of slices to display.
    cmap : str
        Colormap name.
    ncols : int
        Number of columns in the grid.
    title : str, optional
        Super-title for the figure.
    figsize : tuple, optional
        Figure size.  If *None* it is computed automatically.
    axes : array-like of Axes, optional
        Pre-existing axes (must have at least *n_slices* elements).

    Returns
    -------
    (fig, axes) : tuple
    """
    data = np.asarray(vol.data if hasattr(vol, "data") else vol)
    n_total = data.shape[axis]
    indices = np.linspace(0, n_total - 1, n_slices, dtype=int)

    nrows = int(np.ceil(n_slices / ncols))

    if axes is not None:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
    else:
        if figsize is None:
            figsize = (3 * ncols, 3 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.asarray(axes).ravel()

    for i, idx in enumerate(indices):
        slc = [slice(None)] * 3
        slc[axis] = int(idx)
        slice_data = data[tuple(slc)]
        # Transpose for correct orientation when slicing along non-z axes
        axes[i].imshow(slice_data.T, cmap=cmap, origin="lower", aspect="equal")
        axes[i].set_title(f"{idx}")
        axes[i].axis("off")

    # Hide unused axes
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def plot_overlay(
    base_vol: np.ndarray,
    overlay_vol: np.ndarray,
    alpha: float = 0.5,
    base_cmap: str = "gray",
    overlay_cmap: str = "hot",
    threshold: Optional[float] = None,
    coords: Optional[Tuple[int, int, int]] = None,
    figsize: Tuple[float, float] = (12, 4),
    axes: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Orthogonal view with a statistical overlay.

    Parameters
    ----------
    base_vol, overlay_vol : np.ndarray
        3-D base and overlay volumes (same shape).
    alpha : float
        Overlay transparency.
    base_cmap, overlay_cmap : str
        Colormaps for base and overlay.
    threshold : float, optional
        If given, overlay voxels with ``abs(value) < threshold`` are hidden.
    coords : tuple of int, optional
        ``(x, y, z)`` crosshair.  Defaults to volume centre.
    figsize : tuple
        Figure size.
    axes : array-like of Axes, optional
        Three pre-existing axes.

    Returns
    -------
    (fig, axes) : tuple
    """
    base = np.asarray(base_vol.data if hasattr(base_vol, "data") else base_vol)
    over = np.asarray(overlay_vol.data if hasattr(overlay_vol, "data") else overlay_vol)

    if coords is None:
        coords = tuple(s // 2 for s in base.shape[:3])
    x, y, z = coords

    if axes is not None:
        axes = np.asarray(axes).ravel()
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    slice_specs = [
        (base[:, :, z], over[:, :, z], "Axial"),
        (base[x, :, :], over[x, :, :], "Sagittal"),
        (base[:, y, :], over[:, y, :], "Coronal"),
    ]

    for ax, (b_slc, o_slc, label) in zip(axes, slice_specs):
        ax.imshow(b_slc.T, cmap=base_cmap, origin="lower", aspect="equal")

        if threshold is not None:
            o_slc = np.ma.masked_where(np.abs(o_slc) < threshold, o_slc)

        ax.imshow(o_slc.T, cmap=overlay_cmap, origin="lower", aspect="equal", alpha=alpha)
        ax.set_title(label)
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


# Add the plot method to NeuroVol class
def _plot_method(self, **kwargs):
    """Plot this NeuroVol using plot_neuro_vol."""
    return plot_neuro_vol(self, **kwargs)

NeuroVol.plot = _plot_method