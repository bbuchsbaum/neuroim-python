"""Searchlight functionality for neuroimaging analysis.

This module provides searchlight iterators for analyzing local neighborhoods
of voxels within brain masks. Includes exhaustive, random, and clustered
searchlight implementations.

"""

import numpy as np
from typing import Iterator, Union, List, Optional
from joblib import Parallel, delayed
from .neuro_vol import NeuroVol, LogicalNeuroVol
from .neuro_space import NeuroSpace
from .roi import ROIVolWindow, ROIVol, spherical_roi
from .utils import LazyList


def _as_logical_mask(mask: Union[NeuroVol, LogicalNeuroVol]) -> LogicalNeuroVol:
    """Convert numeric masks while treating NaN/Inf as ineligible."""
    if isinstance(mask, LogicalNeuroVol):
        return mask
    data = mask.as_dense().data if hasattr(mask, "as_dense") else mask.data
    values = np.asarray(data)
    if values.ndim != 3:
        raise ValueError(f"searchlight mask must be 3D, got {values.ndim}D")
    return LogicalNeuroVol(np.isfinite(values) & (values != 0), mask.space)


def _validate_radius(mask: LogicalNeuroVol, radius: float) -> float:
    """Validate a physical radius before constructing a deferred iterator."""
    values = np.asarray(radius)
    if values.ndim != 0 or isinstance(radius, (bool, np.bool_)):
        raise ValueError("radius must be a finite numeric scalar")
    try:
        value = float(radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius must be a finite numeric scalar") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("radius must be a positive finite numeric scalar")
    minimum = float(np.min(np.asarray(mask.spacing[:3], dtype=float)))
    if value < minimum:
        raise ValueError(
            "radius is too small; it must be at least the smallest voxel spacing "
            f"({minimum:g})"
        )
    return value


def ellipsoid_shape(scales=(1.0, 1.0, 1.0), jitter: float = 0.0):
    """Return a shape callback for ellipsoidal resampled searchlights."""
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (3,) or np.any(scales <= 0):
        raise ValueError("scales must be a length-3 positive numeric sequence")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")

    def shape_fun(mask, center, radius, iter=0, nonzero=False):
        roi = spherical_roi(mask, center, radius, nonzero=False)
        coords = roi.coords
        if len(coords) == 0:
            return coords
        sc = scales
        if jitter > 0:
            sc = np.maximum(
                scales * (1 + np.random.normal(0, jitter, 3)), np.finfo(float).eps
            )
        centered = coords - np.asarray(center)
        keep = np.sum((centered * sc) ** 2, axis=1) <= radius**2
        return coords[keep]

    return shape_fun


def cube_shape():
    """Return a shape callback for cubic resampled searchlights."""

    def shape_fun(mask, center, radius, iter=0, nonzero=False):
        spc = np.asarray(mask.spacing[:3], dtype=float)
        half_width = np.ceil(radius / spc).astype(int)
        center_arr = np.asarray(center, dtype=int)
        axes = [
            np.arange(center_arr[i] - half_width[i], center_arr[i] + half_width[i] + 1)
            for i in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    return shape_fun


def blobby_shape(drop: float = 0.3, edge_fraction: float = 0.7):
    """Return a shape callback that randomly drops edge voxels from a sphere."""
    if drop < 0 or drop > 1:
        raise ValueError("drop must be in [0, 1]")
    if edge_fraction <= 0 or edge_fraction > 1:
        raise ValueError("edge_fraction must be in (0, 1]")

    def shape_fun(mask, center, radius, iter=0, nonzero=False):
        roi = spherical_roi(mask, center, radius, nonzero=False)
        coords = roi.coords
        if len(coords) == 0:
            return coords
        dist = np.sqrt(np.sum((coords - np.asarray(center)) ** 2, axis=1))
        threshold = np.quantile(dist, edge_fraction)
        edge = dist >= threshold
        drop_mask = edge & (np.random.random(len(dist)) < drop)
        return coords[~drop_mask]

    return shape_fun


def _find_center_index(coords: np.ndarray, center_grid: np.ndarray) -> int:
    """Return the row index of the center voxel in ROI coordinates.

    Falls back to 0 if the exact center coordinate is not present, matching
    `neuroim2`'s defensive behavior.
    """
    if len(coords) == 0:
        return 0

    matches = np.all(coords == center_grid, axis=1)
    center_rows = np.where(matches)[0]
    return int(center_rows[0]) if len(center_rows) else 0


def _mask_indices(mask: LogicalNeuroVol, nonzero_only: bool) -> np.ndarray:
    """Get linear indices for either mask voxels or all voxels.

    Uses Fortran-order indexing to match NeuroSpace mapping.
    """
    if nonzero_only:
        return np.where(mask.data.ravel(order="F"))[0]
    return np.arange(np.prod(mask.data.shape), dtype=int)


def _center_grid_from_index(mask: LogicalNeuroVol, center_idx: int) -> np.ndarray:
    """Convert a linear index to a center-grid vector."""
    center_grid = mask.space.index_to_grid(center_idx)
    if center_grid.ndim == 2:
        center_grid = center_grid[0]
    return center_grid.astype(int)


def searchlight_iterator(
    mask: Union[NeuroVol, LogicalNeuroVol],
    radius: float,
    eager: bool = False,
    nonzero: bool = False,
    cores: int = 0,
) -> Union[LazyList, List[ROIVolWindow]]:
    """Create an exhaustive searchlight iterator.

    This function generates an exhaustive searchlight iterator that returns
    ROIVolWindow objects for each searchlight sphere within the provided mask.
    The iterator visits every non-zero voxel in the mask as a potential center voxel.

    Parameters
    ----------
    mask : NeuroVol or LogicalNeuroVol
        A NeuroVol object representing the brain mask
    radius : float
        The radius (in mm) of the spherical searchlight
    eager : bool, optional
        Whether to eagerly compute the searchlight ROIs. Default is False,
        which uses lazy evaluation
    nonzero : bool, optional
        Whether to include only coordinates with nonzero values in the
        supplied mask. Default is False
    cores : int, optional
        Number of cores to use for parallel computation. Default is 0,
        which uses a single core. (Currently not implemented)

    Returns
    -------
    LazyList or list of ROIVolWindow
        A deferred_list object containing ROIVolWindow objects, each
        representing a searchlight region

    """
    mask = _as_logical_mask(mask)
    radius = _validate_radius(mask, radius)

    # Iterate over nonzero voxels as centers.
    indices = _mask_indices(mask, nonzero_only=True)
    centers = mask.space.index_to_grid(indices)

    def generate_searchlight(idx):
        """Generate a single searchlight ROI."""
        center_idx = int(indices[idx])
        center_grid = centers[idx]

        # Create spherical ROI around this center
        roi = spherical_roi(mask, center_grid, radius, fill=None, nonzero=nonzero)

        # Create ROIVolWindow with parent tracking
        return ROIVolWindow(
            # Mask values: 1 for in-mask members, 0 for out-of-mask ones when
            # nonzero=False (as in neuroim2).
            data=np.asarray(roi.data, dtype=float),
            space=roi.space,
            coords=roi.coords,
            parent_index=center_idx,
            center_index=_find_center_index(roi.coords, center_grid),
        )

    if eager:
        # Eagerly compute all searchlights
        if cores > 1:
            # Use parallel processing
            return Parallel(n_jobs=cores, prefer="threads")(
                delayed(generate_searchlight)(i) for i in range(len(indices))
            )
        else:
            # Single-threaded processing
            return [generate_searchlight(i) for i in range(len(indices))]
    else:
        # Return lazy list (single-threaded for now)
        return LazyList(generate_searchlight, len(indices))


def searchlight_coords(
    mask: Union[NeuroVol, LogicalNeuroVol],
    radius: float,
    nonzero: bool = False,
    cores: int = 0,
) -> LazyList:
    """Create an exhaustive searchlight iterator for voxel coordinates.

    This function generates an exhaustive searchlight iterator that returns
    voxel coordinates for each searchlight sphere within the provided mask.
    Centers are always the finite, non-zero voxels of the mask. ``nonzero``
    controls whether zero-valued voxels are retained as window members.

    Parameters
    ----------
    mask : NeuroVol or LogicalNeuroVol
        A NeuroVol object representing the brain mask
    radius : float
        The radius (in mm) of the spherical searchlight
    nonzero : bool, optional
        Whether to keep only nonzero mask voxels within each window. It does
        not change center eligibility. Default is False.
    cores : int, optional
        Number of cores to use for parallel computation. Default is 0,
        which uses a single core. (Currently not implemented)

    Returns
    -------
    LazyList
        A deferred_list object containing matrices of integer-valued
        voxel coordinates, each representing a searchlight region

    """
    mask = _as_logical_mask(mask)
    radius = _validate_radius(mask, radius)

    # Match the exhaustive window API: centers are always mask voxels;
    # ``nonzero`` only filters the membership of each generated window.
    indices = _mask_indices(mask, nonzero_only=True)
    centers = mask.space.index_to_grid(indices)

    def generate_coords(idx):
        """Generate coordinates for a single searchlight."""
        center_grid = centers[idx]

        # Create spherical ROI around this center
        roi = spherical_roi(mask, center_grid, radius, fill=None, nonzero=nonzero)

        # Return just the coordinates as a matrix
        return roi.coords

    # For coordinates, we can parallelize the generation if cores > 1
    if cores > 1:
        # Generate all coords in parallel and wrap in LazyList
        coords_list = Parallel(n_jobs=cores, prefer="threads")(
            delayed(generate_coords)(i) for i in range(len(indices))
        )
        # Return a pre-computed LazyList
        return LazyList(lambda i: coords_list[i], len(coords_list))
    else:
        # Return lazy list with deferred computation
        return LazyList(generate_coords, len(indices))


def random_searchlight(
    mask: Union[NeuroVol, LogicalNeuroVol], radius: float, nonzero: bool = True
) -> List[ROIVolWindow]:
    """Create a spherical random searchlight iterator.

    This function generates a spherical random searchlight iterator
    for analyzing local neighborhoods of voxels within a given radius
    in a brain mask. The algorithm randomly selects centers and removes
    all voxels within the searchlight from future consideration.

    Parameters
    ----------
    mask : NeuroVol or LogicalNeuroVol
        A NeuroVol object representing the brain mask
    radius : float
        The radius of the searchlight sphere (in mm)
    nonzero : bool, optional
        If True, keep only nonzero voxels in each searchlight (default True)

    Returns
    -------
    list of ROIVolWindow
        A list of ROIVolWindow objects, each representing a spherical
        searchlight region

    """
    mask = _as_logical_mask(mask)
    radius = _validate_radius(mask, radius)

    # Eligible centres are the nonzero mask voxels. Each window keeps only
    # voxels no earlier window claimed (plus out-of-mask voxels when
    # nonzero=False), so the windows partition the mask, as in neuroim2.
    mask_indices = _mask_indices(mask, nonzero_only=True)
    n_total = len(mask_indices)
    lookup = np.full(int(np.prod(mask.shape)), -1, dtype=np.int64)
    lookup[mask_indices] = np.arange(n_total)

    # Swap-with-last free list: free[:n_free] are unclaimed mask positions and
    # pos[v] is where v sits in it (-1 once claimed). Claiming costs O(batch),
    # not O(remaining), so the whole run is linear in the mask size.
    free = np.arange(n_total)
    pos = np.arange(n_total)
    n_free = n_total

    def claim(positions) -> None:
        nonlocal n_free
        for v in positions:
            p = pos[v]
            last = free[n_free - 1]
            free[p] = last
            pos[last] = p
            pos[v] = -1
            n_free -= 1

    searchlights = []
    while n_free > 0:
        center_pos = int(free[np.random.randint(n_free)])
        center_idx = int(mask_indices[center_pos])
        center_grid = _center_grid_from_index(mask, center_idx)

        roi = spherical_roi(mask, center_grid, radius, fill=None, nonzero=nonzero)
        roi_lookup = lookup[mask.space.grid_to_index(roi.coords)]
        in_mask = roi_lookup >= 0
        available = np.zeros(len(roi_lookup), dtype=bool)
        available[in_mask] = pos[roi_lookup[in_mask]] >= 0
        if not available.any():
            claim([center_pos])
            continue

        kept = available | (~in_mask & (not nonzero))
        coords = roi.coords[kept]
        claim(roi_lookup[available])

        searchlights.append(
            ROIVolWindow(
                data=np.asarray(roi.data, dtype=float)[kept],
                space=roi.space,
                coords=coords,
                parent_index=center_idx,
                center_index=_find_center_index(coords, center_grid),
            )
        )

    return searchlights


def clustered_searchlight(
    mask: Union[NeuroVol, LogicalNeuroVol],
    radius: float = 0,
    cvol: Optional[NeuroVol] = None,
    csize: Optional[int] = None,
) -> Iterator[ROIVol]:
    """Create a clustered searchlight iterator.

    This function generates searchlight regions based on pre-defined clusters
    or by performing k-means clustering on the mask coordinates.

    Parameters
    ----------
    mask : NeuroVol or LogicalNeuroVol
        A NeuroVol object representing the brain mask
    radius : float
        The radius parameter (currently not used in clustered searchlight)
    cvol : NeuroVol, optional
        A NeuroVol containing cluster labels for each voxel
    csize : int, optional
        Number of clusters to create using k-means if cvol is not provided

    Yields
    ------
    ROIVol
        ROIVol objects, each representing a cluster region

    Raises
    ------
    ValueError
        If neither cvol nor csize is provided

    """
    mask = _as_logical_mask(mask)

    if cvol is None and csize is None:
        raise ValueError("Must provide either 'cvol' or 'csize' argument")

    # Get nonzero voxel indices and coordinates
    mask_indices = _mask_indices(mask, nonzero_only=True)
    mask_coords = mask.space.index_to_coord(mask_indices)

    if cvol is None:
        # Perform k-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=csize, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(mask_coords)
    else:
        # Cluster labels from the provided volume: a ClusteredNeuroVol keeps
        # labels for its own mask voxels; any other volume has a label array.
        if hasattr(cvol, "clusters") and hasattr(cvol, "_mask_indices"):
            # Only voxels inside the clustered volume's own mask carry a label.
            in_cvol = np.zeros(int(np.prod(mask.shape)), dtype=bool)
            in_cvol[cvol._mask_indices] = True
            label_flat = np.zeros(int(np.prod(mask.shape)), dtype=int)
            label_flat[cvol._mask_indices] = np.asarray(cvol.clusters, dtype=int)
            mask_indices = mask_indices[in_cvol[mask_indices]]
        else:
            label_flat = np.asarray(cvol.data).ravel(order="F").astype(int)
        cluster_labels = label_flat[mask_indices]

    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)

    # Generate ROI for each cluster
    for cluster_id in unique_clusters:
        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = mask_indices[cluster_mask]

        # Convert to grid coordinates
        cluster_coords = mask.space.index_to_grid(cluster_indices)

        # Create ROIVol for this cluster
        cluster_data = np.ones(len(cluster_coords))
        yield ROIVol(space=mask.space, coords=cluster_coords, data=cluster_data)


def bootstrap_searchlight(
    mask: Union[NeuroVol, LogicalNeuroVol], radius: float = 8, iter: int = 100
) -> List[ROIVolWindow]:
    """Create a bootstrap searchlight iterator.

    This function generates a bootstrap searchlight iterator by randomly
    sampling center voxels with replacement.

    Parameters
    ----------
    mask : NeuroVol or LogicalNeuroVol
        A NeuroVol object representing the brain mask
    radius : float, optional
        The radius of the searchlight sphere in mm. Default is 8
    iter : int, optional
        Number of bootstrap iterations. Default is 100

    Returns
    -------
    list of ROIVolWindow
        A list of ROIVolWindow objects, each representing a spherical
        searchlight region

    """
    mask = _as_logical_mask(mask)
    radius = _validate_radius(mask, radius)

    # Get all nonzero voxel indices.
    mask_indices = _mask_indices(mask, nonzero_only=True)

    # Sample with replacement
    sample_indices = np.random.choice(mask_indices, size=iter, replace=True)

    searchlights = []
    for center_idx in sample_indices:
        center_grid = _center_grid_from_index(mask, center_idx)

        # Create spherical ROI
        roi = spherical_roi(mask, center_grid, radius, fill=1.0, nonzero=True)

        # Create ROIVolWindow
        searchlights.append(
            ROIVolWindow(
                data=roi.data,
                space=roi.space,
                coords=roi.coords,
                parent_index=center_idx,
                center_index=_find_center_index(roi.coords, center_grid),
            )
        )

    return searchlights
