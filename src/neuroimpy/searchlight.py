"""Searchlight functionality for neuroimaging analysis.

This module provides searchlight iterators for analyzing local neighborhoods
of voxels within brain masks. Includes exhaustive, random, and clustered
searchlight implementations.

Direct translation of R's neuroim2 searchlight functions.
"""

import numpy as np
from typing import Iterator, Union, List, Optional
from joblib import Parallel, delayed
from .neuro_vol import NeuroVol, LogicalNeuroVol
from .neuro_space import NeuroSpace
from .roi import ROIVolWindow, ROIVol, spherical_roi
from .utils import LazyList


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
        return np.where(mask.data.ravel(order='F'))[0]
    return np.arange(np.prod(mask.data.shape), dtype=int)


def _center_grid_from_index(mask: LogicalNeuroVol, center_idx: int) -> np.ndarray:
    """Convert a linear index to a center-grid vector."""
    center_grid = mask.space.index_to_grid(center_idx)
    if center_grid.ndim == 2:
        center_grid = center_grid[0]
    return center_grid.astype(int)


def searchlight_iterator(mask: Union[NeuroVol, LogicalNeuroVol], radius: float, 
                        eager: bool = False, nonzero: bool = False, 
                        cores: int = 0) -> Union[LazyList, List[ROIVolWindow]]:
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
        
    R Equivalent
    ------------
    neuroim2::searchlight
    """
    # Convert to LogicalNeuroVol if needed
    if not isinstance(mask, LogicalNeuroVol):
        mask = mask.as_logical()
    
    # Iterate over nonzero voxels as centers.
    indices = _mask_indices(mask, nonzero_only=True)
    centers = mask.space.index_to_grid(indices)
    
    def generate_searchlight(idx):
        """Generate a single searchlight ROI."""
        center_idx = int(indices[idx])
        center_grid = centers[idx]
        
        # Create spherical ROI around this center
        roi = spherical_roi(mask, center_grid, radius, 
                           fill=1.0, nonzero=nonzero)
        
        # Create ROIVolWindow with parent tracking
        return ROIVolWindow(
            data=roi.data,
            space=roi.space, 
            coords=roi.coords,
            parent_index=center_idx,
            center_index=_find_center_index(roi.coords, center_grid)
        )
    
    if eager:
        # Eagerly compute all searchlights
        if cores > 1:
            # Use parallel processing
            return Parallel(n_jobs=cores)(
                delayed(generate_searchlight)(i) for i in range(len(indices))
            )
        else:
            # Single-threaded processing
            return [generate_searchlight(i) for i in range(len(indices))]
    else:
        # Return lazy list (single-threaded for now)
        return LazyList(generate_searchlight, len(indices))


def searchlight_coords(mask: Union[NeuroVol, LogicalNeuroVol], radius: float,
                      nonzero: bool = False, cores: int = 0) -> LazyList:
    """Create an exhaustive searchlight iterator for voxel coordinates.
    
    This function generates an exhaustive searchlight iterator that returns
    voxel coordinates for each searchlight sphere within the provided mask.
    By default centers are taken from every voxel, and can be restricted to
    non-zero voxels with `nonzero=True`.
    
    Parameters
    ----------
    mask : NeuroVol or LogicalNeuroVol
        A NeuroVol object representing the brain mask
    radius : float
        The radius (in mm) of the spherical searchlight
    nonzero : bool, optional
        Whether to include only coordinates with nonzero values in the
        supplied mask. Default is False
    cores : int, optional
        Number of cores to use for parallel computation. Default is 0,
        which uses a single core. (Currently not implemented)
        
    Returns
    -------
    LazyList
        A deferred_list object containing matrices of integer-valued
        voxel coordinates, each representing a searchlight region
        
    R Equivalent
    ------------
    neuroim2::searchlight_coords
    """
    # Convert to LogicalNeuroVol if needed
    if not isinstance(mask, LogicalNeuroVol):
        mask = mask.as_logical()
    
    # Iterate over all voxels as centers when `nonzero` is False.
    # For nonzero centers, only iterate over nonzero voxels.
    # By default (nonzero=False), search all voxels as centers.
    # When nonzero=True, only nonzero voxels define centers.
    indices = _mask_indices(mask, nonzero_only=nonzero)
    centers = mask.space.index_to_grid(indices)
    
    def generate_coords(idx):
        """Generate coordinates for a single searchlight."""
        center_grid = centers[idx]
        
        # Create spherical ROI around this center
        roi = spherical_roi(mask, center_grid, radius, 
                           fill=1.0, nonzero=nonzero)
        
        # Return just the coordinates as a matrix
        return roi.coords
    
    # For coordinates, we can parallelize the generation if cores > 1
    if cores > 1:
        # Generate all coords in parallel and wrap in LazyList
        coords_list = Parallel(n_jobs=cores)(
            delayed(generate_coords)(i) for i in range(len(indices))
        )
        # Return a pre-computed LazyList
        return LazyList(lambda i: coords_list[i], len(coords_list))
    else:
        # Return lazy list with deferred computation
        return LazyList(generate_coords, len(indices))


def random_searchlight(mask: Union[NeuroVol, LogicalNeuroVol], 
                      radius: float,
                      nonzero: bool = True) -> List[ROIVolWindow]:
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
        
    R Equivalent
    ------------
    neuroim2::random_searchlight
    """
    # Convert to LogicalNeuroVol if needed
    if not isinstance(mask, LogicalNeuroVol):
        mask = mask.as_logical()
    
    # Get nonzero voxel indices as eligible centers.
    mask_indices = _mask_indices(mask, nonzero_only=True)
    remaining_indices = set(mask_indices)
    searchlights = []
    
    while remaining_indices:
        # Randomly select a center from remaining voxels
        center_idx = np.random.choice(list(remaining_indices))
        center_grid = _center_grid_from_index(mask, center_idx)
        
        # Create spherical ROI
        roi = spherical_roi(mask, center_grid, radius, 
                           fill=1.0, nonzero=nonzero)
        
        # Get indices of all voxels in this searchlight
        roi_indices = mask.space.grid_to_index(roi.coords)
        
        # Remove these indices from remaining set
        remaining_indices -= set(roi_indices)
        
        # Create ROIVolWindow
        searchlights.append(ROIVolWindow(
            data=roi.data,
            space=roi.space,
            coords=roi.coords,
            parent_index=center_idx,
            center_index=_find_center_index(roi.coords, center_grid)
        ))
    
    return searchlights


def clustered_searchlight(mask: Union[NeuroVol, LogicalNeuroVol], radius: float,
                         cvol: Optional[NeuroVol] = None, 
                         csize: Optional[int] = None) -> Iterator[ROIVol]:
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
        
    R Equivalent
    ------------
    neuroim2::clustered_searchlight
    """
    # Convert to LogicalNeuroVol if needed
    if not isinstance(mask, LogicalNeuroVol):
        mask = mask.as_logical()
    
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
        # Extract cluster labels from provided volume
        cluster_labels = cvol.data.ravel(order='F')[mask_indices].astype(int)
    
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
        yield ROIVol(
            space=mask.space,
            coords=cluster_coords,
            data=cluster_data
        )


def bootstrap_searchlight(mask: Union[NeuroVol, LogicalNeuroVol], 
                         radius: float = 8, iter: int = 100) -> List[ROIVolWindow]:
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
        
    R Equivalent
    ------------
    neuroim2::bootstrap_searchlight
    """
    # Convert to LogicalNeuroVol if needed
    if not isinstance(mask, LogicalNeuroVol):
        mask = mask.as_logical()
    
    # Get all nonzero voxel indices.
    mask_indices = _mask_indices(mask, nonzero_only=True)
    
    # Sample with replacement
    sample_indices = np.random.choice(mask_indices, size=iter, replace=True)
    
    searchlights = []
    for center_idx in sample_indices:
        center_grid = _center_grid_from_index(mask, center_idx)
        
        # Create spherical ROI
        roi = spherical_roi(mask, center_grid, radius, 
                           fill=1.0, nonzero=True)
        
        # Create ROIVolWindow
        searchlights.append(ROIVolWindow(
            data=roi.data,
            space=roi.space,
            coords=roi.coords,
            parent_index=center_idx,
            center_index=_find_center_index(roi.coords, center_grid)
        ))
    
    return searchlights
