"""Statistical operations for neuroimaging data.

This module provides functions for splitting, partitioning, and
analyzing neuroimaging data using various statistical approaches.

Direct translation of R's neuroim2 statistical functions.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Union
from .neuro_vol import NeuroVol, DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .neuro_space import NeuroSpace
from .roi import ROIVol
from .clustered_neuro_vol import ClusteredNeuroVol


def split_blocks(x: Union[NeuroVol, NeuroVec], indices: Optional[np.ndarray] = None,
                 block_ids: Optional[np.ndarray] = None, *,
                 nblocks: Optional[int] = None,
                 mask: Optional['LogicalNeuroVol'] = None) -> List[Union[NeuroVol, NeuroVec]]:
    """Split a NeuroVol or NeuroVec into blocks using indices.

    Creates separate volumes/vectors for each unique block ID, where each
    block contains only the data from voxels with that block ID.

    Parameters
    ----------
    x : NeuroVol or NeuroVec
        The data to split
    indices : np.ndarray, optional
        1D indices into the data array
    block_ids : np.ndarray, optional
        Block ID for each index (same length as indices)
    nblocks : int, optional
        Number of blocks to create (auto-generates indices and block_ids)
    mask : LogicalNeuroVol, optional
        Mask to use when generating blocks

    Returns
    -------
    list of NeuroVol or NeuroVec
        List of blocks, one for each unique block ID

    R Equivalent
    ------------
    neuroim2::split_blocks
    """
    # Handle nblocks convenience parameter
    if nblocks is not None:
        if isinstance(x, NeuroVec):
            n_voxels = np.prod(x.shape[:3])
            all_indices = np.arange(n_voxels)
            block_ids = np.repeat(np.arange(nblocks), int(np.ceil(n_voxels / nblocks)))[:n_voxels]
            indices = all_indices
        else:
            n_voxels = np.prod(x.shape)
            all_indices = np.arange(n_voxels)
            block_ids = np.repeat(np.arange(nblocks), int(np.ceil(n_voxels / nblocks)))[:n_voxels]
            indices = all_indices

    if indices is None or block_ids is None:
        raise ValueError("Must provide either (indices, block_ids) or nblocks")

    if len(indices) != len(block_ids):
        raise ValueError("indices and block_ids must have same length")
    
    unique_blocks = np.unique(block_ids)
    blocks = []
    
    for block_id in unique_blocks:
        # Get indices for this block
        block_mask = block_ids == block_id
        block_indices = indices[block_mask]
        
        if isinstance(x, NeuroVec):
            # Extract data for this block
            from .neuro_vec import SparseNeuroVec
            
            # Need to reshape x.data to 4D if it isn't already
            if x.data.ndim == 2:
                data_4d = x.data.reshape(x.shape)
            else:
                data_4d = x.data
            
            # Create array to hold time series for this block
            # Time is last dimension, so shape[-1] is n_timepoints
            block_series = np.zeros((len(block_indices), x.shape[-1]))  # (n_voxels, n_timepoints)
            
            for i, idx in enumerate(block_indices):
                # Convert linear index to grid coordinates (spatial dimensions)
                coords = np.unravel_index(idx, x.shape[:3], order='F')
                # Extract time series for this voxel (time is last)
                block_series[i, :] = data_4d[coords[0], coords[1], coords[2], :]
            
            # Create mask for this block
            # For NeuroVec, time is always last dimension (x, y, z, time)
            spatial_dims = x.space.dim[:3]
            spatial_shape = x.shape[:3]
                
            mask_data = np.zeros(spatial_shape, dtype=bool)
            mask_flat = mask_data.ravel(order='F')
            mask_flat[block_indices] = True
            mask_data = mask_flat.reshape(spatial_shape, order='F')
            mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=spatial_dims))
            
            # Create SparseNeuroVec for this block
            # SparseNeuroVec expects data as (n_timepoints, n_voxels), so transpose
            block_vec = SparseNeuroVec(
                data=block_series.T,
                space=x.space,
                mask=mask
            )
            blocks.append(block_vec)
            
        else:  # NeuroVol
            # Extract data for this block
            # Use Fortran order for consistency with R
            flat_data = x.data.ravel(order='F')
            block_data = flat_data[block_indices]
            
            # Create SparseNeuroVol for this block
            block_vol = SparseNeuroVol(
                data=block_data,
                space=x.space,
                indices=block_indices
            )
            blocks.append(block_vol)
    
    return blocks


def split_clusters(x: Union[NeuroVol, NeuroVec],
                   clusters: Union[ClusteredNeuroVol, NeuroVol, None] = None,
                   *, mask=None, k: Optional[int] = None) -> List[ROIVol]:
    """Split data into ROIs based on cluster labels.
    
    Creates a list of ROIVol objects, one for each cluster, containing
    the data values from the input volume/vector.
    
    Parameters
    ----------
    x : NeuroVol or NeuroVec
        The data to split
    clusters : ClusteredNeuroVol or NeuroVol
        Volume containing integer cluster labels
        
    Returns
    -------
    list of ROIVol
        List of ROI volumes, one for each cluster
        
    R Equivalent
    ------------
    neuroim2::split_clusters
    """
    # Handle mask + k convenience API: auto-generate clusters
    if clusters is None and mask is not None and k is not None:
        if hasattr(mask, 'data'):
            mask_arr = mask.data
        else:
            mask_arr = np.asarray(mask, dtype=bool)
        coords_all = np.column_stack(np.where(mask_arr))
        n_vox = len(coords_all)
        # Simple partition into k roughly equal groups
        labels = np.zeros(n_vox, dtype=int)
        for i in range(k):
            start = i * n_vox // k
            end = (i + 1) * n_vox // k
            labels[start:end] = i + 1
        rois = []
        spatial_space = x.space if isinstance(x, NeuroVol) else NeuroSpace(
            dim=x.space.dim[:3], spacing=x.space.spacing[:3], origin=x.space.origin[:3])
        for lab in range(1, k + 1):
            lab_mask = labels == lab
            lab_coords = coords_all[lab_mask]
            if isinstance(x, NeuroVec):
                data_values = []
                for coord in lab_coords:
                    data_values.append(np.mean(x[tuple(coord)]))
                data = np.array(data_values)
            else:
                data = x.data[lab_coords[:, 0], lab_coords[:, 1], lab_coords[:, 2]]
            roi = ROIVol(data=data, space=spatial_space, coords=lab_coords)
            rois.append(roi)
        return rois

    if clusters is None:
        raise ValueError("Must provide clusters or (mask, k)")

    # For NeuroVec, compare spatial dimensions only
    if isinstance(x, NeuroVec):
        # Get spatial dimensions of NeuroVec (first 3 dimensions)
        x_spatial_dim = x.space.dim[1:4] if x.shape[0] == x.space.dim[0] else x.space.dim[:3]
        if not np.array_equal(x_spatial_dim, clusters.space.dim):
            raise ValueError("x and clusters must have same spatial dimensions")
    else:
        if x.space != clusters.space:
            raise ValueError("x and clusters must have same space")

    # Get cluster labels
    if isinstance(clusters, ClusteredNeuroVol):
        # Use the labeled data directly
        unique_labels = np.unique(clusters.clusters)
        cluster_data = clusters.as_dense().data
    else:
        # Assume it's a regular volume with integer labels
        cluster_data = clusters.data
        unique_labels = np.unique(cluster_data[cluster_data > 0])

    rois = []

    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        # Get voxel coordinates for this cluster
        label_mask = cluster_data == label
        coords = np.column_stack(np.where(label_mask))
        
        # Extract data values
        if isinstance(x, NeuroVec):
            # For NeuroVec, extract time series for each voxel
            # and average across voxels in the cluster
            data_values = []
            for coord in coords:
                series = x[tuple(coord)]
                data_values.append(np.mean(series))
            data = np.array(data_values)
        else:
            # For NeuroVol, extract values directly
            data = x.data[label_mask]

        # Create ROIVol
        roi = ROIVol(data=data, space=x.space, coords=coords)
        rois.append(roi)
    
    return rois


def split_fill(x: NeuroVec, fac: np.ndarray) -> Dict[int, NeuroVec]:
    """Split a NeuroVec by factor levels and fill a new NeuroVec.
    
    Splits the input NeuroVec according to levels of a factor, creating
    a new NeuroVec for each level containing only the volumes from that level.
    
    Parameters
    ----------
    x : NeuroVec
        The 4D data to split
    fac : np.ndarray
        Factor array with length equal to number of volumes
        
    Returns
    -------
    dict
        Dictionary mapping factor levels to NeuroVec objects
        
    R Equivalent
    ------------
    neuroim2::split_fill
    """
    # Determine number of volumes (time dimension)
    # For all NeuroVec types, time is last dimension (x, y, z, time)
    n_volumes = x.shape[3]
        
    if len(fac) != n_volumes:
        raise ValueError(f"Length of factor ({len(fac)}) must equal number of volumes ({n_volumes})")
    
    levels = np.unique(fac)
    result = {}
    
    for level in levels:
        # Get indices for this level
        level_mask = fac == level
        level_indices = np.where(level_mask)[0]
        n_volumes = len(level_indices)
        
        # Extract subset of volumes
        if isinstance(x, DenseNeuroVec):
            # For dense, extract the relevant volumes
            # When data is 4D, subset along first dimension (time)
            subset_data = x.data[level_indices, :, :, :]
            
            # Create new space matching the subset shape
            new_dim = list(subset_data.shape)  # This will be [n_volumes, x, y, z]
            new_space = NeuroSpace(dim=new_dim, spacing=x.space.spacing, origin=x.space.origin,
                                  trans=x.space.trans, axes=x.space.axes)
            
            level_vec = DenseNeuroVec(subset_data, new_space)
        else:
            # For sparse, need to handle carefully
            from .neuro_vec import SparseNeuroVec
            # Extract subset of time points
            # SparseNeuroVec data is (time, voxels), so index first dimension
            subset_data = x.data[level_indices, :]
            
            # Create new space with updated time dimension
            new_dim = list(x.space.dim[:3]) + [n_volumes]
            new_space = NeuroSpace(dim=new_dim, spacing=x.space.spacing, origin=x.space.origin,
                                  trans=x.space.trans, axes=x.space.axes)
            
            level_vec = SparseNeuroVec(subset_data, new_space, x.mask)
            
        result[level] = level_vec
    
    return result


def split_reduce(x: NeuroVec, fac: np.ndarray, 
                 FUN: Callable[[np.ndarray], float]) -> NeuroVol:
    """Reduce a NeuroVec by applying a function across factor levels.
    
    Splits the NeuroVec by factor levels and applies a reduction function
    to each voxel's time series within each level, returning a volume
    where each voxel contains the reduced value.
    
    Parameters
    ----------
    x : NeuroVec
        The 4D data to reduce
    fac : np.ndarray
        Factor array with length equal to number of volumes
    FUN : callable
        Function to apply to each voxel's values within a level
        
    Returns
    -------
    NeuroVol
        Volume containing reduced values
        
    R Equivalent
    ------------
    neuroim2::split_reduce
    """
    # Determine number of volumes (time dimension)
    # For all NeuroVec types, time is last dimension (x, y, z, time)
    n_volumes = x.shape[3]
        
    if len(fac) != n_volumes:
        raise ValueError(f"Length of factor ({len(fac)}) must equal number of volumes ({n_volumes})")
    
    # Create 3D space for output volume
    # For all NeuroVec types, time is last dimension, so spatial dims are first 3
    vol_space = NeuroSpace(dim=x.space.dim[:3], spacing=x.space.spacing[:3], 
                          origin=x.space.origin[:3])
    
    # Initialize output array
    out_data = np.zeros(x.shape[:3])
    
    # For each voxel
    if isinstance(x, DenseNeuroVec):
        # Reshape data to proper 4D shape
        data_4d = x.data.reshape(x.shape)
        
        # Vectorized processing of all voxels
        # Reshape to (n_voxels, n_timepoints)
        n_voxels = np.prod(x.shape[:3])
        n_time = x.shape[3]
        data_2d = data_4d.reshape(n_voxels, n_time, order='F')
        
        # Process all voxels at once
        unique_levels = np.unique(fac)
        for vox_idx in range(n_voxels):
            voxel_series = data_2d[vox_idx, :]
            
            # Group by factor and apply function
            reduced_values = []
            for level in unique_levels:
                level_mask = fac == level
                level_values = voxel_series[level_mask]
                if len(level_values) > 0:
                    reduced_values.append(FUN(level_values))
            
            # Store mean of reduced values
            if reduced_values:
                # Convert voxel index back to 3D coordinates
                coords = np.unravel_index(vox_idx, out_data.shape, order='C')
                out_data[coords] = np.mean(reduced_values)
    else:
        # Handle sparse case
        from .neuro_vec import SparseNeuroVec
        # For sparse, process each stored voxel
        out_indices = []
        out_values = []
        
        # SparseNeuroVec data is (time, voxels)
        for vox_idx in range(x.data.shape[1]):
            voxel_series = x.data[:, vox_idx]
            
            # Group by factor and apply function
            reduced_values = []
            for level in np.unique(fac):
                level_mask = fac == level
                level_values = voxel_series[level_mask]
                if len(level_values) > 0:
                    reduced_values.append(FUN(level_values))
            
            # Store mean of reduced values
            if reduced_values:
                out_indices.append(x._lookup[vox_idx])
                out_values.append(np.mean(reduced_values))
        
        if out_indices:
            return SparseNeuroVol(out_values, vol_space, out_indices)
        else:
            return DenseNeuroVol(np.zeros(vol_space.dim), vol_space)
    
    return DenseNeuroVol(out_data, vol_space)


def split_scale(x: NeuroVec, fac: np.ndarray, 
                center: bool = True, scale: bool = True) -> NeuroVec:
    """Scale a NeuroVec within factor levels.
    
    Centers and/or scales the data within each level of a factor,
    returning a new NeuroVec with the transformed data.
    
    Parameters
    ----------
    x : NeuroVec
        The 4D data to scale
    fac : np.ndarray
        Factor array with length equal to number of volumes
    center : bool
        Whether to center (subtract mean) within each level
    scale : bool
        Whether to scale (divide by SD) within each level
        
    Returns
    -------
    NeuroVec
        Scaled NeuroVec
        
    R Equivalent
    ------------
    neuroim2::split_scale
    """
    # Determine number of volumes (time dimension)
    # For all NeuroVec types, time is last dimension (x, y, z, time)
    n_volumes = x.shape[3]
        
    if len(fac) != n_volumes:
        raise ValueError(f"Length of factor ({len(fac)}) must equal number of volumes ({n_volumes})")
    
    if isinstance(x, DenseNeuroVec):
        # Copy data and reshape to 2D (n_voxels, n_timepoints)
        scaled_data = x.data.copy()
        n_voxels = np.prod(x.shape[:3])
        data_2d = scaled_data.reshape(n_voxels, x.shape[3], order='F')
        
        # Process each level
        for level in np.unique(fac):
            level_mask = fac == level
            level_indices = np.where(level_mask)[0]
            
            # Get data for this level (select time points)
            level_data = data_2d[:, level_indices]
            
            if center:
                # Center by subtracting mean across time points
                level_mean = np.mean(level_data, axis=1, keepdims=True)
                level_data = level_data - level_mean
            
            if scale:
                # Scale by dividing by SD across time points
                level_std = np.std(level_data, axis=1, keepdims=True)
                # Avoid division by zero
                level_std[level_std == 0] = 1.0
                level_data = level_data / level_std
            
            # Put back
            data_2d[:, level_indices] = level_data
        
        # Reshape back to 4D
        scaled_4d = data_2d.reshape(x.shape, order='F')
        return DenseNeuroVec(scaled_4d, x.space)
    else:
        # Handle sparse case
        from .neuro_vec import SparseNeuroVec
        # SparseNeuroVec data is (time, voxels) - copy and scale in place
        scaled_data = x.data.copy()

        for level in np.unique(fac):
            level_mask = fac == level
            level_indices = np.where(level_mask)[0]

            # Get data for this level (select time points)
            level_data = scaled_data[level_indices, :]

            if center:
                level_mean = np.mean(level_data, axis=0, keepdims=True)
                level_data = level_data - level_mean

            if scale:
                level_std = np.std(level_data, axis=0, keepdims=True)
                level_std[level_std == 0] = 1.0
                level_data = level_data / level_std

            scaled_data[level_indices, :] = level_data

        return SparseNeuroVec(scaled_data, x.space, x.mask, x.label)


def partition(x: NeuroVol, k: int, 
              method: str = "kmeans", mask: Optional[LogicalNeuroVol] = None) -> ClusteredNeuroVol:
    """Partition a volume into k regions.
    
    Uses clustering algorithms to partition the brain volume into
    k distinct regions based on voxel values.
    
    Parameters
    ----------
    x : NeuroVol
        The volume to partition
    k : int
        Number of partitions/clusters
    method : str
        Clustering method ("kmeans" or others)
    mask : LogicalNeuroVol, optional
        Mask to restrict clustering to specific voxels
        
    Returns
    -------
    ClusteredNeuroVol
        Clustered volume with partition labels
        
    R Equivalent
    ------------
    neuroim2::partition
    """
    from sklearn.cluster import KMeans
    
    # Get mask - either provided or non-zero voxels
    if mask is not None:
        mask_data = mask.data
    else:
        mask_data = x.data != 0
    
    mask_indices = np.where(mask_data.ravel())[0]
    values = x.data.ravel()[mask_indices].reshape(-1, 1)
    
    if len(values) < k:
        raise ValueError(f"Number of non-zero voxels ({len(values)}) is less than k ({k})")
    
    if method == "kmeans":
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(values)
        
        # Create label array
        label_array = np.zeros(x.shape, dtype=int)
        label_array.ravel()[mask_indices] = labels + 1  # 1-indexed
        
        # Create mask volume
        from .neuro_vol import LogicalNeuroVol
        mask_vol = LogicalNeuroVol(mask_data, x.space)
        
        # Create clustered volume
        clustered = ClusteredNeuroVol(
            mask=mask_vol,
            clusters=labels + 1  # 1-indexed like R
        )
        
        return clustered
    else:
        raise ValueError(f"Unknown method: {method}")


def map_values(x: NeuroVol, lookup: Dict[float, float]) -> NeuroVol:
    """Map values in a volume using a lookup table.
    
    Replaces values in the volume according to a lookup dictionary.
    
    Parameters
    ----------
    x : NeuroVol
        The volume to transform
    lookup : dict
        Dictionary mapping old values to new values
        
    Returns
    -------
    NeuroVol
        Volume with mapped values
        
    R Equivalent
    ------------
    neuroim2::map_values
    """
    # Create copy of data
    mapped_data = x.data.copy()
    
    # Apply mapping
    for old_val, new_val in lookup.items():
        mapped_data[x.data == old_val] = new_val
    
    # Return same type of volume
    if isinstance(x, SparseNeuroVol):
        # For sparse, only map the non-zero values
        mapped_values = x.data.copy()
        for i, val in enumerate(x.data):
            if val in lookup:
                mapped_values[i] = lookup[val]
        return SparseNeuroVol(
            data=mapped_values,
            space=x.space,
            indices=x.indices
        )
    else:
        # Use type() to preserve the exact class, including subclasses
        return type(x)(mapped_data, x.space)


def centroids(x: ClusteredNeuroVol, method: str = "center_of_mass") -> Dict[int, np.ndarray]:
    """Calculate centroids for each cluster.
    
    Computes the centroid (center point) of each cluster using
    the specified method.
    
    Parameters
    ----------
    x : ClusteredNeuroVol
        Clustered volume
    method : str
        Method for computing centroids ("center_of_mass" or "median")
        
    Returns
    -------
    dict
        Dictionary mapping cluster IDs to centroid coordinates
        
    R Equivalent
    ------------
    neuroim2::centroids
    """
    if method == "center_of_mass":
        # Use existing method
        return x.cluster_centers()
    elif method == "median":
        # Compute median coordinates
        centers = {}
        # Get the full volume indices where mask is True
        mask_indices = np.where(x.mask.data.ravel())[0]
        
        for cluster_id, cluster_data_indices in x.cluster_map.items():
            # cluster_data_indices are indices into the masked data
            # We need to convert them to full volume indices
            full_volume_indices = mask_indices[cluster_data_indices]
            # Convert to 3D coordinates
            coords = np.array(np.unravel_index(full_volume_indices, x.mask.shape)).T
            # Median of each dimension
            center = np.median(coords, axis=0)
            centers[cluster_id] = center
        return centers
    else:
        raise ValueError(f"Unknown method: {method}")