"""Statistical operations for neuroimaging data.

This module provides functions for splitting, partitioning, and
analyzing neuroimaging data using various statistical approaches.

Statistical helpers for neuroimaging volumes and vector series.
"""

import numpy as np
from typing import Any, List, Dict, Callable, Optional, Union, cast
from .neuro_vol import NeuroVol, DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec, _scale_along
from .neuro_space import NeuroSpace
from .roi import ROIVec, ROIVol
from .clustered_neuro_vol import ClusteredNeuroVol


def split_blocks(
    x: Union[NeuroVol, NeuroVec],
    indices: Optional[np.ndarray] = None,
    block_ids: Optional[np.ndarray] = None,
    *,
    nblocks: Optional[int] = None,
    mask: Optional["LogicalNeuroVol"] = None,
) -> List[Union[NeuroVol, NeuroVec]]:
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

    """
    # Handle nblocks convenience parameter
    if nblocks is not None:
        spatial_shape = tuple(int(d) for d in x.shape[:3])
        if mask is not None:
            # Blocks over the mask voxels only (Fortran linear order)
            mask_arr = np.asarray(getattr(mask, "data", mask), dtype=bool)
            if mask_arr.shape != spatial_shape:
                raise ValueError("mask must match the spatial dimensions of x")
            all_indices = np.flatnonzero(mask_arr.ravel(order="F"))
        else:
            all_indices = np.arange(int(np.prod(spatial_shape)))
        n_voxels = len(all_indices)
        block_ids = np.repeat(np.arange(nblocks), int(np.ceil(n_voxels / nblocks)))[
            :n_voxels
        ]
        indices = all_indices

    if indices is None or block_ids is None:
        raise ValueError("Must provide either (indices, block_ids) or nblocks")

    if len(indices) != len(block_ids):
        raise ValueError("indices and block_ids must have same length")

    unique_blocks = np.unique(block_ids)
    blocks: List[Union[NeuroVol, NeuroVec]] = []

    for block_id in unique_blocks:
        # Get indices for this block
        block_mask = np.asarray(block_ids) == block_id
        # Sorted and unique: the block's sparse mask reads voxels in ascending
        # Fortran order, so the series must be gathered in that order too.
        block_indices = np.unique(np.asarray(indices)[block_mask])

        if isinstance(x, NeuroVec):
            # Extract data for this block
            from .neuro_vec import SparseNeuroVec

            # Dense 4-D values (a SparseNeuroVec's .data is time x voxels)
            data_4d = x.data if isinstance(x, DenseNeuroVec) else x.to_dense().data

            # Create array to hold time series for this block
            # Time is last dimension, so shape[-1] is n_timepoints
            block_series = np.zeros(
                (len(block_indices), x.shape[-1])
            )  # (n_voxels, n_timepoints)

            for i, idx in enumerate(block_indices):
                # Convert linear index to grid coordinates (spatial dimensions)
                coords = np.unravel_index(idx, x.shape[:3], order="F")
                # Extract time series for this voxel (time is last)
                block_series[i, :] = data_4d[coords[0], coords[1], coords[2], :]

            # Create mask for this block
            # For NeuroVec, time is always last dimension (x, y, z, time)
            spatial_dims = x.space.dim[:3]
            spatial_shape = x.shape[:3]

            mask_data = np.zeros(spatial_shape, dtype=bool)
            mask_flat = mask_data.ravel(order="F")
            mask_flat[block_indices] = True
            mask_data = mask_flat.reshape(spatial_shape, order="F")
            block_mask_vol = LogicalNeuroVol(
                mask_data, NeuroSpace(dim=spatial_dims, trans=x.space.affine)
            )

            # Create SparseNeuroVec for this block
            # SparseNeuroVec expects data as (n_timepoints, n_voxels), so transpose
            block_vec = SparseNeuroVec(
                data=block_series.T,
                space=x.space,
                mask=block_mask_vol,
                orientation="time_x_voxels",
            )
            blocks.append(block_vec)

        else:  # NeuroVol
            # Extract data for this block
            # Use Fortran order for consistency with R
            flat_data = x.data.ravel(order="F")
            block_data = flat_data[block_indices]

            # Create SparseNeuroVol for this block
            block_vol = SparseNeuroVol(
                data=block_data, space=x.space, indices=block_indices
            )
            blocks.append(block_vol)

    return blocks


def split_clusters(
    x: Union[NeuroVol, NeuroVec],
    clusters: Union[ClusteredNeuroVol, NeuroVol, None] = None,
    *,
    mask=None,
    k: Optional[int] = None,
) -> List[Union[ROIVol, ROIVec]]:
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
    list of ROIVol or ROIVec
        One ROI per cluster, voxels in Fortran (linear-index) order: an
        ROIVol of values for a NeuroVol, an ROIVec of time series (time x
        voxels) for a NeuroVec, as in neuroim2.

    """
    spatial_shape = tuple(int(d) for d in x.shape[:3])
    spatial_space = (
        x.space
        if isinstance(x, NeuroVol)
        else NeuroSpace(dim=spatial_shape, trans=x.space.affine)
    )
    if isinstance(x, NeuroVec):
        dense_4d = x.data if isinstance(x, DenseNeuroVec) else x.to_dense().data
    else:
        dense_3d = x.as_dense().data if hasattr(x, "as_dense") else x.data

    def _roi(flat_idx):
        # Voxels in Fortran (linear-index) order, as in cluster_map and R.
        coords = np.column_stack(np.unravel_index(flat_idx, spatial_shape, order="F"))
        if isinstance(x, NeuroVec):
            # One time series per voxel (time x voxels), as neuroim2's ROIVec
            series = dense_4d[coords[:, 0], coords[:, 1], coords[:, 2], :]
            return ROIVec(data=series.T, space=spatial_space, coords=coords)
        values = dense_3d[coords[:, 0], coords[:, 1], coords[:, 2]]
        return ROIVol(data=values, space=spatial_space, coords=coords)

    # Handle mask + k convenience API: auto-generate clusters
    if clusters is None and mask is not None and k is not None:
        mask_arr = np.asarray(getattr(mask, "data", mask), dtype=bool)
        mask_idx = np.flatnonzero(mask_arr.ravel(order="F"))
        # Simple partition into k roughly equal groups of mask voxels
        n_vox = len(mask_idx)
        return [_roi(mask_idx[i * n_vox // k : (i + 1) * n_vox // k]) for i in range(k)]

    if clusters is None:
        raise ValueError("Must provide clusters or (mask, k)")

    if not np.array_equal(np.asarray(clusters.space.dim[:3]), spatial_shape):
        raise ValueError(
            "x and clusters must have same space (spatial dimensions differ)"
        )
    spatial_space.compatible_with(clusters.space)

    # Cluster labels per voxel, in Fortran order
    if isinstance(clusters, ClusteredNeuroVol):
        label_flat = clusters.as_dense().data.ravel(order="F")
        in_cluster = np.zeros(label_flat.size, dtype=bool)
        in_cluster[clusters._mask_indices] = True
        unique_labels = np.unique(clusters.clusters)
    else:
        label_flat = np.asarray(clusters.data).ravel(order="F")
        in_cluster = label_flat > 0
        unique_labels = np.unique(label_flat[in_cluster])

    rois = []
    for label in unique_labels:
        if label == 0 and not isinstance(clusters, ClusteredNeuroVol):
            continue  # background
        rois.append(_roi(np.flatnonzero(in_cluster & (label_flat == label))))

    return rois


def _factor_levels(fac: np.ndarray) -> np.ndarray:
    """Sorted distinct levels of a factor (the row order of split results)."""
    return np.unique(fac)


def split_volumes(x: NeuroVec, fac: np.ndarray) -> Dict[Any, NeuroVec]:
    """Split a NeuroVec into sub-vectors by a per-volume factor.

    Parameters
    ----------
    x : NeuroVec
        The 4D data to split (any backend).
    fac : np.ndarray
        Factor with one entry per volume.

    Returns
    -------
    dict
        Maps each factor level to a NeuroVec holding that level's volumes,
        in their original order.
    """
    fac = np.asarray(fac)
    n_volumes = x.shape[3]
    if len(fac) != n_volumes:
        raise ValueError(
            f"Length of factor ({len(fac)}) must equal number of volumes ({n_volumes})"
        )

    result: Dict[Any, NeuroVec] = {}
    for level in _factor_levels(fac):
        level_indices = np.where(fac == level)[0]
        new_space = NeuroSpace(
            dim=list(x.space.dim[:3]) + [len(level_indices)],
            spacing=x.space.spacing,
            origin=x.space.origin,
            trans=x.space.trans,
            axes=x.space.axes,
        )
        if isinstance(x, DenseNeuroVec):
            # Time is the last axis of the (x, y, z, t) array
            level_vec: NeuroVec = DenseNeuroVec(x.data[..., level_indices], new_space)
        elif _is_sparse_vec(x):
            from .neuro_vec import SparseNeuroVec

            sparse_x = cast(SparseNeuroVec, x)
            # SparseNeuroVec data is (time, voxels)
            level_vec = SparseNeuroVec(
                sparse_x.data[level_indices, :],
                new_space,
                sparse_x.mask,
                orientation="time_x_voxels",
            )
        else:
            # Big, file-backed and mapped vectors read only the volumes needed
            level_vec = x.sub_vector(level_indices)
        result[level] = level_vec

    return result


def _is_sparse_vec(x: NeuroVec) -> bool:
    from .neuro_vec import SparseNeuroVec

    return isinstance(x, SparseNeuroVec)


def split_fill(
    x: Union[NeuroVol, NeuroVec],
    fac: np.ndarray,
    FUN: Optional[Callable[[np.ndarray], Any]] = None,
) -> Any:
    """Replace each group of voxels with a summary of that group.

    As neuroim2's ``split_fill``: the voxels of ``x`` are split into
    disjoint sets by ``fac``, ``FUN`` is applied to each set's values, and a
    volume of the same shape is returned with every voxel replaced by its
    set's result. ``FUN`` may return a scalar (broadcast over the set) or an
    array with one value per voxel in the set.

    Parameters
    ----------
    x : NeuroVol
        The volume to summarise.
    fac : np.ndarray
        Factor with one entry per voxel, either flat in the package's
        linear (Fortran) voxel order or with the volume's 3-D shape.
    FUN : callable
        Applied to each set's 1-D array of values.

    Returns
    -------
    DenseNeuroVol
        Volume in ``x``'s space holding the filled values.

    Notes
    -----
    Calling ``split_fill(vec, fac)`` on a NeuroVec with a per-volume factor
    is deprecated; use :func:`split_volumes`, which it forwards to.
    """
    if isinstance(x, NeuroVec):
        import warnings

        warnings.warn(
            "split_fill(NeuroVec, fac) is deprecated; it splits volumes by a "
            "per-volume factor, which is now split_volumes(). split_fill() "
            "follows neuroim2 and fills the voxels of a NeuroVol.",
            DeprecationWarning,
            stacklevel=2,
        )
        return split_volumes(x, fac)

    if FUN is None:
        raise TypeError("split_fill() requires FUN for a NeuroVol")

    shape = tuple(int(d) for d in x.shape[:3])
    fac = np.asarray(fac)
    if fac.shape == shape:
        fac = fac.ravel(order="F")
    n_voxels = int(np.prod(shape))
    if fac.ndim != 1 or len(fac) != n_voxels:
        raise ValueError(
            f"fac must have one entry per voxel ({n_voxels}), got shape {fac.shape}"
        )

    values = np.asarray(x.as_dense().data).ravel(order="F")
    groups = [(np.where(fac == level)[0], level) for level in _factor_levels(fac)]
    results = []
    for idx, level in groups:
        res = np.asarray(FUN(values[idx]))
        if res.size == len(idx):
            results.append(res.ravel())
        elif res.size == 1:
            results.append(np.full(len(idx), res.item()))
        else:
            raise ValueError(
                f"FUN returned {res.size} values for level {level!r}; expected "
                f"1 or {len(idx)} (the number of voxels in the level)"
            )

    out = np.zeros(
        n_voxels, dtype=np.result_type(*results) if results else values.dtype
    )
    for (idx, _), res in zip(groups, results):
        out[idx] = res
    return DenseNeuroVol(out.reshape(shape, order="F"), x.space)


def split_reduce(
    x: NeuroVec,
    fac: np.ndarray,
    FUN: Optional[Callable[[np.ndarray], Any]] = None,
) -> np.ndarray:
    """Summarise a NeuroVec within the levels of a factor.

    As neuroim2's ``split_reduce``, ``fac`` may have one entry per volume or
    one entry per voxel:

    - **Per volume:** for each level, ``FUN`` is applied to every voxel's
      values at that level's time points. The result is a
      ``(n_levels, n_voxels)`` array, with voxels in linear (Fortran) order.
    - **Per voxel:** for each level, ``FUN`` is applied across that level's
      voxels at every time point. The result is a ``(n_levels, n_time)``
      array, e.g. the mean time series of each parcel.

    When the volume and voxel counts are equal, ``fac`` is taken as per
    voxel, as in neuroim2.

    Parameters
    ----------
    x : NeuroVec
        The 4D data (any backend; non-dense backends are loaded into memory).
    fac : np.ndarray
        Factor, per volume or per voxel. A per-voxel factor may also have the
        volume's 3-D shape.
    FUN : callable, optional
        Reduces a 1-D array to a scalar. Defaults to the mean.

    Returns
    -------
    np.ndarray
        One row per level, in sorted level order (``np.unique(fac)``).
    """
    shape = tuple(int(d) for d in x.shape[:3])
    n_voxels = int(np.prod(shape))
    n_volumes = int(x.shape[3])
    fac = np.asarray(fac)
    if fac.shape == shape:
        fac = fac.ravel(order="F")
    if fac.ndim != 1 or len(fac) not in (n_voxels, n_volumes):
        raise ValueError(
            "length of fac must equal the number of voxels "
            f"({n_voxels}) or the number of volumes ({n_volumes})"
        )

    # voxels x time, voxels in linear (Fortran) order
    data_4d = x.data if isinstance(x, DenseNeuroVec) else x.to_dense().data
    mat = np.asarray(data_4d).reshape(n_voxels, n_volumes, order="F")

    def reduce(block: np.ndarray, axis: int) -> np.ndarray:
        if FUN is None:
            return cast(np.ndarray, block.mean(axis=axis))
        return np.apply_along_axis(FUN, axis, block)

    by_voxel = len(fac) == n_voxels
    rows = []
    for level in _factor_levels(fac):
        idx = np.where(fac == level)[0]
        if by_voxel:
            rows.append(reduce(mat[idx, :], axis=0))
        else:
            rows.append(reduce(mat[:, idx], axis=1))
    return np.vstack(rows)


def split_scale(
    x: NeuroVec, fac: np.ndarray, center: bool = True, scale: bool = True
) -> NeuroVec:
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

    """
    # Determine number of volumes (time dimension)
    # For all NeuroVec types, time is last dimension (x, y, z, time)
    n_volumes = x.shape[3]

    if len(fac) != n_volumes:
        raise ValueError(
            f"Length of factor ({len(fac)}) must equal number of volumes ({n_volumes})"
        )

    if isinstance(x, DenseNeuroVec):
        # Float copy (integer input would truncate the scaled values), reshaped
        # to 2D (n_voxels, n_timepoints)
        scaled_data = np.array(x.data, dtype=np.float64, copy=True)
        n_voxels = np.prod(x.shape[:3])
        data_2d = scaled_data.reshape(n_voxels, x.shape[3], order="F")

        # Process each level
        for level in np.unique(fac):
            level_mask = fac == level
            level_indices = np.where(level_mask)[0]

            # Get data for this level (select time points)
            level_data = data_2d[:, level_indices]

            # base::scale semantics (n - 1 denominator) per level
            level_data = _scale_along(level_data, axis=1, center=center, scale=scale)

            # Put back
            data_2d[:, level_indices] = level_data

        # Reshape back to 4D
        scaled_4d = data_2d.reshape(x.shape, order="F")
        return DenseNeuroVec(scaled_4d, x.space)
    elif not _is_sparse_vec(x):
        # Big, file-backed and mapped vectors: scale a dense copy
        return split_scale(x.to_dense(), fac, center=center, scale=scale)
    else:
        from .neuro_vec import SparseNeuroVec

        # SparseNeuroVec data is (time, voxels) - float copy, scaled in place
        scaled_data = np.array(x.data, dtype=np.float64, copy=True)

        for level in np.unique(fac):
            level_mask = fac == level
            level_indices = np.where(level_mask)[0]

            # Get data for this level (select time points)
            level_data = scaled_data[level_indices, :]

            level_data = _scale_along(level_data, axis=0, center=center, scale=scale)

            scaled_data[level_indices, :] = level_data

        sparse_x = cast(SparseNeuroVec, x)
        return SparseNeuroVec(
            scaled_data,
            x.space,
            sparse_x.mask,
            sparse_x.label,
            orientation="time_x_voxels",
        )


def partition(
    x: NeuroVol, k: int, method: str = "kmeans", mask: Optional[LogicalNeuroVol] = None
) -> ClusteredNeuroVol:
    """Partition a volume into k regions.

    Runs k-means on the world coordinates of the nonzero (or masked)
    voxels, giving k spatially compact parcels, as neuroim2 does. Voxel
    intensities only define which voxels take part.

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

    """
    from sklearn.cluster import KMeans

    # Get mask - either provided or non-zero voxels
    if mask is not None:
        mask_data = mask.data
    else:
        mask_data = x.data != 0

    mask_data = np.asarray(mask_data, dtype=bool)
    mask_indices = np.where(mask_data.ravel(order="F"))[0]
    # k-means on voxel coordinates (mm), not intensities
    coords = x.space.index_to_coord(mask_indices)[:, :3]

    if len(coords) < k:
        raise ValueError(
            f"Number of non-zero voxels ({len(coords)}) is less than k ({k})"
        )

    if method == "kmeans":
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(coords)

        # Create mask volume
        from .neuro_vol import LogicalNeuroVol

        mask_vol = LogicalNeuroVol(mask_data, x.space)

        # Create clustered volume
        clustered = ClusteredNeuroVol(
            mask=mask_vol, clusters=labels + 1  # 1-indexed like R
        )

        return clustered
    else:
        raise ValueError(f"Unknown method: {method}")


def map_values(x: NeuroVol, lookup: Dict[float, float]) -> NeuroVol:
    """Map values in a volume using a lookup table.

    Replaces values in the volume according to a lookup dictionary; values
    not in ``lookup`` become 0 and the result is a float ``DenseNeuroVol``,
    as in neuroim2.

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

    """
    # As neuroim2: every voxel is looked up (including the implicit zeros of
    # a sparse volume); values with no entry become 0; the result is a
    # float DenseNeuroVol (mapped values need not fit the input dtype).
    source = x.as_dense().data if hasattr(x, "as_dense") else np.asarray(x.data)
    source = np.asarray(source)
    mapped_data = np.zeros(source.shape, dtype=np.float64)
    for old_val, new_val in lookup.items():
        mapped_data[source == old_val] = new_val
    return DenseNeuroVol(mapped_data, x.space)


def centroids(
    x: ClusteredNeuroVol, method: str = "center_of_mass"
) -> Dict[int, np.ndarray]:
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

    """
    if method == "center_of_mass":
        # Use existing method
        return x.cluster_centers()
    elif method == "median":
        # Compute median coordinates
        centers = {}
        for cluster_id, full_volume_indices in x.cluster_map.items():
            # Convert to 3D coordinates
            coords = np.array(
                np.unravel_index(full_volume_indices, x.mask.shape, order="F")
            ).T
            # Median of each dimension
            center = np.median(coords, axis=0)
            centers[cluster_id] = center
        return centers
    else:
        raise ValueError(f"Unknown method: {method}")
