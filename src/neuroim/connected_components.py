"""Connected components functionality for neuroimaging data.

This module provides functions for identifying and labeling connected
components in 3D neuroimaging data, supporting various connectivity patterns.

Connected-component utilities for labeled 3D neuroimaging data.
"""

import numpy as np
from scipy.ndimage import label, maximum_filter, distance_transform_edt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd

from .neuro_vol import NeuroVol, LogicalNeuroVol, DenseNeuroVol
from .neuro_space import NeuroSpace
from .clustered_neuro_vol import ClusteredNeuroVol


@dataclass
class ConnCompResult:
    """Result of connected components analysis."""

    index: ClusteredNeuroVol
    size: NeuroVol
    voxels: List[np.ndarray]
    cluster_table: Optional[pd.DataFrame] = None
    local_maxima: Optional[np.ndarray] = None


def conn_comp(
    x: NeuroVol,
    threshold: float = 0,
    cluster_table: bool = True,
    local_maxima: bool = True,
    local_maxima_dist: float = 15,
    connect: str = "26-connect",
) -> ConnCompResult:
    """Find connected components in an image.

    This function identifies and labels spatially connected regions in
    neuroimaging data, supporting both binary masks and thresholded volumes.

    Parameters
    ----------
    x : NeuroVol
        The image object
    threshold : float, optional
        Threshold defining lower intensity bound for image mask. Default is 0.
    cluster_table : bool, optional
        Whether to return cluster statistics table. Default is True.
    local_maxima : bool, optional
        Whether to compute local maxima within clusters. Default is True.
    local_maxima_dist : float, optional
        Minimum distance between local maxima in mm. Default is 15.
    connect : str, optional
        Connectivity pattern: "26-connect", "18-connect", or "6-connect".
        Default is "26-connect".

    Returns
    -------
    ConnCompResult
        Object containing:
        - index: ClusteredNeuroVol with cluster labels
        - size: NeuroVol with cluster sizes
        - voxels: List of cluster voxel coordinates
        - cluster_table: DataFrame with cluster statistics (if requested)
        - local_maxima: Array of local maxima coordinates (if requested)

    R Equivalent
    ------------
    neuroim2::conn_comp
    """
    # Apply threshold to create binary mask
    if isinstance(x, LogicalNeuroVol):
        mask_data = x.data
    else:
        mask_data = x.data > threshold

    # Get connectivity structure
    structure = _get_structure(connect)

    # Find connected components
    labeled_array, num_features = label(mask_data, structure=structure)

    # Create size array
    size_array = np.zeros_like(labeled_array, dtype=int)
    voxels_list = []

    # Compute sizes and collect voxel coordinates
    for i in range(1, num_features + 1):
        mask = labeled_array == i
        size = np.sum(mask)
        size_array[mask] = size

        # Get voxel coordinates (grid indices)
        coords = np.column_stack(np.where(mask))
        voxels_list.append(coords)

    # Create ClusteredNeuroVol for index
    mask_vol = LogicalNeuroVol(mask_data, x.space)
    cluster_labels = labeled_array[mask_data]
    index_vol = ClusteredNeuroVol(mask_vol, cluster_labels)

    # Create NeuroVol for sizes
    size_vol = DenseNeuroVol(size_array.astype(float), x.space)

    result = ConnCompResult(index=index_vol, size=size_vol, voxels=voxels_list)

    # Compute cluster table if requested
    if cluster_table and num_features > 0:
        result.cluster_table = _compute_cluster_table(
            x, labeled_array, voxels_list, x.space
        )

    # Find local maxima if requested
    if local_maxima and num_features > 0:
        result.local_maxima = _find_local_maxima(
            x, labeled_array, local_maxima_dist, x.space
        )

    return result


def conn_comp_3D(
    mask: Union[np.ndarray, LogicalNeuroVol], connect: str = "26-connect"
) -> Dict[str, np.ndarray]:
    """Extract connected components from a 3D binary mask.

    Identifies and labels connected components in a 3D binary mask using
    a two-pass algorithm. The function supports different connectivity
    constraints and returns both component indices and their sizes.

    Parameters
    ----------
    mask : np.ndarray or LogicalNeuroVol
        A 3D logical array or LogicalNeuroVol representing the binary mask
    connect : str, optional
        Connectivity constraint: "26-connect", "18-connect", or "6-connect".
        Default is "26-connect".

    Returns
    -------
    dict
        Dictionary with two components:
        - 'index': 3D array of integers with cluster indices (0 = background)
        - 'size': 3D array of integers with cluster sizes (0 = background)

    R Equivalent
    ------------
    neuroim2::conn_comp_3D
    """
    # Extract mask data if LogicalNeuroVol
    if isinstance(mask, LogicalNeuroVol):
        mask_data = mask.data
    else:
        mask_data = mask

    if mask_data.ndim != 3:
        raise ValueError("Mask must be 3D")

    if mask_data.dtype != bool:
        raise ValueError("Mask must be boolean/logical")

    # Get connectivity structure
    structure = _get_structure(connect)

    # Find connected components
    labeled_array, num_features = label(mask_data, structure=structure)

    # Create size array
    size_array = np.zeros_like(labeled_array, dtype=int)

    # Compute sizes for each component
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        component_size = np.sum(component_mask)
        size_array[component_mask] = component_size

    return {"index": labeled_array, "size": size_array}


def _get_structure(connectivity: str) -> np.ndarray:
    """Get the structuring element for the specified connectivity.

    Parameters
    ----------
    connectivity : str
        Connectivity pattern: "6-connect", "18-connect", or "26-connect"

    Returns
    -------
    np.ndarray
        3x3x3 boolean array defining the connectivity structure
    """
    if connectivity == "6-connect":
        # Only face-adjacent voxels
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = True
        structure[0, 1, 1] = True
        structure[2, 1, 1] = True
        structure[1, 0, 1] = True
        structure[1, 2, 1] = True
        structure[1, 1, 0] = True
        structure[1, 1, 2] = True
        return structure

    elif connectivity == "18-connect":
        # Face and edge-adjacent voxels
        structure = np.zeros((3, 3, 3), dtype=bool)
        # Add center
        structure[1, 1, 1] = True
        # Add faces (6)
        structure[0, 1, 1] = True
        structure[2, 1, 1] = True
        structure[1, 0, 1] = True
        structure[1, 2, 1] = True
        structure[1, 1, 0] = True
        structure[1, 1, 2] = True
        # Add edges (12)
        structure[0, 0, 1] = True
        structure[0, 2, 1] = True
        structure[2, 0, 1] = True
        structure[2, 2, 1] = True
        structure[0, 1, 0] = True
        structure[0, 1, 2] = True
        structure[2, 1, 0] = True
        structure[2, 1, 2] = True
        structure[1, 0, 0] = True
        structure[1, 0, 2] = True
        structure[1, 2, 0] = True
        structure[1, 2, 2] = True
        return structure

    elif connectivity == "26-connect":
        # All neighbors in 3x3x3 cube
        return np.ones((3, 3, 3), dtype=bool)

    else:
        raise ValueError(
            f"Invalid connectivity: {connectivity}. "
            "Choose from '6-connect', '18-connect', or '26-connect'."
        )


def _compute_cluster_table(
    vol: NeuroVol,
    labeled_array: np.ndarray,
    voxels_list: List[np.ndarray],
    space: NeuroSpace,
) -> pd.DataFrame:
    """Compute cluster statistics table.

    Parameters
    ----------
    vol : NeuroVol
        Original volume
    labeled_array : np.ndarray
        Labeled cluster array
    voxels_list : List[np.ndarray]
        List of voxel coordinates for each cluster
    space : NeuroSpace
        Spatial reference

    Returns
    -------
    pd.DataFrame
        Table with columns: index, x, y, z, N, Area, value
    """
    rows = []

    for i, voxels in enumerate(voxels_list, start=1):
        if len(voxels) == 0:
            continue

        # Get center of mass in grid coordinates
        center_grid = np.mean(voxels, axis=0)

        # Convert to world coordinates
        center_world = space.grid_to_coord(center_grid.reshape(1, -1))[0]

        # Get cluster size
        n_voxels = len(voxels)

        # Calculate area in mm^3
        voxel_volume = np.prod(space.spacing)
        area = n_voxels * voxel_volume

        # Get mean value in cluster
        cluster_mask = labeled_array == i
        mean_value = np.mean(vol.data[cluster_mask])

        rows.append(
            {
                "index": i,
                "x": center_world[0],
                "y": center_world[1],
                "z": center_world[2],
                "N": n_voxels,
                "Area": area,
                "value": mean_value,
            }
        )

    return pd.DataFrame(rows)


def _find_local_maxima(
    vol: NeuroVol, labeled_array: np.ndarray, min_distance: float, space: NeuroSpace
) -> np.ndarray:
    """Find local maxima within clusters.

    Parameters
    ----------
    vol : NeuroVol
        Original volume
    labeled_array : np.ndarray
        Labeled cluster array
    min_distance : float
        Minimum distance between maxima in mm
    space : NeuroSpace
        Spatial reference

    Returns
    -------
    np.ndarray
        Array with columns: index, x, y, z, value
    """
    # Convert minimum distance to voxels
    min_dist_voxels = min_distance / np.mean(space.spacing)

    maxima_list = []

    # Process each cluster
    num_clusters = np.max(labeled_array)
    for i in range(1, num_clusters + 1):
        cluster_mask = labeled_array == i

        # Extract cluster data
        cluster_data = np.where(cluster_mask, vol.data, -np.inf)

        # Find local maxima using maximum filter
        local_max = maximum_filter(cluster_data, size=3)
        maxima_mask = (cluster_data == local_max) & cluster_mask

        # Get maxima coordinates
        maxima_coords = np.column_stack(np.where(maxima_mask))

        if len(maxima_coords) == 0:
            continue

        # Get values at maxima
        maxima_values = vol.data[maxima_mask]

        # Sort by value (descending)
        # Handle boolean data
        if maxima_values.dtype == bool:
            # For boolean data, no need to sort
            sort_idx = np.arange(len(maxima_values))
        else:
            sort_idx = np.argsort(-maxima_values)
        maxima_coords = maxima_coords[sort_idx]
        maxima_values = maxima_values[sort_idx]

        # Apply minimum distance constraint
        keep_mask = np.ones(len(maxima_coords), dtype=bool)
        for j in range(len(maxima_coords)):
            if not keep_mask[j]:
                continue
            # Remove nearby maxima
            for k in range(j + 1, len(maxima_coords)):
                if keep_mask[k]:
                    dist = np.linalg.norm(maxima_coords[j] - maxima_coords[k])
                    if dist < min_dist_voxels:
                        keep_mask[k] = False

        # Keep only filtered maxima
        maxima_coords = maxima_coords[keep_mask]
        maxima_values = maxima_values[keep_mask]

        # Convert to world coordinates
        for coord, value in zip(maxima_coords, maxima_values):
            world_coord = space.grid_to_coord(coord.reshape(1, -1))[0]
            maxima_list.append(
                [i, world_coord[0], world_coord[1], world_coord[2], value]
            )

    if maxima_list:
        return np.array(maxima_list)
    else:
        return np.empty((0, 5))
