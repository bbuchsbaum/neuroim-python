"""Orthogonal slice extraction for neuroimaging volumes.

This module provides functionality to extract orthogonal slices (axial, sagittal, coronal)
from 3D neuroimaging volumes at specified world-space coordinates.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from .neuro_vol import NeuroVol
from .neuro_slice import NeuroSlice, slice as extract_slice
from .neuro_space import NeuroSpace


def extract_orthogonal_slices(
    vol: NeuroVol,
    world_point: np.ndarray,
    slice_types: Optional[List[str]] = None
) -> Dict[str, NeuroSlice]:
    """Extract orthogonal slices from a volume at a given world-space point.
    
    This function extracts axial, sagittal, and/or coronal slices from a 3D
    neuroimaging volume at the specified world-space coordinate.
    
    Parameters
    ----------
    vol : NeuroVol
        The 3D volume to extract slices from
    world_point : np.ndarray
        World-space coordinates (x, y, z) at which to extract slices
    slice_types : List[str], optional
        List of slice types to extract. Valid values are 'axial', 'sagittal', 'coronal'.
        If None, all three slice types are extracted.
        
    Returns
    -------
    Dict[str, NeuroSlice]
        Dictionary mapping slice type names to NeuroSlice objects
        
    Raises
    ------
    ValueError
        If world_point is outside volume bounds or has wrong dimensions
    TypeError
        If vol is not a NeuroVol
        
    Examples
    --------
    >>> # Extract all orthogonal slices at world point (10, 20, 30)
    >>> slices = extract_orthogonal_slices(vol, np.array([10, 20, 30]))
    >>> axial_slice = slices['axial']
    >>> 
    >>> # Extract only sagittal slice
    >>> slices = extract_orthogonal_slices(vol, np.array([10, 20, 30]), ['sagittal'])
    >>> sagittal_slice = slices['sagittal']
    """
    if not isinstance(vol, NeuroVol):
        raise TypeError("vol must be a NeuroVol object")
    
    if vol.space.ndim != 3:
        raise ValueError("Volume must be 3-dimensional")
    
    # Validate world point
    world_point = np.asarray(world_point)
    if world_point.shape != (3,):
        raise ValueError("world_point must be a 3-element array")
    
    # Convert world coordinates to grid indices
    grid_point = vol.space.coord_to_grid(world_point.reshape(1, -1))[0]
    
    # Check bounds
    for i, (idx, dim) in enumerate(zip(grid_point, vol.shape)):
        if idx < 0 or idx >= dim:
            raise ValueError(f"World point {world_point} is outside volume bounds. "
                           f"Grid index {idx} is outside range [0, {dim-1}] for axis {i}")
    
    # Default to all slice types if not specified
    if slice_types is None:
        slice_types = ['axial', 'sagittal', 'coronal']
    
    # Validate slice types
    valid_types = {'axial', 'sagittal', 'coronal'}
    for slice_type in slice_types:
        if slice_type not in valid_types:
            raise ValueError(f"Invalid slice type: {slice_type}. "
                           f"Valid types are: {valid_types}")
    
    # Extract requested slices
    result = {}
    
    if 'axial' in slice_types:
        result['axial'] = extract_axial_slice(vol, world_point)
    
    if 'sagittal' in slice_types:
        result['sagittal'] = extract_sagittal_slice(vol, world_point)
    
    if 'coronal' in slice_types:
        result['coronal'] = extract_coronal_slice(vol, world_point)
    
    return result


_PLANE_WORLD_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def _plane_native_axis(vol: NeuroVol, plane: str) -> int:
    """Native voxel axis whose slices form ``plane`` under the volume affine."""
    from nibabel.orientations import io_orientation

    if plane not in _PLANE_WORLD_AXIS:
        raise ValueError(f"Invalid slice type: {plane}")
    world_axes = io_orientation(np.asarray(vol.space.affine, dtype=float))[:, 0]
    return int(np.flatnonzero(world_axes == _PLANE_WORLD_AXIS[plane])[0])


def _plane_slice(vol: NeuroVol, world_point: np.ndarray, plane: str) -> NeuroSlice:
    axis = _plane_native_axis(vol, plane)
    grid_point = vol.space.coord_to_grid(np.asarray(world_point).reshape(1, -1))[0]
    index = int(np.round(grid_point[axis]))
    if index < 0 or index >= vol.shape[axis]:
        raise ValueError(f"world point lies outside the volume along the {plane} axis")
    return extract_slice(vol, index, axis)


def extract_axial_slice(vol: NeuroVol, world_point: np.ndarray) -> NeuroSlice:
    """Extract an axial slice at the given world-space point.
    
    An axial slice is a horizontal slice that shows left-right and 
    anterior-posterior dimensions.
    
    Parameters
    ----------
    vol : NeuroVol
        The 3D volume to extract slice from
    world_point : np.ndarray
        World-space coordinates (x, y, z) at which to extract slice
        
    Returns
    -------
    NeuroSlice
        2D axial slice at the specified z-coordinate
        
    Notes
    -----
    The axial slice is extracted along the z-axis (inferior-superior axis),
    showing the x-y plane (left-right and anterior-posterior).
    """
    # The axial plane is found through the affine, so this works however
    # the data are stored (not just for RAS/LPI-ordered arrays).
    return _plane_slice(vol, world_point, "axial")


def extract_sagittal_slice(vol: NeuroVol, world_point: np.ndarray) -> NeuroSlice:
    """Extract a sagittal slice at the given world-space point.
    
    A sagittal slice is a vertical slice that divides the brain into 
    left and right portions, showing anterior-posterior and inferior-superior
    dimensions.
    
    Parameters
    ----------
    vol : NeuroVol
        The 3D volume to extract slice from
    world_point : np.ndarray
        World-space coordinates (x, y, z) at which to extract slice
        
    Returns
    -------
    NeuroSlice
        2D sagittal slice at the specified x-coordinate
        
    Notes
    -----
    The sagittal slice is extracted along the x-axis (left-right axis),
    showing the y-z plane (anterior-posterior and inferior-superior).
    """
    # The sagittal plane is found through the affine, so this works however
    # the data are stored (not just for RAS/LPI-ordered arrays).
    return _plane_slice(vol, world_point, "sagittal")


def extract_coronal_slice(vol: NeuroVol, world_point: np.ndarray) -> NeuroSlice:
    """Extract a coronal slice at the given world-space point.
    
    A coronal slice is a vertical slice that divides the brain into 
    anterior and posterior portions, showing left-right and inferior-superior
    dimensions.
    
    Parameters
    ----------
    vol : NeuroVol
        The 3D volume to extract slice from
    world_point : np.ndarray
        World-space coordinates (x, y, z) at which to extract slice
        
    Returns
    -------
    NeuroSlice
        2D coronal slice at the specified y-coordinate
        
    Notes
    -----
    The coronal slice is extracted along the y-axis (anterior-posterior axis),
    showing the x-z plane (left-right and inferior-superior).
    """
    # The coronal plane is found through the affine, so this works however
    # the data are stored (not just for RAS/LPI-ordered arrays).
    return _plane_slice(vol, world_point, "coronal")


def get_slice_orientation(vol: NeuroVol, slice_type: str) -> str:
    """Get the anatomical orientation of a slice type for the given volume.
    
    Parameters
    ----------
    vol : NeuroVol
        The volume to check orientation for
    slice_type : str
        Type of slice: 'axial', 'sagittal', or 'coronal'
        
    Returns
    -------
    str
        Anatomical orientation string (e.g., 'LR-PA' for axial in LPI space)
        
    Notes
    -----
    This function returns the anatomical orientation of the 2D slice axes
    based on the volume's coordinate system. For example:
    - Axial in LPI: shows Left-Right and Posterior-Anterior
    - Sagittal in LPI: shows Posterior-Anterior and Inferior-Superior
    - Coronal in LPI: shows Left-Right and Inferior-Superior
    """
    from nibabel.orientations import io_orientation
    from .axis import axis_names

    axes = axis_names(vol.space.axes)
    normal = _plane_native_axis(vol, slice_type)
    world_axes = io_orientation(np.asarray(vol.space.affine, dtype=float))[:, 0]
    # In-plane native axes, ordered by the world axis they run along (x, y, z).
    in_plane = sorted((a for a in range(3) if a != normal), key=lambda a: world_axes[a])
    return f"{axes[in_plane[0]]}-{axes[in_plane[1]]}"


def get_world_bounds_for_slice(
    vol: NeuroVol, 
    slice_type: str,
    slice_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get world-space bounds for a slice.
    
    Parameters
    ----------
    vol : NeuroVol
        The volume to get bounds from
    slice_type : str
        Type of slice: 'axial', 'sagittal', or 'coronal'
    slice_index : int, optional
        Grid index of the slice. If None, returns bounds for entire dimension.
        
    Returns
    -------
    min_bounds : np.ndarray
        Minimum world coordinates for the slice
    max_bounds : np.ndarray
        Maximum world coordinates for the slice
    """
    # Per-axis world extent of the voxel centres (corners can be flipped).
    corners = vol.space.bounds()[:, :3]
    min_bounds = corners.min(axis=0)
    max_bounds = corners.max(axis=0)

    if slice_index is not None:
        world_axis = _PLANE_WORLD_AXIS.get(slice_type)
        if world_axis is None:
            raise ValueError(f"Invalid slice type: {slice_type}")
        native = _plane_native_axis(vol, slice_type)

        # World coordinate of the slice along the plane's normal
        grid_point = np.zeros(3)
        grid_point[native] = slice_index
        world_coord = vol.space.grid_to_coord(grid_point.reshape(1, -1))[0]

        min_bounds[world_axis] = world_coord[world_axis]
        max_bounds[world_axis] = world_coord[world_axis]

    return min_bounds, max_bounds