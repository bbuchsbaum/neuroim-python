"""Resampling and reorientation functions for neuroimaging data.

This module provides functions for resampling and reorienting neuroimaging
volumes and vectors to match different spatial configurations.
"""

import warnings

import numpy as np
from typing import Union, List, Tuple, Optional

# Try to import nibabel for resampling support
try:
    import nibabel as nib
    from nibabel.processing import resample_from_to
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from .neuro_vol import NeuroVol, DenseNeuroVol, SparseNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .neuro_space import NeuroSpace
from .axis import (
    NamedAxis, LEFT_RIGHT, RIGHT_LEFT, ANT_POST, POST_ANT,
    INF_SUP, SUP_INF
)

def resample(source: NeuroVol, target: Union[NeuroVol, NeuroSpace],
             interpolation: int = 3) -> DenseNeuroVol:
    """Resample an image to match the space of another image.

    This function resamples a source image to match the spatial properties
    (dimensions, resolution, and orientation) of a target image or space.

    Parameters
    ----------
    source : NeuroVol
        The source volume to be resampled
    target : NeuroVol or NeuroSpace
        The target space to match the dimensions and orientation
    interpolation : int
        Type of interpolation to be applied:
        - 0: nearest neighbor
        - 1: trilinear
        - 3: cubic spline (default)

    Returns
    -------
    DenseNeuroVol
        The resampled source image with the same spatial properties as target

    Raises
    ------
    ImportError
        If nibabel is not installed
    ValueError
        If interpolation is not 0, 1, or 3

    Examples
    --------
    >>> # Resample to match another volume
    >>> resampled = resample(source_vol, target_vol, interpolation=1)
    >>>
    >>> # Resample to match a specific space
    >>> new_space = NeuroSpace(dim=[128, 128, 64], spacing=[2, 2, 2])
    >>> resampled = resample(source_vol, new_space)
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for resampling. "
                         "Install with: pip install nibabel")

    if interpolation not in [0, 1, 3]:
        raise ValueError("interpolation must be 0 (nearest), 1 (linear), or 3 (cubic)")

    from .clustered_neuro_vol import ClusteredNeuroVol
    from .neuro_vol import LogicalNeuroVol

    preserve_clusters = isinstance(source, ClusteredNeuroVol)
    if preserve_clusters and interpolation != 0:
        warnings.warn(
            "ClusteredNeuroVol resampling uses nearest-neighbor interpolation",
            UserWarning,
            stacklevel=2,
        )
        interpolation = 0

    # Convert source NeuroVol to nibabel image
    if preserve_clusters:
        source_data = source.as_dense().data.astype(np.int32)
    else:
        source_data = source.data if hasattr(source, "data") else source.as_dense().data
    source_nifti = nib.Nifti1Image(source_data, source.space.trans)

    if isinstance(target, NeuroSpace):
        # Create a dummy nibabel image for the target space
        target_shape = tuple(int(d) for d in target.dim[:3])
        target_nifti = nib.Nifti1Image(np.zeros(target_shape), target.trans)
        # Create a target space that preserves the source axes orientation
        target_space = NeuroSpace(
            dim=target.dim,
            spacing=target.spacing,
            origin=target.origin,
            axes=source.space.axes,  # Preserve source axes orientation
            trans=target.trans
        )
    else:
        # Convert target NeuroVol to nibabel image
        target_data = target.data if hasattr(target, "data") else target.as_dense().data
        target_nifti = nib.Nifti1Image(target_data, target.space.trans)
        target_space = target.space

    # Perform resampling
    resampled_nifti = resample_from_to(source_nifti, target_nifti, order=interpolation)

    resampled_data = resampled_nifti.get_fdata()
    if preserve_clusters:
        labels = np.rint(resampled_data).astype(source.clusters.dtype)
        mask = labels != 0
        return ClusteredNeuroVol(
            LogicalNeuroVol(mask, target_space),
            labels.ravel(order="F")[mask.ravel(order="F")],
            label_map=dict(source.label_map),
        )

    return DenseNeuroVol(resampled_data.astype(np.float32), target_space)

def resample_vec(source: NeuroVec, target: Union[NeuroVec, NeuroSpace],
                 interpolation: int = 3) -> DenseNeuroVec:
    """Resample a 4D NeuroVec to match the space of another image.

    Parameters
    ----------
    source : NeuroVec
        The source 4D vector to be resampled
    target : NeuroVec or NeuroSpace
        The target space to match (must be 4D if NeuroSpace)
    interpolation : int
        Type of interpolation (0: nearest, 1: linear, 3: cubic)

    Returns
    -------
    DenseNeuroVec
        The resampled source vector

    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for resampling. "
                         "Install with: pip install nibabel")

    # Extract individual volumes and resample
    n_vols = source.shape[3]
    resampled_vols = []

    # Get target 3D space by dropping the last dimension (time/4th dimension)
    if isinstance(target, NeuroSpace):
        if target.ndim != 4:
            raise ValueError("Target NeuroSpace must be 4D for NeuroVec resampling")
        target_3d_space = target.drop_dim(3)  # Drop the 4th dimension
    else:
        target_3d_space = target.space.drop_dim(3)  # Drop the 4th dimension

    # Resample each volume
    for i in range(n_vols):
        vol = source.vols(i)
        resampled_vol = resample(vol, target_3d_space, interpolation)
        resampled_vols.append(resampled_vol.data)

    # Stack resampled volumes
    resampled_data = np.stack(resampled_vols, axis=-1)

    # Create target 4D space that preserves source axes
    if isinstance(target, NeuroSpace):
        # Create a 4D space that preserves the source axes orientation
        target_4d_space = NeuroSpace(
            dim=target.dim,
            spacing=target.spacing,
            origin=target.origin,
            axes=source.space.axes,  # Preserve source axes orientation
            trans=target.trans if hasattr(target, 'trans') else None
        )
    else:
        target_4d_space = target.space

    return DenseNeuroVec(resampled_data, target_4d_space)

def resample_to(source: Union[NeuroVol, NeuroVec],
                target: Union[NeuroVol, NeuroVec, NeuroSpace],
                method: str = "linear",
                engine: str = "nibabel") -> Union[DenseNeuroVol, DenseNeuroVec]:
    """Resample a volume or vector to a target using neuroim2-style names."""
    method_map = {"nearest": 0, "linear": 1, "cubic": 3}
    if method not in method_map:
        raise ValueError("method must be one of 'nearest', 'linear', or 'cubic'")
    if engine != "nibabel":
        raise ValueError("Only the nibabel resampling engine is supported")
    interpolation = method_map[method]
    if isinstance(source, NeuroVec):
        return resample_vec(source, target, interpolation=interpolation)
    return resample(source, target, interpolation=interpolation)

def reorient(x: Union[NeuroSpace, NeuroVol], orient: Union[str, List[str]]) -> Union[NeuroSpace, NeuroVol]:
    """Remap the grid-to-world coordinates mapping of an image.

    Parameters
    ----------
    x : NeuroSpace or NeuroVol
        The object to reorient
    orient : str or list of str
        The orientation code indicating the remapped axes.
        Can be a 3-letter string like "RAS" or list like ["R", "A", "S"]
        Each position represents:
        - First: "R" (Right) or "L" (Left)
        - Second: "A" (Anterior) or "P" (Posterior)
        - Third: "S" (Superior) or "I" (Inferior)

    Returns
    -------
    NeuroSpace or NeuroVol
        A reoriented version of x

    Examples
    --------
    >>> # Reorient a space to RAS
    >>> space_ras = reorient(space, "RAS")
    >>>
    >>> # Or using a list
    >>> space_ras = reorient(space, ["R", "A", "S"])
    """
    # Parse orientation string
    if isinstance(orient, str):
        if len(orient) != 3:
            raise ValueError("Orientation string must be exactly 3 characters")
        orient = list(orient.upper())
    elif isinstance(orient, list):
        if len(orient) != 3:
            raise ValueError("Orientation list must have exactly 3 elements")
        orient = [o.upper() for o in orient]
    else:
        raise TypeError("orient must be a string or list")

    # Validate orientation codes
    valid_lr = ["L", "R"]
    valid_ap = ["A", "P"]
    valid_si = ["S", "I"]

    if orient[0] not in valid_lr:
        raise ValueError(f"First orientation must be L or R, got {orient[0]}")
    if orient[1] not in valid_ap:
        raise ValueError(f"Second orientation must be A or P, got {orient[1]}")
    if orient[2] not in valid_si:
        raise ValueError(f"Third orientation must be S or I, got {orient[2]}")

    # Map orientation codes to axes
    axis_map = {
        "L": LEFT_RIGHT,
        "R": RIGHT_LEFT,
        "A": ANT_POST,
        "P": POST_ANT,
        "S": SUP_INF,
        "I": INF_SUP
    }

    new_axes = [axis_map[o] for o in orient]

    if isinstance(x, NeuroSpace):
        # For NeuroSpace, create new space with reoriented axes
        return _reorient_space(x, new_axes)
    elif isinstance(x, NeuroVol):
        # For NeuroVol, reorient the underlying space and data
        # Get current axes using axis attributes
        axis_attrs = ['i', 'j', 'k']
        old_axes = [getattr(x.space.axes, axis_attrs[i]) for i in range(min(3, x.space.ndim))]

        # Compute permutation and flips
        perm_indices, flip_axes = _compute_reorientation(old_axes, new_axes)

        # Apply permutation to data
        data = np.transpose(x.data, perm_indices)

        # Apply flips
        for axis, should_flip in enumerate(flip_axes[:len(perm_indices)]):
            if should_flip:
                data = np.flip(data, axis=axis)

        # Create new space with reoriented axes
        new_space = _reorient_space(x.space, new_axes)

        # Update dimensions in new space to match permuted data
        new_dim = [data.shape[i] for i in range(len(data.shape))]
        new_space = NeuroSpace(
            dim=new_dim,
            origin=new_space.origin,
            spacing=new_space.spacing,
            axes=new_space.axes,
            trans=new_space.trans
        )

        return DenseNeuroVol(data, new_space)
    else:
        raise TypeError("x must be NeuroSpace or NeuroVol")

def _get_permutation_matrix(axes) -> np.ndarray:
    """Get permutation matrix from axis set.

    Returns a matrix where columns are the direction vectors of each axis.
    """
    ndim = min(3, axes.ndim)  # Only handle spatial dimensions
    perm_mat = np.zeros((3, ndim))

    # Access axes as attributes
    axis_attrs = ['i', 'j', 'k']
    for idx in range(ndim):
        axis = getattr(axes, axis_attrs[idx])
        # Convert direction to a proper 3D vector
        if isinstance(axis.direction, (list, tuple)):
            direction = axis.direction
        elif isinstance(axis.direction, (int, float)):
            # Single value direction - expand to 3D vector
            dir_vec = [0, 0, 0]
            dir_vec[idx] = axis.direction
            direction = dir_vec
        else:
            direction = [0, 0, 0]
            direction[idx] = 1

        perm_mat[:, idx] = direction

    return perm_mat

def _compute_reorientation(old_axes: List[NamedAxis], new_axes: List[NamedAxis]) -> Tuple[List[int], List[bool]]:
    """Compute permutation indices and flip flags for reorientation.

    Returns
    -------
    perm_indices : list of int
        Permutation indices for np.transpose
    flip_axes : list of bool
        Whether to flip each axis after permutation
    """
    perm_indices = []
    flip_axes = []

    # For reorientation, we need to match axes by their anatomical direction
    # not by name, since we're converting from generic axes to anatomical ones
    for new_idx, new_ax in enumerate(new_axes[:3]):  # Only handle first 3 spatial axes
        # Find which old axis best matches this new axis direction
        # The new axis direction tells us which dimension it represents
        new_dir = np.array(new_ax.direction)
        abs_dir = np.abs(new_dir)

        # Find which dimension has the largest component
        dim_idx = np.argmax(abs_dir)

        # This is the old axis we want to map from
        perm_indices.append(dim_idx)

        # Check if we need to flip (if the direction is negative)
        # For standard axes, positive direction is assumed
        flip_axes.append(new_dir[dim_idx] < 0)

    # Add any remaining dimensions (time, etc.) unchanged
    n_old = len(old_axes)
    for i in range(3, n_old):
        perm_indices.append(i)
        flip_axes.append(False)

    return perm_indices, flip_axes

def _reorient_space(space: NeuroSpace, new_axes: List[NamedAxis]) -> NeuroSpace:
    """Helper function to reorient a NeuroSpace."""

    # Import AxisSet classes
    from .axis import AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D

    # Create appropriate AxisSet based on dimensionality
    ndim = space.ndim
    if ndim == 1:
        axes_set = AxisSet1D(new_axes[0])
    elif ndim == 2:
        axes_set = AxisSet2D(new_axes[0], new_axes[1])
    elif ndim == 3:
        axes_set = AxisSet3D(new_axes[0], new_axes[1], new_axes[2])
    elif ndim == 4:
        # For 4D, we need to add the time axis
        axes_set = AxisSet4D(new_axes[0], new_axes[1], new_axes[2], space.axes.l)
    elif ndim == 5:
        # For 5D, preserve last two axes
        axes_set = AxisSet5D(new_axes[0], new_axes[1], new_axes[2],
                            space.axes.l, space.axes.m)
    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}")

    # Compute new transformation matrix
    # Get permutation matrices for old and new axes
    old_perm = _get_permutation_matrix(space.axes)
    new_perm = _get_permutation_matrix(axes_set)

    # Update transformation matrix: new_trans = new_perm.T @ old_trans
    # This accounts for the reorientation of axes
    old_trans = space.trans
    if old_trans is not None and old_trans.shape[0] >= ndim:
        # Extract the relevant part of the transformation matrix
        trans_sub = old_trans[:ndim, :]

        # Apply the reorientation
        new_trans_sub = new_perm.T @ old_perm @ trans_sub

        # Reconstruct full transformation matrix
        new_trans = np.eye(old_trans.shape[0])
        new_trans[:ndim, :] = new_trans_sub
        new_trans = new_trans[:old_trans.shape[0], :old_trans.shape[1]]

        # Update origin based on new transformation
        new_origin = new_trans[:ndim, -1] if new_trans.shape[1] > ndim else space.origin
    else:
        new_trans = space.trans
        new_origin = space.origin

    return NeuroSpace(
        dim=space.dim,
        origin=new_origin,
        spacing=space.spacing,
        axes=axes_set,
        trans=new_trans
    )
