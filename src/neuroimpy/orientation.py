"""Orientation and affine utility functions for neuroimaging data.

This module provides functions for working with image orientations and
affine transformations. It wraps nibabel's orientation utilities with
a clean neuroimpy API.
"""

import numpy as np
from typing import Tuple, Optional, Sequence, Union

import nibabel as nib
import nibabel.orientations as nibo


def affine_to_orientation(affine: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """Convert a 4x4 affine matrix to an orientation array.

    The orientation array has shape (n, 2) where each row contains
    ``(axis, direction)``: *axis* is the input axis that most closely
    corresponds to the output axis (0, 1, or 2) and *direction* is 1 if
    the axes are aligned or -1 if they are flipped.

    Parameters
    ----------
    affine : ndarray, shape (4, 4)
        Affine transformation matrix.
    tol : float, optional
        Threshold for detecting zero columns. Passed to
        ``nibabel.orientations.io_orientation``.

    Returns
    -------
    ndarray, shape (3, 2)
        Orientation array.

    Examples
    --------
    >>> import numpy as np
    >>> ornt = affine_to_orientation(np.diag([2, 3, 4, 1]))
    >>> ornt  # doctest: +NORMALIZE_WHITESPACE
    array([[0., 1.],
           [1., 1.],
           [2., 1.]])
    """
    affine = np.asarray(affine, dtype=float)
    return nibo.io_orientation(affine, tol=tol)


def affine_to_axcodes(affine: np.ndarray,
                      labels: Optional[Sequence[Tuple[str, str]]] = None,
                      tol: Optional[float] = None) -> Tuple[str, ...]:
    """Convert a 4x4 affine matrix to axis codes such as ``('R', 'A', 'S')``.

    Parameters
    ----------
    affine : ndarray, shape (4, 4)
        Affine transformation matrix.
    labels : sequence of (str, str) pairs, optional
        Labels for each axis pair (e.g. ``(('L','R'), ('P','A'), ('I','S'))``).
        Defaults to the nibabel standard ``(('L','R'), ('P','A'), ('I','S'))``.
    tol : float, optional
        Tolerance for ``io_orientation``.

    Returns
    -------
    tuple of str
        Axis codes, e.g. ``('R', 'A', 'S')``.

    Examples
    --------
    >>> import numpy as np
    >>> affine_to_axcodes(np.diag([1, 1, 1, 1]))
    ('R', 'A', 'S')
    """
    affine = np.asarray(affine, dtype=float)
    kwargs = {}
    if labels is not None:
        kwargs["labels"] = labels
    if tol is not None:
        kwargs["tol"] = tol
    return tuple(nibo.aff2axcodes(affine, **kwargs))


def axcodes_to_orientation(axcodes: Union[str, Sequence[str]],
                           labels: Optional[Sequence[Tuple[str, str]]] = None) -> np.ndarray:
    """Convert axis codes (e.g. ``'RAS'``) to an orientation array.

    Parameters
    ----------
    axcodes : str or sequence of str
        Axis direction codes, e.g. ``'RAS'`` or ``('R', 'A', 'S')``.
    labels : sequence of (str, str) pairs, optional
        Labels for each axis pair. Defaults to nibabel standard.

    Returns
    -------
    ndarray, shape (N, 2)
        Orientation array.

    Examples
    --------
    >>> axcodes_to_orientation('RAS')  # doctest: +NORMALIZE_WHITESPACE
    array([[0., 1.],
           [1., 1.],
           [2., 1.]])
    """
    kwargs = {}
    if labels is not None:
        kwargs["labels"] = labels
    return nibo.axcodes2ornt(axcodes, **kwargs)


def orientation_to_axcodes(ornt: np.ndarray,
                           labels: Optional[Sequence[Tuple[str, str]]] = None) -> Tuple[str, ...]:
    """Convert an orientation array to axis codes."""
    kwargs = {}
    if labels is not None:
        kwargs["labels"] = labels
    return tuple(nibo.ornt2axcodes(np.asarray(ornt), **kwargs))


def orientation_transform(start_ornt: np.ndarray, end_ornt: np.ndarray) -> np.ndarray:
    """Return the orientation transform from one orientation to another."""
    return nibo.ornt_transform(np.asarray(start_ornt), np.asarray(end_ornt))


def axcodes(x, labels: Optional[Sequence[Tuple[str, str]]] = None) -> Tuple[str, ...]:
    """Return axis codes for a NeuroSpace-like object, affine, or orientation array."""
    arr = np.asarray(getattr(x, "trans", x), dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return orientation_to_axcodes(arr, labels=labels)
    return affine_to_axcodes(arr, labels=labels)


def apply_orientation(data: np.ndarray, ornt: np.ndarray) -> np.ndarray:
    """Apply an orientation transform to a data array.

    This reorders and/or flips the axes of *data* according to the
    orientation array *ornt*.

    Parameters
    ----------
    data : ndarray
        Data array (at least 3-D).
    ornt : ndarray, shape (N, 2)
        Orientation array as returned by :func:`affine_to_orientation`.

    Returns
    -------
    ndarray
        Reoriented data array.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(24).reshape(2, 3, 4)
    >>> ornt = np.array([[0, -1], [1, 1], [2, 1]])
    >>> out = apply_orientation(data, ornt)
    >>> out.shape
    (2, 3, 4)
    """
    return nibo.apply_orientation(data, ornt)


def apply_affine(aff: np.ndarray, pts: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Apply a homogeneous affine transform to points.

    Points are stored with coordinates on the last axis.  Vectors, matrices,
    and higher-dimensional point arrays are returned with the same leading
    shape.  ``inplace`` is accepted for neuroim2 API compatibility.
    """
    aff = np.asarray(aff, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if aff.ndim != 2 or aff.shape[0] < 2 or aff.shape[1] < 2:
        raise ValueError("aff must be a 2D homogeneous affine matrix")

    nd_in = aff.shape[1] - 1
    nd_out = aff.shape[0] - 1
    if pts.ndim == 1:
        if pts.shape[0] != nd_in:
            raise ValueError("For vector pts, length must match affine input dimension")
        pts_mat = pts.reshape(1, nd_in)
        vector_input = True
    else:
        if pts.shape[-1] != nd_in:
            raise ValueError("Last dimension of pts must match affine input dimension")
        pts_mat = pts.reshape(-1, nd_in)
        vector_input = False

    linear = aff[:nd_out, :nd_in]
    offset = aff[:nd_out, -1]
    out = pts_mat @ linear.T + offset
    if vector_input:
        return out[0]
    return out.reshape(pts.shape[:-1] + (nd_out,))


def append_diag(aff: np.ndarray, steps: Sequence[float],
                starts: Optional[Sequence[float]] = None) -> np.ndarray:
    """Append diagonal axes to a homogeneous affine."""
    aff = np.asarray(aff, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if aff.ndim != 2 or aff.shape[0] < 2 or aff.shape[1] < 2:
        raise ValueError("aff must be a 2D homogeneous affine matrix")
    if steps.ndim != 1 or steps.size == 0 or not np.all(np.isfinite(steps)):
        raise ValueError("steps must contain at least one finite numeric value")
    if starts is None:
        starts_arr = np.zeros_like(steps)
    else:
        starts_arr = np.asarray(starts, dtype=float)
        if starts_arr.shape != steps.shape or not np.all(np.isfinite(starts_arr)):
            raise ValueError("starts must be None or have the same finite length as steps")

    old_out = aff.shape[0] - 1
    old_in = aff.shape[1] - 1
    n_steps = steps.size
    out = np.zeros((old_out + n_steps + 1, old_in + n_steps + 1), dtype=float)
    out[:old_out, :old_in] = aff[:old_out, :old_in]
    out[:old_out, -1] = aff[:old_out, -1]
    for i, step in enumerate(steps):
        out[old_out + i, old_in + i] = step
    out[old_out:old_out + n_steps, -1] = starts_arr
    out[-1, -1] = 1.0
    return out


def voxel_sizes(affine: np.ndarray) -> np.ndarray:
    """Compute voxel sizes as column norms of the affine linear block."""
    affine = np.asarray(affine, dtype=float)
    if affine.ndim != 2 or affine.shape[0] < 2 or affine.shape[1] < 2:
        raise ValueError("affine must be a 2D homogeneous affine matrix")
    linear = affine[:-1, :-1]
    return np.sqrt(np.sum(linear * linear, axis=0))


def orientation_inverse_affine(ornt: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Compute the affine that reverses the given orientation transform.

    Given an orientation *ornt* and the *shape* of the data **after** the
    orientation has been applied, return the 4x4 affine that maps
    coordinates in the reoriented space back to the original space.

    Parameters
    ----------
    ornt : ndarray, shape (N, 2)
        Orientation array.
    shape : sequence of int
        Shape of the data after the orientation was applied.

    Returns
    -------
    ndarray, shape (4, 4)
        Inverse affine matrix.

    Examples
    --------
    >>> import numpy as np
    >>> ornt = np.array([[0, 1], [1, 1], [2, 1]])
    >>> orientation_inverse_affine(ornt, (10, 20, 30))  # doctest: +NORMALIZE_WHITESPACE
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """
    return nibo.inv_ornt_aff(ornt, shape)


def obliquity(affine: np.ndarray) -> float:
    """Compute the obliquity angle (in degrees) of an affine matrix.

    Obliquity measures how far the affine is from a pure axis-aligned
    (cardinal) orientation. An obliquity of 0 means perfectly aligned.

    Parameters
    ----------
    affine : ndarray, shape (4, 4)
        Affine transformation matrix.

    Returns
    -------
    float
        Maximum obliquity angle in degrees.

    Examples
    --------
    >>> import numpy as np
    >>> obliquity(np.diag([2, 2, 2, 1]))
    0.0
    """
    affine = np.asarray(affine, dtype=float)
    # Extract the 3x3 rotation/scale part
    rs = affine[:3, :3]
    # Normalize columns to get direction cosines
    norms = np.sqrt(np.sum(rs ** 2, axis=0))
    norms[norms == 0] = 1.0
    cosines = rs / norms
    # The obliquity is the maximum angle between each column and
    # its closest cardinal axis
    max_angle = 0.0
    for i in range(3):
        col = cosines[:, i]
        # Angle to closest cardinal axis = arccos(max |component|)
        max_component = np.max(np.abs(col))
        max_component = np.clip(max_component, 0, 1)
        angle = np.degrees(np.arccos(max_component))
        if angle > max_angle:
            max_angle = angle
    return float(max_angle)


def vox2out_vox(affine: np.ndarray, shape: Sequence[int]) -> Tuple[np.ndarray, Sequence[int]]:
    """Compute the voxel-to-output-voxel mapping for canonical orientation.

    This determines the transform needed to reorient the data to the
    closest canonical (RAS+) orientation and returns the new shape.

    Parameters
    ----------
    affine : ndarray, shape (4, 4)
        Affine transformation matrix.
    shape : sequence of int
        Shape of the data array.

    Returns
    -------
    ornt : ndarray, shape (3, 2)
        Orientation transform from current to canonical (RAS+).
    new_shape : tuple of int
        Shape of the data after reorientation.

    Examples
    --------
    >>> import numpy as np
    >>> ornt, new_shape = vox2out_vox(np.diag([1, 1, 1, 1]), (10, 20, 30))
    >>> new_shape
    (10, 20, 30)
    """
    affine = np.asarray(affine, dtype=float)
    ornt = nibo.io_orientation(affine)
    # Compute the reoriented shape
    shape = tuple(int(s) for s in shape)
    new_shape = tuple(shape[int(ornt[i, 0])] for i in range(len(ornt)))
    return ornt, new_shape


def perm_mat(ornt: np.ndarray) -> np.ndarray:
    """Compute a permutation matrix from an orientation array.

    The permutation matrix encodes axis reordering and flipping implied
    by the orientation.

    Parameters
    ----------
    ornt : ndarray, shape (N, 2)
        Orientation array.

    Returns
    -------
    ndarray, shape (N, N)
        Signed permutation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> ornt = np.array([[2, -1], [1, 1], [0, 1]])
    >>> perm_mat(ornt)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.,  0., -1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.]])
    """
    ornt = np.asarray(ornt, dtype=float)
    n = len(ornt)
    mat = np.zeros((n, n))
    for i in range(n):
        src_axis = int(ornt[i, 0])
        direction = ornt[i, 1]
        mat[i, src_axis] = direction
    return mat


def rescale_affine(affine: np.ndarray, shape: Sequence[int],
                   zooms: Sequence[float],
                   new_shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """Rescale an affine to reflect new voxel sizes.

    Adjusts the affine so the field of view is preserved (centered)
    when changing voxel sizes.

    Parameters
    ----------
    affine : ndarray, shape (4, 4)
        Original affine transformation matrix.
    shape : sequence of int
        Original data shape (at least 3 elements).
    zooms : sequence of float
        New voxel sizes for each spatial axis.
    new_shape : sequence of int, optional
        New data shape. If ``None``, computed from *shape* and *zooms*
        to preserve the field of view.

    Returns
    -------
    ndarray, shape (4, 4)
        Rescaled affine transformation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> aff = np.diag([2, 2, 2, 1])
    >>> aff[:3, 3] = [0, 0, 0]
    >>> new_aff = rescale_affine(aff, (10, 10, 10), (1, 1, 1))
    >>> new_aff[:3, :3].diagonal()
    array([1., 1., 1.])
    """
    affine = np.asarray(affine, dtype=float).copy()
    shape = np.asarray(shape[:3], dtype=float)
    zooms = np.asarray(zooms[:3], dtype=float)

    # Current voxel sizes from the affine
    old_zooms = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    old_zooms[old_zooms == 0] = 1.0

    if new_shape is None:
        # Compute new shape to preserve FOV
        fov = shape * old_zooms
        new_shape_arr = np.ceil(fov / zooms).astype(int)
    else:
        new_shape_arr = np.asarray(new_shape[:3], dtype=float)

    # Scale the rotation/direction part
    scale = zooms / old_zooms
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * scale[np.newaxis, :]

    # Adjust origin so the center of the FOV stays the same
    old_center = affine[:3, :3] @ ((shape - 1) / 2.0) + affine[:3, 3]
    new_origin = old_center - new_affine[:3, :3] @ ((new_shape_arr - 1) / 2.0)
    new_affine[:3, 3] = new_origin

    return new_affine
