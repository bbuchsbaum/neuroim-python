"""Resampling and reorientation functions for neuroimaging data.

This module provides functions for resampling and reorienting neuroimaging
volumes and vectors to match different spatial configurations.
"""

import warnings

import numpy as np
from typing import Union, List

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
from .axis import LEFT_RIGHT, RIGHT_LEFT, ANT_POST, POST_ANT, INF_SUP, SUP_INF
from .results import ResampleParams, hash_neurospace, receipt_for


_LOW_OVERLAP_WARNING_THRESHOLD = 0.05


def _world_bounding_box(space: NeuroSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return the axis-aligned world bounds of the first three voxel axes."""
    if len(space.dim) < 3:
        raise ValueError("resampling requires at least three spatial dimensions")
    spatial_dim = np.asarray(space.dim[:3], dtype=float)
    # Voxel cells, rather than voxel centres, give non-zero physical extent to
    # singleton axes and describe the full support handed to the interpolator.
    edges = [(-0.5, dim - 0.5) for dim in spatial_dim]
    corners = np.array(np.meshgrid(*edges, indexing="ij")).reshape(3, -1).T
    world = space.grid_to_coord(corners)
    return np.min(world, axis=0), np.max(world, axis=0)


def _target_overlap_fraction(
    source_space: NeuroSpace, target_space: NeuroSpace
) -> tuple[float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Fraction of the target world AABB covered by the source world AABB."""
    source_bounds = _world_bounding_box(source_space)
    target_bounds = _world_bounding_box(target_space)
    overlap = np.maximum(
        0.0,
        np.minimum(source_bounds[1], target_bounds[1])
        - np.maximum(source_bounds[0], target_bounds[0]),
    )
    target_extent = np.maximum(target_bounds[1] - target_bounds[0], np.finfo(float).eps)
    fraction = float(np.prod(overlap) / np.prod(target_extent))
    return fraction, source_bounds, target_bounds


def _warn_if_low_overlap(
    source_space: NeuroSpace,
    target_space: NeuroSpace,
    *,
    threshold: float = _LOW_OVERLAP_WARNING_THRESHOLD,
) -> None:
    """Warn when less than ``threshold`` of the target world AABB overlaps.

    The 5% default is deliberately conservative. Axis-aligned world bounding
    boxes are a loose preflight check for obviously misplaced targets, not an
    intersection test for rotated image volumes.
    """
    fraction, source_bounds, target_bounds = _target_overlap_fraction(
        source_space, target_space
    )
    if not np.isfinite(fraction) or fraction >= threshold:
        return

    def _format_bounds(bounds: tuple[np.ndarray, np.ndarray]) -> str:
        return " ".join(
            f"[{low:.1f}, {high:.1f}]" for low, high in zip(bounds[0], bounds[1])
        )

    warnings.warn(
        "The resample target covers little of the source image: their world "
        f"bounding boxes overlap over {100.0 * fraction:.1f}% of the target, "
        "so most of the output will be background. "
        f"Source spans {_format_bounds(source_bounds)} and target spans "
        f"{_format_bounds(target_bounds)}. Check the target affine or use "
        "resample_to() with the image whose grid you want to match.",
        UserWarning,
        stacklevel=3,
    )


def _resample_method_name(
    source_space: NeuroSpace,
    target_space: NeuroSpace,
    interpolation: int,
    *,
    vector: bool = False,
) -> str:
    op = "resample_vec" if vector else "resample"
    return (
        f"{op}(order={interpolation},"
        f"source={hash_neurospace(source_space)},"
        f"target={hash_neurospace(target_space)})"
    )


def resample(
    source: NeuroVol, target: Union[NeuroVol, NeuroSpace], interpolation: int = 3
) -> DenseNeuroVol:
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

    Warns
    -----
    UserWarning
        If less than 5% of the target's axis-aligned world bounding box
        overlaps the source's. Resampling still proceeds normally.

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
        raise ImportError(
            "nibabel is required for resampling. " "Install with: pip install nibabel"
        )

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
        # Anatomical axes follow the target affine (the grid the data now
        # lives on), not the source's axes.
        target_space = NeuroSpace(
            dim=target.dim,
            spacing=target.spacing,
            origin=target.origin,
            trans=target.trans,
        )
    else:
        # Convert target NeuroVol to nibabel image
        target_data = target.data if hasattr(target, "data") else target.as_dense().data
        target_nifti = nib.Nifti1Image(target_data, target.space.trans)
        target_space = target.space

    _warn_if_low_overlap(source.space, target_space)

    # Perform resampling
    resampled_nifti = resample_from_to(source_nifti, target_nifti, order=interpolation)

    resampled_data = resampled_nifti.get_fdata()
    if preserve_clusters:
        labels = np.rint(resampled_data).astype(source.clusters.dtype)
        mask = labels != 0
        out = ClusteredNeuroVol(
            LogicalNeuroVol(mask, target_space),
            labels.ravel(order="F")[mask.ravel(order="F")],
            label_map=dict(source.label_map),
        )
        out.provenance = receipt_for(
            target_space,
            n_voxels=int(np.prod(tuple(int(d) for d in target_space.dim[:3]))),
            params=ResampleParams(
                method_name=_resample_method_name(
                    source.space, target_space, interpolation
                ),
                interpolation=int(interpolation),
            ),
        )
        return out

    out = DenseNeuroVol(resampled_data.astype(np.float32), target_space)
    out.provenance = receipt_for(
        target_space,
        n_voxels=int(np.prod(tuple(int(d) for d in target_space.dim[:3]))),
        params=ResampleParams(
            method_name=_resample_method_name(
                source.space, target_space, interpolation
            ),
            interpolation=int(interpolation),
        ),
    )
    return out


def resample_vec(
    source: NeuroVec, target: Union[NeuroVec, NeuroSpace], interpolation: int = 3
) -> DenseNeuroVec:
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
        raise ImportError(
            "nibabel is required for resampling. " "Install with: pip install nibabel"
        )

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

    if isinstance(target, NeuroSpace):
        target_4d_space = target
    else:
        target_4d_space = target.space

    out = DenseNeuroVec(resampled_data, target_4d_space)
    out.provenance = receipt_for(
        target_3d_space,
        n_voxels=int(np.prod(tuple(int(d) for d in target_3d_space.dim[:3]))),
        params=ResampleParams(
            method_name=_resample_method_name(
                source.space.drop_dim(3),
                target_3d_space,
                interpolation,
                vector=True,
            ),
            interpolation=int(interpolation),
        ),
    )
    return out


def resample_to(
    source: Union[NeuroVol, NeuroVec],
    target: Union[NeuroVol, NeuroVec, NeuroSpace],
    method: str = "linear",
    engine: str = "nibabel",
) -> Union[DenseNeuroVol, DenseNeuroVec]:
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


def reorient(
    x: Union[NeuroSpace, NeuroVol, NeuroVec], orient: Union[str, List[str]]
) -> Union[NeuroSpace, NeuroVol, NeuroVec]:
    """Remap the grid-to-world coordinates mapping of an image.

    Parameters
    ----------
    x : NeuroSpace, NeuroVol, or NeuroVec
        The object to reorient
    orient : str or list of str
        The orientation code indicating the remapped axes.
        Can be a 3-letter string like "RAS" or list like ["R", "A", "S"]
        The letters must contain exactly one left/right, one anterior/posterior,
        and one inferior/superior axis.  Their order is the desired voxel-axis
        order, so both canonical codes such as ``"RAS"`` and permutations such
        as ``"PIR"`` are valid.

    Returns
    -------
    NeuroSpace, NeuroVol, or NeuroVec
        A reoriented version of x

    Examples
    --------
    >>> # Reorient a space to RAS
    >>> space_ras = reorient(space, "RAS")
    >>>
    >>> # Or using a list
    >>> space_ras = reorient(space, ["R", "A", "S"])
    """
    if not HAS_NIBABEL:
        raise ImportError(
            "nibabel is required for reorientation. Install with: pip install nibabel"
        )

    # Parse and validate the target orientation.
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

    groups = ({"L", "R"}, {"A", "P"}, {"S", "I"})
    if any(sum(code in group for code in orient) != 1 for group in groups):
        raise ValueError(
            "Orientation must contain exactly one of L/R, one of A/P, "
            f"and one of S/I; got {''.join(orient)!r}"
        )

    # Map orientation codes to axes
    # ``orient`` uses nibabel's "increases towards" letters (RAS: x runs to
    # the Right), so "R" is the Left-to-Right axis.
    axis_map = {
        "R": LEFT_RIGHT,
        "L": RIGHT_LEFT,
        "P": ANT_POST,
        "A": POST_ANT,
        "I": SUP_INF,
        "S": INF_SUP,
    }

    new_axes = [axis_map[o] for o in orient]

    if not isinstance(x, (NeuroSpace, NeuroVol, NeuroVec)):
        raise TypeError("x must be NeuroSpace, NeuroVol, or NeuroVec")

    space = x if isinstance(x, NeuroSpace) else x.space
    if space.ndim < 3:
        raise ValueError("Reorientation requires at least three spatial dimensions")

    start = nib.orientations.io_orientation(space.affine)
    end = nib.orientations.axcodes2ornt(tuple(orient))
    transform = nib.orientations.ornt_transform(start, end)
    new_space = _reorient_space(space, new_axes, transform)

    if isinstance(x, NeuroSpace):
        return new_space

    if isinstance(x, (DenseNeuroVol, DenseNeuroVec)):
        source_data = x.data
    else:
        source_data = x.to_dense().data
    data = nib.orientations.apply_orientation(source_data, transform)
    if isinstance(x, NeuroVec):
        out = DenseNeuroVec(data, new_space, label=getattr(x, "label", ""))
        if hasattr(x, "provenance"):
            out.provenance = x.provenance
        return out
    return DenseNeuroVol(
        data,
        new_space,
        label=getattr(x, "label", ""),
        provenance=getattr(x, "provenance", None),
    )


def _reorient_space(space: NeuroSpace, new_axes, transform: np.ndarray) -> NeuroSpace:
    """Apply a nibabel orientation transform to a spatial grid contract."""
    from .axis import AxisSetND, axis_set

    spatial_order = np.argsort(transform[:, 0].astype(int))
    new_dim = tuple(int(space.dim[i]) for i in spatial_order) + tuple(
        int(d) for d in space.dim[3:]
    )

    # ``inv_ornt_aff`` maps coordinates in the new array back to coordinates
    # in the old array.  Right-multiplying the grid-to-world transform is
    # therefore the exact change of voxel basis, including flip translations.
    spatial_change = nib.orientations.inv_ornt_aff(
        transform, tuple(int(d) for d in space.dim[:3])
    )
    if space.ndim == 3:
        new_trans = space.affine @ spatial_change
    else:
        change = np.eye(space.ndim + 1)
        change[:3, :3] = spatial_change[:3, :3]
        change[:3, -1] = spatial_change[:3, 3]
        new_trans = space.trans @ change

    if isinstance(space.axes, AxisSetND):
        old_axes = list(space.axes)
    else:
        old_axes = [getattr(space.axes, attr) for attr in "ijklm"[: space.ndim]]
    axes_set = axis_set(space.ndim, list(new_axes) + old_axes[3:])

    return NeuroSpace(dim=new_dim, axes=axes_set, trans=new_trans)
