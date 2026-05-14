"""Typed coordinate primitives and structural Protocols for neuroim.

Two surfaces live in this module:

1. Coordinate newtypes (``VoxelCoord``, ``WorldCoord``) for the "I forgot to
   apply the affine" bug.  Zero runtime cost; purely a type-layer guard.
2. Structural ``Protocol`` types extracted from how the adapter (WP-2) and
   result objects (WP-4) actually consume objects.  These are *post-hoc*
   specifications of observed contracts, not prior design.  Public function
   signatures consume Protocols; ABCs remain an internal implementation
   detail.

Per the consensus decision matrix (sticky ``post-01KRKFEWY2`` in the
``neuroim-python-pythonic-value`` mote topic), ``NeuroVolLike`` /
``NeuroVecLike`` callers do not need to inherit from the ABC tree.  Any
object exposing the documented attributes is accepted.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    NewType,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray


# --- Coordinate primitives --------------------------------------------------

VoxelCoord = NewType("VoxelCoord", NDArray[np.integer])
WorldCoord = NewType("WorldCoord", NDArray[np.floating])

CoordLike = Union[NDArray[np.number], "list[float]", "list[int]", "tuple[float, ...]"]


def voxel_coord(coords: CoordLike) -> VoxelCoord:
    arr = np.asarray(coords)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(
                f"voxel_coord 1-D input must have length 3, got {arr.shape}"
            )
    elif arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(
                f"voxel_coord 2-D input must have shape (N, 3), got {arr.shape}"
            )
    else:
        raise ValueError(f"voxel_coord input must be 1-D or 2-D, got ndim={arr.ndim}")
    return VoxelCoord(arr)


def world_coord(coords: CoordLike) -> WorldCoord:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(
                f"world_coord 1-D input must have length 3, got {arr.shape}"
            )
    elif arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(
                f"world_coord 2-D input must have shape (N, 3), got {arr.shape}"
            )
    else:
        raise ValueError(f"world_coord input must be 1-D or 2-D, got ndim={arr.ndim}")
    return WorldCoord(arr)


# --- Structural Protocols (WP-6) -------------------------------------------


@runtime_checkable
class HasSpace(Protocol):
    """Anything that carries a NeuroSpace-shaped spatial frame.

    The protocol intentionally only requires ``.space``; downstream callers
    interrogate the space object itself for ``dim``, ``affine``, ``spacing``,
    ``trans``, etc.
    """

    space: Any  # NeuroSpace; stays Any to avoid a circular import here


@runtime_checkable
class SupportsDense(Protocol):
    """Object that can materialize a dense representation."""

    def as_dense(self) -> Any: ...


@runtime_checkable
class SupportsSparse(Protocol):
    """Object that can materialize a sparse representation."""

    def as_sparse(self, mask: Any = None) -> Any: ...


@runtime_checkable
class NeuroVolLike(Protocol):
    """A 3-D spatial container with a numpy data buffer.

    Extracted from how :mod:`neuroim.searchlight_high_level`,
    :mod:`neuroim.operations`, and :mod:`neuroim.roi` actually consume
    volumes: a ``.space``, a ``.shape``, and a ``.data`` ndarray.  ABC
    subclasses such as ``DenseNeuroVol``/``LogicalNeuroVol`` satisfy this
    structurally; user code does not have to inherit from ``NeuroVol``.
    """

    space: Any
    shape: Tuple[int, ...]
    data: np.ndarray


@runtime_checkable
class NeuroVecLike(Protocol):
    """A 4-D voxel-series container.

    Required attributes:

    - ``space``: a :class:`~neuroim.neuro_space.NeuroSpace`.
    - ``shape``: the 4-tuple ``(x, y, z, t)``.
    - ``series(coords) -> ndarray``: extract a ``(T, V)`` time-by-voxel
      matrix at the given grid coordinates.

    Used by :func:`neuroim.searchlight` to walk per-searchlight time
    series.  Callers that already have their own backend can satisfy this
    Protocol without inheriting from ``NeuroVec``.
    """

    space: Any
    shape: Tuple[int, int, int, int]

    def series(self, coords: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class MaskLike(Protocol):
    """A 3-D boolean spatial mask.

    Concrete subclasses (``LogicalNeuroVol``, ``DenseNeuroVol`` whose
    ``data`` is bool-castable) satisfy this Protocol.  Callers can also
    pass a hand-rolled object exposing a 3-D boolean ``.data`` and a
    NeuroSpace-shaped ``.space``.
    """

    space: Any
    data: np.ndarray

    def as_logical(self) -> Any: ...


#: Callable returning the per-voxel offset table for a searchlight shape.
#: Existing shape factories ``ellipsoid_shape``, ``cube_shape``, and
#: ``blobby_shape`` in :mod:`neuroim.searchlight` are valid
#: ``ShapeFunction`` values.
ShapeFunction = Callable[..., Any]


__all__ = [
    "CoordLike",
    "HasSpace",
    "MaskLike",
    "NeuroVecLike",
    "NeuroVolLike",
    "ShapeFunction",
    "SupportsDense",
    "SupportsSparse",
    "VoxelCoord",
    "WorldCoord",
    "voxel_coord",
    "world_coord",
]
