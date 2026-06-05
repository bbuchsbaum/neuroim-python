# mypy: disable-error-code="arg-type,assignment,attr-defined,call-overload,no-any-return,no-untyped-def,type-arg"
import numpy as np
from typing import Any, Tuple, Union, Optional, List
from .axis import (
    AxisSet,
    AxisSet1D,
    AxisSet2D,
    AxisSet3D,
    AxisSet4D,
    AxisSet5D,
    AxisSetND,
    NamedAxis,
    find_anatomy_3d,
    axis_names,
    axis_set,
)
from .protocols import VoxelCoord, WorldCoord
from .exceptions import InvalidSpaceError, ImmutableError, SpaceMismatchError


def _readonly_array(value: np.ndarray, *, dtype=None) -> np.ndarray:
    arr = np.array(value, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


class NeuroSpace:
    """Spatial contract mapping image grid coordinates to world coordinates.

    Parameters
    ----------
    dim : tuple of int
        Dimensions of the image space
    spacing : tuple of float, optional
        Voxel dimensions (default: all 1.0)
    origin : tuple of float, optional
        Origin coordinates (default: all 0.0)
    axes : AxisSet, optional
        Axis specification (default: determined from trans or standard axes)
    trans : ndarray, optional
        4x4 transformation matrix (default: created from spacing/origin)
    """

    @staticmethod
    def _fit_axis_vector(
        values: np.ndarray, ndim: int, fill: float, name: str
    ) -> np.ndarray:
        """Conform a per-axis vector (spacing/origin) to ``ndim`` axes.

        A user describing a 4D series naturally supplies only the three
        *spatial* spacings — the time axis has no spatial extent.  Rather than
        letting a length mismatch surface as a cryptic numpy broadcast error
        deep in the affine construction, pad any trailing (e.g. temporal) axes
        with a neutral default (unit spacing / zero origin).  An over-long
        vector is a genuine mistake and raises a clear error.
        """
        values = np.atleast_1d(values)
        n = values.shape[0]
        if n == ndim:
            return values
        if n < ndim:
            return np.concatenate([values, np.full(ndim - n, fill, dtype=float)])
        raise InvalidSpaceError(
            f"{name} has {n} entries but the space is {ndim}D; "
            f"provide at most {ndim} values"
        )

    def __init__(
        self,
        dim: Union[Tuple[int, ...], List[int], np.ndarray],
        spacing: Optional[Union[Tuple[float, ...], List[float], np.ndarray]] = None,
        origin: Optional[Union[Tuple[float, ...], List[float], np.ndarray]] = None,
        axes: Optional[
            Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D]
        ] = None,
        trans: Optional[np.ndarray] = None,
    ):
        object.__setattr__(self, "_frozen", False)

        # Convert to numpy arrays
        object.__setattr__(self, "dim", np.array(dim, dtype=int, copy=True))

        # Validate dimensions
        if len(self.dim) == 0:
            raise ValueError("Dimensions cannot be empty")
        if np.any(self.dim <= 0):
            raise InvalidSpaceError("All dimensions must be positive")

        ndim = len(self.dim)

        # Track whether user explicitly provided spacing/origin
        _spacing_provided = spacing is not None
        _origin_provided = origin is not None

        # Set defaults if not provided
        if spacing is None:
            spacing = np.ones(ndim)
        else:
            spacing = np.array(spacing, dtype=float, copy=True)
            spacing = self._fit_axis_vector(spacing, ndim, 1.0, "spacing")

        if origin is None:
            origin = np.zeros(ndim)
        else:
            origin = np.array(origin, dtype=float, copy=True)
            origin = self._fit_axis_vector(origin, ndim, 0.0, "origin")

        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)

        # Validate spacing
        if np.any(self.spacing <= 0):
            raise InvalidSpaceError("Spacing values must be positive")

        # Handle transformation matrix
        if trans is None:
            # Create transformation matrix from spacing and origin
            if ndim <= 3:
                trans_array = np.eye(4)
                trans_array[:ndim, :ndim] = np.diag(self.spacing)
                trans_array[:ndim, 3] = self.origin
                object.__setattr__(self, "trans", trans_array)
            else:
                # For 4D+ spaces, use extended transformation matrix
                trans_array = np.eye(ndim + 1)
                trans_array[:ndim, :ndim] = np.diag(self.spacing)
                trans_array[:ndim, ndim] = self.origin
                object.__setattr__(self, "trans", trans_array)
        else:
            object.__setattr__(self, "trans", np.array(trans, dtype=float, copy=True))
            # For 4D+ spaces, allow larger transformation matrices
            if ndim <= 3 and self.trans.shape != (4, 4):
                raise InvalidSpaceError(
                    f"trans must be 4x4 matrix for {ndim}D space, got {self.trans.shape}"
                )
            elif ndim > 3 and self.trans.shape != (ndim + 1, ndim + 1):
                # Try to use 4x4 for first 3 dimensions
                if self.trans.shape == (4, 4):
                    # Extend to full size
                    full_trans = np.eye(ndim + 1)
                    full_trans[:3, :3] = self.trans[:3, :3]
                    full_trans[:3, -1] = self.trans[:3, 3]
                    # Additional dimensions get identity
                    for i in range(3, ndim):
                        full_trans[i, i] = (
                            self.spacing[i] if i < len(self.spacing) else 1.0
                        )
                    object.__setattr__(self, "trans", full_trans)
                elif self.trans.shape == (ndim + 1, ndim):
                    compact = self.trans
                    full_trans = np.eye(ndim + 1)
                    full_trans[:3, :3] = compact[:3, :3]
                    full_trans[:3, -1] = compact[:3, -1]
                    for i in range(3, ndim):
                        full_trans[i, i] = compact[i, -1]
                    object.__setattr__(self, "trans", full_trans)
                else:
                    raise InvalidSpaceError(
                        f"trans must be {ndim+1}x{ndim+1} matrix for {ndim}D space, got {self.trans.shape}"
                    )

            # Extract spacing and origin from the affine when not provided explicitly;
            # when trans is supplied, it takes precedence over the scalar inputs.
            if trans is not None and not _spacing_provided:
                object.__setattr__(
                    self,
                    "spacing",
                    np.sqrt(np.sum(self.trans[:ndim, :ndim] ** 2, axis=0)),
                )
            if trans is not None and not _origin_provided:
                if ndim <= 3:
                    object.__setattr__(self, "origin", self.trans[:ndim, 3])
                else:
                    object.__setattr__(self, "origin", self.trans[:ndim, ndim])

        # Calculate inverse transformation
        try:
            object.__setattr__(self, "inverse", np.linalg.inv(self.trans))
        except np.linalg.LinAlgError:
            rank = np.linalg.matrix_rank(self.trans)
            if ndim < 3 and rank >= ndim + 1:
                object.__setattr__(self, "inverse", np.linalg.pinv(self.trans))
            else:
                raise InvalidSpaceError("Transformation matrix must be invertible")

        # Handle axes
        # Convert list of axis names to AxisSet
        if isinstance(axes, (list, tuple)):
            from .axis import axis_set

            axes = axis_set(len(axes), list(axes))

        if axes is None:
            if ndim == 3 and trans is not None:
                # Use nearest anatomy like R does
                axes = self._nearest_anatomy()
            else:
                # Create default axes
                axes = self._default_axes(ndim)

        object.__setattr__(self, "axes", axes)

        # Validate dimensions match
        if self.axes.ndim != ndim:
            raise ValueError(
                f"Axes dimensionality ({self.axes.ndim}) must match space dimensionality ({ndim})"
            )

        for name, dtype in (
            ("dim", int),
            ("spacing", float),
            ("origin", float),
            ("trans", float),
            ("inverse", float),
        ):
            object.__setattr__(
                self, name, _readonly_array(getattr(self, name), dtype=dtype)
            )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise ImmutableError("NeuroSpace is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def from_affine(
        cls,
        affine: np.ndarray,
        shape: Union[Tuple[int, ...], List[int], np.ndarray],
        *,
        header: Optional[Any] = None,
    ) -> "NeuroSpace":
        """Create a space from a nibabel-style spatial affine and shape."""
        dim = tuple(int(d) for d in shape)
        affine = np.asarray(affine, dtype=float)
        if affine.shape != (4, 4):
            raise ValueError(f"affine must be 4x4, got {affine.shape}")

        spacing = None
        if header is not None and hasattr(header, "get_zooms"):
            zooms = tuple(float(z) for z in header.get_zooms()[: len(dim)])
            if len(zooms) == len(dim):
                spacing = zooms

        if spacing is None:
            spatial_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
            if len(dim) > 3:
                spacing = tuple(spatial_spacing) + tuple(1.0 for _ in dim[3:])
            else:
                spacing = tuple(spatial_spacing[: len(dim)])

        origin = tuple(affine[:3, 3]) + tuple(0.0 for _ in dim[3:])
        return cls(dim, spacing=spacing, origin=origin, trans=affine)

    @classmethod
    def from_nibabel(cls, img: Any) -> "NeuroSpace":
        """Create a space from a nibabel ``SpatialImage``-like object."""
        missing = [name for name in ("shape", "affine") if not hasattr(img, name)]
        if missing:
            raise TypeError(
                "from_nibabel expects a SpatialImage-like object with "
                + ", ".join(missing)
            )
        return cls.from_affine(
            img.affine,
            img.shape,
            header=getattr(img, "header", None),
        )

    @property
    def affine(self) -> np.ndarray:
        """Return the 4x4 spatial affine for this space."""
        if self.trans.shape == (4, 4):
            return self.trans.copy()
        affine = np.eye(4)
        affine[:3, :3] = self.trans[:3, :3]
        affine[:3, 3] = self.trans[:3, -1]
        return affine

    def compatible_with(self, other: "NeuroSpace", *, atol: float = 1e-6) -> bool:
        """Return True when two spaces share spatial shape and affine.

        Raises
        ------
        ValueError
            If spatial dimensions or affine differ.  The message names the
            mismatched field and includes both values so callers can diagnose
            mask/space mistakes directly.
        """
        if not isinstance(other, NeuroSpace):
            raise TypeError(f"other must be NeuroSpace, got {type(other).__name__}")

        self_dim = tuple(int(d) for d in self.dim[:3])
        other_dim = tuple(int(d) for d in other.dim[:3])
        if self_dim != other_dim:
            raise SpaceMismatchError(
                "NeuroSpace mismatch in spatial dim: "
                f"left={self_dim}, right={other_dim}"
            )

        if not np.allclose(self.affine, other.affine, atol=atol):
            raise SpaceMismatchError(
                "NeuroSpace mismatch in affine: "
                f"left={self.affine.tolist()}, right={other.affine.tolist()}"
            )

        return True

    def _default_axes(
        self, ndim: int
    ) -> Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND]:
        """Create default axes for given dimensionality."""
        from .axis import axis_set

        names = ["x", "y", "z", "t", "v"][:ndim]
        if ndim > 5:
            names.extend([f"d{i}" for i in range(6, ndim + 1)])
        return axis_set(ndim, names)

    def _nearest_anatomy(self) -> AxisSet3D:
        """Determine anatomical orientation from transformation matrix.

        This determines the closest standard anatomical orientation
        (like RAS, LPI, etc.) based on the transformation matrix.
        """
        from .axis import find_anatomy_3d

        if self.trans is None or self.trans.shape[0] < 3:
            # Default to LPI if no transformation matrix
            return find_anatomy_3d("LPI")

        # Extract the 3x3 rotation/scale part of the transformation matrix
        # Make a COPY to avoid modifying self.trans in-place
        mat33 = self.trans[:3, :3].copy()

        # Normalize each column (axis direction)
        for i in range(3):
            col = mat33[:, i]
            norm = np.linalg.norm(col)
            if norm > 0:
                mat33[:, i] = col / norm

        # Find the best matching anatomical orientation
        best_score = -1
        best_orient = "LPI"

        # Check all possible anatomical orientations
        orientations = [
            "RAS",
            "LAS",
            "RPS",
            "LPS",
            "RAI",
            "LAI",
            "RPI",
            "LPI",
            "RSA",
            "LSA",
            "RSP",
            "LSP",
            "RIA",
            "LIA",
            "RIP",
            "LIP",
            "ARS",
            "ALS",
            "PRS",
            "PLS",
            "ARI",
            "ALI",
            "PRI",
            "PLI",
            "ASR",
            "ASL",
            "PSR",
            "PSL",
            "AIR",
            "AIL",
            "PIR",
            "PIL",
            "SRA",
            "SLA",
            "SRP",
            "SLP",
            "IRA",
            "ILA",
            "IRP",
            "ILP",
            "SAR",
            "SAL",
            "SPR",
            "SPL",
            "IAR",
            "IAL",
            "IPR",
            "IPL",
        ]

        for orient in orientations:
            try:
                anat = find_anatomy_3d(orient)
                # Get the permutation matrix for this orientation
                anat_mat = np.zeros((3, 3))
                axis_attrs = ["i", "j", "k"]
                for i in range(3):
                    axis = getattr(anat, axis_attrs[i])
                    if isinstance(axis.direction, (list, tuple)):
                        anat_mat[:, i] = axis.direction
                    elif isinstance(axis.direction, (int, float)):
                        # Single value direction - expand to 3D vector
                        dir_vec = [0, 0, 0]
                        dir_vec[i] = axis.direction
                        anat_mat[:, i] = dir_vec
                    else:
                        anat_mat[:, i] = [0, 0, 0]
                        anat_mat[i, i] = 1

                # Compute similarity score (sum of absolute dot products)
                score = 0
                for i in range(3):
                    for j in range(3):
                        score += abs(np.dot(mat33[:, i], anat_mat[:, j]))

                if score > best_score:
                    best_score = score
                    best_orient = orient
            except:
                # Skip invalid orientations
                continue

        return find_anatomy_3d(best_orient)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dim)

    @property
    def nvoxels(self) -> int:
        """Total number of voxels."""
        return int(np.prod(self.dim))

    def bounds(self) -> np.ndarray:
        """Get spatial bounds of the image space.

        Returns
        -------
        ndarray
            2 x ndim array with min bounds in first row, max in second row"""
        # Grid corners in voxel space
        min_grid = np.zeros(self.ndim)
        max_grid = self.dim - 1

        # Convert to world coordinates
        min_coord = self.grid_to_coord(min_grid.reshape(1, -1))[0]
        max_coord = self.grid_to_coord(max_grid.reshape(1, -1))[0]

        # Stack as 2 x ndim array
        return np.vstack([min_coord, max_coord])

    def centroid(self) -> np.ndarray:
        """Get the spatial centroid of the image space.

        Returns
        -------
        ndarray
            Centroid coordinates in world space"""
        center_grid = (self.dim - 1) / 2.0
        return self.grid_to_coord(center_grid.reshape(1, -1))[0]

    def _spatial_affine(self) -> np.ndarray:
        """Return the 4x4 spatial sub-affine of an N-D ``trans`` matrix.

        Used to honor 3-D spatial queries on 4-D and 5-D NeuroSpaces
        (e.g. world-mm seeding into a 4-D BOLD).  Assumes the time / extra
        axes are separable from the spatial 3x3 block — the standard case
        for neuroimaging.
        """
        return self.trans[np.ix_([0, 1, 2, -1], [0, 1, 2, -1])]

    # Coordinate transformation methods
    def coord_to_grid(self, coords: np.ndarray) -> np.ndarray:
        """Convert real-world coordinates to grid indices.

        Parameters
        ----------
        coords : ndarray
            Real-world coordinates, shape ``(n, ndim)`` or ``(ndim,)``.
            On a 4-D or 5-D space, ``(n, 3)`` is also accepted and
            interpreted as a spatial-only query that returns 3-D grid
            indices.

        Returns
        -------
        ndarray
            Grid indices (0-based), matching the trailing input dim.
        """
        coords = np.atleast_2d(coords)
        spatial_query = self.ndim > 3 and coords.shape[1] == 3
        if not spatial_query and coords.shape[1] != self.ndim:
            raise ValueError(
                f"Coordinates must have {self.ndim} dimensions "
                f"(or 3 for spatial-only on a {self.ndim}-D space), "
                f"got {coords.shape[1]}"
            )

        if spatial_query:
            spatial = self._spatial_affine()
            homogeneous = np.column_stack([coords, np.ones(len(coords))])
            transformed = homogeneous @ np.linalg.inv(spatial).T
            return transformed[:, :3]

        if self.ndim <= 3:
            homogeneous = np.column_stack([coords, np.ones(len(coords))])
            transformed = homogeneous @ self.inverse.T
            return transformed[:, : self.ndim]

        return (coords - self.origin) / self.spacing

    def grid_to_coord(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid indices to real-world coordinates.

        Parameters
        ----------
        grid : ndarray
            Grid indices (0-based), shape ``(n, ndim)`` or ``(ndim,)``.
            On a 4-D or 5-D space, ``(n, 3)`` is also accepted and
            interpreted as a spatial-only query that returns 3-D world
            coordinates.

        Returns
        -------
        ndarray
            Real-world coordinates, matching the trailing input dim.
        """
        grid = np.atleast_2d(grid)
        spatial_query = self.ndim > 3 and grid.shape[1] == 3
        if not spatial_query and grid.shape[1] != self.ndim:
            raise ValueError(
                f"Grid indices must have {self.ndim} dimensions "
                f"(or 3 for spatial-only on a {self.ndim}-D space), "
                f"got {grid.shape[1]}"
            )

        if spatial_query:
            spatial = self._spatial_affine()
            homogeneous = np.column_stack([grid, np.ones(len(grid))])
            return (homogeneous @ spatial.T)[:, :3]

        if self.ndim <= 3:
            homogeneous = np.column_stack([grid, np.ones(len(grid))])
            transformed = homogeneous @ self.trans.T
            return transformed[:, : self.ndim]

        return grid * self.spacing + self.origin

    def grid_to_world(self, coords: VoxelCoord) -> WorldCoord:
        """Convert voxel-grid coordinates to world coordinates.

        Preserves input dimensionality: 1D input returns 1D output.
        """
        was_1d = np.ndim(coords) == 1
        result = self.grid_to_coord(np.atleast_2d(coords))
        return WorldCoord(result[0] if was_1d else result)

    def world_to_grid(self, coords: WorldCoord) -> VoxelCoord:
        """Convert world coordinates to nearest voxel-grid coordinates."""
        was_1d = np.ndim(coords) == 1
        result = np.round(self.coord_to_grid(np.atleast_2d(coords))).astype(np.int64)
        return VoxelCoord(result[0] if was_1d else result)

    def coord_to_index(self, coords: np.ndarray) -> np.ndarray:
        """Convert real-world coordinates to 1D indices.

        Parameters
        ----------
        coords : ndarray
            Real-world coordinates, shape (n, ndim)

        Returns
        -------
        ndarray
            1D indices (0-based)"""
        grid = self.coord_to_grid(coords)
        grid_rounded = np.round(grid).astype(int)
        return self.grid_to_index(grid_rounded)

    def index_to_coord(self, idx: Union[int, np.ndarray]) -> np.ndarray:
        """Convert 1D indices to real-world coordinates.

        Parameters
        ----------
        idx : int or ndarray
            1D indices (0-based)

        Returns
        -------
        ndarray
            Real-world coordinates"""
        grid = self.index_to_grid(idx)
        return self.grid_to_coord(grid)

    def grid_to_index(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid indices to 1D indices.

        Parameters
        ----------
        grid : ndarray
            Grid indices (0-based), shape (n, ndim)

        Returns
        -------
        ndarray
            1D indices (0-based)"""
        grid = np.atleast_2d(grid)
        if grid.shape[1] != self.ndim:
            raise ValueError(
                f"Grid indices must have {self.ndim} dimensions, got {grid.shape[1]}"
            )

        # Use C-style (row-major) ordering to match R
        indices = np.zeros(len(grid), dtype=int)
        for i in range(len(grid)):
            indices[i] = np.ravel_multi_index(grid[i], self.dim, order="F")
        return indices

    def index_to_grid(self, idx: Union[int, np.ndarray]) -> np.ndarray:
        """Convert 1D indices to grid indices.

        Parameters
        ----------
        idx : int or ndarray
            1D indices (0-based)

        Returns
        -------
        ndarray
            Grid indices (0-based), shape (n, ndim)"""
        idx = np.atleast_1d(idx)

        # Use Fortran-style (column-major) ordering to match R
        grid_tuple = np.unravel_index(idx, self.dim, order="F")
        return np.column_stack(grid_tuple)

    def grid_to_grid(
        self, source_grid: np.ndarray, target_space: "NeuroSpace"
    ) -> np.ndarray:
        """Transform grid coordinates from this space to another space.

        Parameters
        ----------
        source_grid : ndarray
            Grid indices in this space
        target_space : NeuroSpace
            Target space to transform to

        Returns
        -------
        ndarray
            Grid indices in target space"""
        # Convert to world coordinates in source space
        coords = self.grid_to_coord(source_grid)
        # Convert to grid in target space
        return target_space.coord_to_grid(coords)

    # Dimension manipulation methods
    def dim_of(self, axis: Union[int, str, NamedAxis]) -> int:
        """Get the length of a given dimension.

        Parameters
        ----------
        axis : int, str, or NamedAxis
            Axis to query

        Returns
        -------
        int
            Length of the dimension"""
        if isinstance(axis, int):
            if axis < 0 or axis >= self.ndim:
                raise ValueError(
                    f"Axis index {axis} out of range for {self.ndim}D space"
                )
            return self.dim[axis]
        elif isinstance(axis, str):
            axis_idx = self.which_dim(axis)
            return self.dim[axis_idx]
        elif isinstance(axis, NamedAxis):
            axis_idx = self.which_dim(axis.axis)
            return self.dim[axis_idx]
        else:
            raise TypeError(f"axis must be int, str, or NamedAxis, got {type(axis)}")

    def which_dim(self, axis: Union[str, NamedAxis]) -> int:
        """Get the index of a dimension by axis name.

        Parameters
        ----------
        axis : str or NamedAxis
            Axis to find

        Returns
        -------
        int
            0-based index of the axis"""
        if isinstance(axis, NamedAxis):
            axis = axis.axis

        names = axis_names(self.axes)
        try:
            return names.index(axis)
        except ValueError:
            raise ValueError(f"Axis '{axis}' not found in space")

    def get_subspace(self, dims) -> "NeuroSpace":
        """Return a subspace containing only the specified dimensions.

        Parameters
        ----------
        dims : iterable of int
            0-based indices of dimensions to keep.

        Returns
        -------
        NeuroSpace
            New space with only the requested dimensions.
        """
        dims = list(dims)
        if len(dims) == 0:
            raise ValueError("dims must include at least one dimension")
        for d in dims:
            if d < 0 or d >= self.ndim:
                raise ValueError(
                    f"Dimension index {d} out of range for {self.ndim}D space"
                )

        new_dim = self.dim[dims]
        new_spacing = self.spacing[dims]
        new_origin = self.origin[dims]
        ndim_new = len(dims)

        if ndim_new <= 3:
            new_trans = np.eye(4)
            new_trans[:ndim_new, :ndim_new] = self.trans[np.ix_(dims, dims)]
            new_trans[:ndim_new, 3] = self.trans[dims, -1]
        else:
            new_trans = np.eye(ndim_new + 1)
            new_trans[:ndim_new, :ndim_new] = self.trans[np.ix_(dims, dims)]
            new_trans[:ndim_new, -1] = self.trans[dims, -1]

        return NeuroSpace(new_dim, new_spacing, new_origin, trans=new_trans)

    def drop_dim(self, dimnum: int) -> "NeuroSpace":
        """Remove a dimension from the space.

        Parameters
        ----------
        dimnum : int
            0-based index of dimension to drop

        Returns
        -------
        NeuroSpace
            New space with dimension removed"""
        if self.ndim < 2:
            raise ValueError(
                "Cannot drop dimension from space with less than 2 dimensions"
            )

        if dimnum < 0 or dimnum >= self.ndim:
            raise ValueError(f"dimnum {dimnum} out of range for {self.ndim}D space")

        # Create new arrays with dimension removed
        new_dim = np.delete(self.dim, dimnum)
        new_origin = np.delete(self.origin, dimnum)
        new_spacing = np.delete(self.spacing, dimnum)

        new_axes = axis_set(
            len(new_dim), [ax for i, ax in enumerate(list(self.axes)) if i != dimnum]
        )

        keep = [i for i in range(self.ndim) if i != dimnum]

        if len(new_dim) <= 3:
            new_trans = np.eye(4)
            new_trans[: len(new_dim), : len(new_dim)] = self.trans[np.ix_(keep, keep)]
            new_trans[: len(new_dim), 3] = self.trans[keep, -1]
        else:
            new_trans = np.eye(len(new_dim) + 1)
            new_trans[: len(new_dim), : len(new_dim)] = self.trans[np.ix_(keep, keep)]
            new_trans[: len(new_dim), -1] = self.trans[keep, -1]

        return NeuroSpace(new_dim, new_spacing, new_origin, new_axes, new_trans)

    def add_dim(self, n: int = 1, size: int = 1) -> "NeuroSpace":
        """Add a dimension to the space.

        Parameters
        ----------
        n : int, optional
            Number of dimensions to add (default 1)
        size : int, optional
            Size of the new dimension(s) (default 1)

        Returns
        -------
        NeuroSpace
            New space with added dimension(s)"""
        if n <= 0:
            raise ValueError("n must be positive")

        # Extend arrays
        new_dim = np.append(self.dim, [size] * n)
        new_origin = np.append(self.origin, [0.0] * n)
        new_spacing = np.append(self.spacing, [1.0] * n)

        # Preserve existing axes and append labeled placeholder axes for each added dimension.
        start = len(self.axes) + 1
        extra_axes = [NamedAxis(f"v{start + i}", 1) for i in range(n)]
        new_axes = axis_set(len(new_dim), list(self.axes) + extra_axes)

        # Preserve affine transform behavior:
        # - 3D and below use fixed 4x4 transforms.
        # - 4D+ use (ndim+1)x(ndim+1) transforms, extending with identity rows/cols.
        source_dim = self.trans.shape[0] - 1
        if new_dim.size <= 3:
            new_trans = np.eye(4)
            keep = min(source_dim, new_dim.size)
            new_trans[:keep, :keep] = self.trans[:keep, :keep]
            new_trans[:keep, 3] = self.trans[:keep, -1]
            for i in range(keep, new_dim.size):
                new_trans[i, i] = new_spacing[i]
        else:
            new_trans = np.eye(len(new_dim) + 1)
            keep = min(source_dim, len(new_dim))
            new_trans[:keep, :keep] = self.trans[:keep, :keep]
            new_trans[:keep, -1] = self.trans[:keep, -1]
            for i in range(source_dim, len(new_dim)):
                new_trans[i, i] = new_spacing[i]

        return NeuroSpace(new_dim, new_spacing, new_origin, new_axes, new_trans)

    def __repr__(self):
        """String representation matching R's show method."""
        axis_str = ", ".join(axis_names(self.axes))
        # Convert numpy types to Python types for cleaner display
        dim_tuple = tuple(int(d) for d in self.dim)
        origin_tuple = tuple(float(o) for o in self.origin)
        spacing_tuple = tuple(float(s) for s in self.spacing)
        return (
            f"NeuroSpace(\n"
            f"  dim     : {dim_tuple}\n"
            f"  origin  : {origin_tuple}\n"
            f"  spacing : {spacing_tuple}\n"
            f"  axes    : {axis_str}\n"
            f"  nvoxels : {self.nvoxels}\n"
            f")"
        )

    def __eq__(self, other):
        """Check equality with another NeuroSpace."""
        if not isinstance(other, NeuroSpace):
            return False

        return (
            np.array_equal(self.dim, other.dim)
            and np.array_equal(self.origin, other.origin)
            and np.array_equal(self.spacing, other.spacing)
            and np.array_equal(self.trans, other.trans)
        )

    def __hash__(self):
        """Hash the immutable spatial contract."""
        return hash(
            (
                tuple(int(d) for d in self.dim),
                tuple(float(o) for o in self.origin),
                tuple(float(s) for s in self.spacing),
                self.trans.shape,
                self.trans.dtype.str,
                self.trans.tobytes(),
            )
        )


def neurospace(
    dim: Union[Tuple[int, ...], List[int]],
    spacing: Optional[Union[Tuple[float, ...], List[float]]] = None,
    origin: Optional[Union[Tuple[float, ...], List[float]]] = None,
    axes: Optional[Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D]] = None,
    trans: Optional[np.ndarray] = None,
) -> NeuroSpace:
    """Construct a :class:`NeuroSpace` from spacing/origin/axes or a 4x4 affine."""
    return NeuroSpace(dim, spacing, origin, axes, trans)
